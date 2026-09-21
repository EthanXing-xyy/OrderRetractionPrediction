# -*- coding: utf-8 -*-
"""s06a_build_seq.py —— 序列数据构建（数据契约 §5）

把「逐条医嘱一行」的表格数据（labels.parquet + features.parquet + splits.parquet）
变成 causal Transformer 用的**定长块**：

    按 hadm_id 分组 → 组内按 (ordertime, poe_seq) 升序 → 切成定长 seq_len=256 的块
    不足 256 的位置补 padding（y13 = -1、key = -1、tokens = 0、num = 0）
    一次住院超过 256 条医嘱就切成多块，**不截断、不丢数据**

产物（契约 §5 + 一个额外的 key.npy）：

    data/seq/<split>_tokens.npy      int16/int32  (n_block, 256, n_cat_field)
    data/seq/<split>_num.npy         float32      (n_block, 256, n_num_field)
    data/seq/<split>_y13.npy         int8         (n_block, 256)   -1 = padding / 不参与损失
    data/seq/<split>_censorbin.npy   int8         (n_block, 256)   -1 = padding
    data/seq/<split>_key.npy         int64        (n_block, 256)   -1 = padding  ← 额外，用来把预测写回契约 §7
    data/seq/meta.json               词表大小、字段顺序、标准化均值方差、块数

关键设计（为什么这么写）
------------------------
1. **内存**：5221 万行、本机可用约 6 GB。做法是
   ① 逐列读 parquet（不把整表读进来）；
   ② 每个 split 再按 `hadm_id % n_shard` 分片，一片一片处理（一个 hadm 的块必在同一片内）；
   ③ 输出用 `open_memmap` 直接写盘，绝不在内存里拼完整数组；
   ④ 排序用 `np.lexsort` + `np.searchsorted` 找组边界，**不用 groupby.apply**。
2. **泄漏**：数值特征的标准化均值/方差**只用 train 算**，写进 meta.json，val/test 复用。
3. **词表 id 偏移**：`token = code + 2`，`0 = PAD`，`1 = MISSING`（原始 code 为 -1）。
   Embedding 用 `padding_idx=0`。词表大小 = 全局最大 code + 3。
   任一字段超过 32767 时整个 tokens 数组改用 int32，并在 meta.json 写明。
4. **in_cohort**：非队列行仍然保留在序列里**当上下文**，但 `y13` 置 -1（不进损失）。
   推理时是否出预测由 `s06_seq_model.py` 回查 labels.parquet 的 `in_cohort` 决定
   （契约 §7：只在 in_cohort 的行上出预测）。`y13 >= 0` **不能**当作 in_cohort 用——
   cause==3（provider 缺失）的行 y13 = -1 却仍在队列里，H1 二分类要用它。

隐私：本脚本只往 metrics / 日志里写**计数与聚合量**，绝不打印任何记录级取值。
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pyarrow.types as pat
from numpy.lib.format import open_memmap

import common as C

# --------------------------------------------------------------------------- 字段分类
# 契约 §3 列出的特征。类别字段走 Embedding，数值字段走小 MLP。
# 契约 §3 的分组 + s03 实际产出的列（2026-09-21 逐字核过 features.parquet 的 schema）。
# 类别 = 走 Embedding；数值 = 走小 MLP。分错的代价很实在：
#   把 provider 编码当数值 → log1p(编号) 是没有意义的量；
#   把「至今第几条医嘱」当类别 → 建一个几万行的 Embedding。
CAT_KNOWN = {
    "f_order_type", "f_order_subtype", "f_transaction_type",
    "f_hour", "f_dow",
    "f_admission_type", "f_admission_location", "f_insurance",
    "f_careunit", "f_curr_service",
    "f_gender", "f_age_bin", "f_anchor_year_group",
    "f_prov_code",            # 开单者账号的类别编码（取值绝不输出，只当模型内部特征，见契约 §1 注）
    "f_prov_n_prior_bin",     # 契约 §3 写的「f_prov_n_prior（分档）」= s03 的 _bin 列
    "f_n_detail",             # 本单有几个 detail 字段，0~十几，当类别
}
NUM_KNOWN = {
    "f_since_admit_h", "f_prior_disc_rate", "f_dt_prev_order_h",
    "f_prov_prior_disc_rate", "f_prov_prior_self_rate",
    "f_n_prior_orders",       # 计数（int32，最大上千）
    "f_prior_disc_known",     # 计数，契约 §3 未登记，s03 实际产出
    "f_prov_n_prior",         # 计数**原值**（未分档，int32，可到十万级）
}
# 长尾非负量先做 log1p 再标准化（时长、计数），否则 z-score 之后仍然是长尾
NUM_LOG1P = {"f_since_admit_h", "f_dt_prev_order_h",
             "f_n_prior_orders", "f_prior_disc_known", "f_prov_n_prior"}

# 兜底规则：契约与上面都没登记的列，**int32/int64 且取值范围大**才改判数值。
# s03 的惯例是「类别编码用 int8/int16、计数用 int32」，所以这条按位宽判不会误伤 f_prov_code(int16)。
AUTO_NUM_MAX = 2000

PAD_ID, MISSING_ID, ID_OFFSET = 0, 1, 2
SPLIT_NAMES = {0: "train", 1: "val", 2: "test"}


# --------------------------------------------------------------------------- 工具
def stamp(path: Path) -> tuple:
    """文件指纹（大小 + 修改时间）。用来确认整个构建过程读的是**同一版** features。"""
    st = path.stat()
    return (st.st_size, st.st_mtime_ns)


def read_col(path: Path, name: str) -> np.ndarray:
    """从 parquet 单独读一列成 numpy（省内存）。整数列 null 填 -1、布尔列填 False、浮点列保留 NaN。"""
    a = pq.read_table(path, columns=[name])[name]
    if pat.is_floating(a.type):
        return a.to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
    if pat.is_boolean(a.type):
        return pc.fill_null(a, False).to_numpy(zero_copy_only=False)
    return pc.fill_null(a, -1).to_numpy(zero_copy_only=False)


def classify_fields(path: Path):
    """按契约 §3 把 features.parquet 的 f_* 列分成类别 / 数值两组（未登记的列按 dtype 兜底）。"""
    sch = pq.ParquetFile(path).schema_arrow
    cat, num, unknown = [], [], []
    for f in sch:
        n = f.name
        if not n.startswith("f_"):
            continue
        if n in CAT_KNOWN or n.startswith("f_detail_"):
            cat.append(n)
        elif n in NUM_KNOWN:
            num.append(n)
        else:                                          # 契约没登记的新列：按 dtype 兜底
            unknown.append(n)
            (num if pat.is_floating(f.type) else cat).append(n)
    return sorted(cat), sorted(num), sorted(unknown)


def block_layout(hadm_sorted: np.ndarray, L: int):
    """给**已按 (hadm, 时间) 排好序**的行，算出每个块在排序后数组里的起点与有效长度。

    返回 (blk_start, blk_len)，都是一维 int64。完全向量化，无 python 循环、无 groupby。
    """
    n = hadm_sorted.size
    if n == 0:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    chg = np.empty(n + 1, bool)
    chg[0] = True
    chg[n] = True
    if n > 1:
        np.not_equal(hadm_sorted[1:], hadm_sorted[:-1], out=chg[1:n])
    bounds = np.flatnonzero(chg)                       # 长度 G+1
    sizes = np.diff(bounds).astype(np.int64)           # 每次住院的医嘱条数
    nblk = -(-sizes // L)                              # ceil
    grp = np.repeat(np.arange(sizes.size, dtype=np.int64), nblk)
    off = np.arange(int(nblk.sum()), dtype=np.int64) - np.repeat(np.cumsum(nblk) - nblk, nblk)
    blk_start = bounds[:-1].astype(np.int64)[grp] + off * L
    blk_len = np.minimum(L, sizes[grp] - off * L)
    return blk_start, blk_len


# --------------------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser(description="构建 causal Transformer 的定长序列块（契约 §5）")
    ap.add_argument("--seq-len", type=int, default=None, help="默认读 config.transformer.seq_len")
    ap.add_argument("--splits", default="train,val,test")
    ap.add_argument("--shards", type=int, default=0, help="0=按行数自动（每片约 600 万行）")
    ap.add_argument("--block-chunk", type=int, default=8000, help="一次往 memmap 写多少个块")
    ap.add_argument("--out", default=None, help="输出目录，默认 data/seq")
    ap.add_argument("--tag", default="", help="日志/指标文件名后缀（合成数据自测用）")
    ap.add_argument("--no-metrics", action="store_true", help="不写 results/metrics（合成数据自测用）")
    args = ap.parse_args()

    step = "s06a_build_seq" + (("_" + args.tag) if args.tag else "")
    t0, _ = C.start_log(step)

    L = args.seq_len or int(C.cfg()["transformer"]["seq_len"])
    outdir = Path(args.out) if args.out else (C.DATA / "seq")
    outdir.mkdir(parents=True, exist_ok=True)
    p_lab = C.DATA / "labels.parquet"
    p_feat = C.DATA / "features.parquet"
    p_spl = C.DATA / "splits.parquet"
    for p in (p_lab, p_feat, p_spl):
        if not p.exists():
            raise FileNotFoundError(f"缺少输入 {p}（s02/s03/s04 还没跑完？）")
    print(f"seq_len={L}  输出目录={outdir}")
    # 本脚本要把 features.parquet 逐列读上百遍（每个分片一遍）。如果构建途中 s03 把它重写了，
    # 前几片用旧值、后几片用新值，**产物会静默地混版**。所以开头记指纹，每片开工前核一次。
    feat_stamp = stamp(p_feat)
    print(f"features 指纹 size={feat_stamp[0]} mtime_ns={feat_stamp[1]}")

    def check_feat():
        if stamp(p_feat) != feat_stamp:
            raise RuntimeError(
                "features.parquet 在构建过程中被改写了（s03 又跑了一遍？）。"
                "已写出的分片用的是旧版特征，会混版——本次构建作废，请等新文件稳定后整体重跑。")

    # ---------------------------------------------------------------- 1. labels 与切分
    print("[1/5] 读 labels 与 splits …")
    lkey = read_col(p_lab, "key").astype(np.int64, copy=False)
    n_rows = lkey.size
    hadm = read_col(p_lab, "hadm_id").astype(np.int32, copy=False)      # 最大 2999 万，int32 够
    ordertime = read_col(p_lab, "ordertime").astype(np.int64, copy=False)
    y13 = read_col(p_lab, "y13").astype(np.int8, copy=False)
    censorbin = np.clip(read_col(p_lab, "censor_bin"), -1, 5).astype(np.int8)
    in_cohort = read_col(p_lab, "in_cohort").astype(bool, copy=False)

    ord_min = int(ordertime.min())
    ord_span = int(ordertime.max()) - ord_min
    if ord_span >= (1 << 39):        # 复合键 = (ordertime-min)<<24 | poe_seq，必须落在 int64 内
        raise RuntimeError(f"ordertime 跨度 {ord_span} 过大，无法压成 int64 复合排序键")

    subj = lkey >> 24
    ss = read_col(p_spl, "subject_id").astype(np.int64, copy=False)
    sp = read_col(p_spl, "split").astype(np.int8, copy=False)
    o = np.argsort(ss, kind="stable")
    ss, sp = ss[o], sp[o]
    del o
    posc = np.clip(np.searchsorted(ss, subj), 0, ss.size - 1)
    found = ss[posc] == subj
    split_of_row = np.where(found, sp[posc], np.int8(-1)).astype(np.int8)
    del posc, found, subj, ss, sp

    # 不可用行：无 hadm_id / 不在切分表里 → 整行不进序列
    n_in_cohort = int(in_cohort.sum())
    y13 = np.where(in_cohort, y13, np.int8(-1)).astype(np.int8)   # 非队列行只当上下文，不进损失
    del in_cohort
    split_of_row = np.where((hadm > 0) & (split_of_row >= 0), split_of_row, np.int8(-1)).astype(np.int8)
    print(f"  labels 行数={n_rows}  in_cohort={n_in_cohort}  "
          f"进入序列={int((split_of_row >= 0).sum())}")

    # ---------------------------------------------------------------- 2. 特征对齐
    print("[2/5] 对齐 features（按 key）…")
    fkey = read_col(p_feat, "key").astype(np.int64, copy=False)
    if fkey.size == n_rows and np.array_equal(fkey, lkey):
        frow = None                                            # 行序完全一致，免映射
        print("  features 与 labels 行序一致，直接按行号对齐")
    else:
        fo = np.argsort(fkey, kind="stable")
        fs = fkey[fo]
        p_ = np.clip(np.searchsorted(fs, lkey), 0, fs.size - 1)
        ok = fs[p_] == lkey
        frow = np.where(ok, fo[p_], 0).astype(np.int64)
        print(f"  features 行序不同，已建映射；labels 中无对应特征的行={int((~ok).sum())}（剔出序列）")
        split_of_row = np.where(ok, split_of_row, np.int8(-1)).astype(np.int8)
        del fo, fs, p_, ok
    del fkey

    cat_fields, num_fields, unknown = classify_fields(p_feat)
    if not cat_fields and not num_fields:
        raise RuntimeError("features.parquet 里找不到任何 f_* 列")
    if unknown:
        print(f"  ⚠ 契约未登记的特征列（按 dtype 兜底分类）：{unknown}")
        # 兜底进了类别组的**宽整数**列，如果取值范围很大，说明是计数不是类别，改走数值通道
        wide = {f.name for f in pq.ParquetFile(p_feat).schema_arrow
                if pat.is_integer(f.type) and f.type.bit_width >= 32}
        moved = []
        for f in [x for x in unknown if x in cat_fields and x in wide]:
            v = read_col(p_feat, f)
            mx, mn = int(v.max()) if v.size else 0, int(v.min()) if v.size else 0
            del v
            if mx > AUTO_NUM_MAX:
                cat_fields.remove(f)
                num_fields.append(f)
                if mn >= 0:
                    NUM_LOG1P.add(f)
                moved.append((f, mx))
        if moved:
            num_fields.sort()
            print(f"  → 取值范围过大，改判为数值（log1p+标准化）：{moved}")
    print(f"  类别字段 {len(cat_fields)} 个：{cat_fields}")
    print(f"  数值字段 {len(num_fields)} 个：{num_fields}")

    # ---------------------------------------------------------------- 3. 词表 + train 标准化参数
    print("[3/5] 统计词表大小与 train 标准化参数（均值方差**只用 train**）…")
    train_mask = split_of_row == 0
    if not train_mask.any():
        raise RuntimeError("train 集为空，检查 splits.parquet")
    trow = np.flatnonzero(train_mask) if frow is None else frow[train_mask]

    cat_vocab = {}
    for f in cat_fields:
        v = read_col(p_feat, f)
        cat_vocab[f] = int(max(int(v.max()) if v.size else -1, -1)) + 1 + ID_OFFSET
        del v
    vmax = max(cat_vocab.values()) if cat_vocab else 0
    tok_dtype = np.int16 if vmax <= 32767 else np.int32
    big = {k: v for k, v in cat_vocab.items() if v > 20000}
    if big:
        print(f"  ⚠ 高基数类别字段（Embedding 参数会很大）：{big}")
    print(f"  tokens dtype = {np.dtype(tok_dtype).name}  最大词表 = {vmax}")

    num_mean, num_std, num_nanfrac = [], [], []
    for f in num_fields:
        tv = read_col(p_feat, f)[trow]
        if f in NUM_LOG1P:
            tv = np.log1p(np.maximum(tv, np.float32(0.0)))
        nf = float(np.isnan(tv).mean()) if tv.size else 0.0
        m = float(np.nanmean(tv)) if (tv.size and nf < 1.0) else 0.0
        s = float(np.nanstd(tv)) if (tv.size and nf < 1.0) else 1.0
        num_mean.append(m if np.isfinite(m) else 0.0)
        num_std.append(s if (np.isfinite(s) and s > 1e-8) else 1.0)
        num_nanfrac.append(nf)
        del tv
    del trow, train_mask
    # 缺失率显著的数值字段额外给一个 0/1 指示通道
    #（缺失本身有信息，例如「本次住院第一条医嘱没有上一条医嘱的间隔」）
    ind_fields = [f for f, nf in zip(num_fields, num_nanfrac) if nf > 1e-3]
    ind_pos = [num_fields.index(f) for f in ind_fields]
    num_out = list(num_fields) + [f"m_{f}" for f in ind_fields]
    n_cat, n_num, n_base = len(cat_fields), len(num_out), len(num_fields)
    print(f"  数值缺失指示通道 {len(ind_fields)} 个：{ind_fields}")
    print(f"  n_cat_field={n_cat}  n_num_field={n_num}")
    mean_a = np.asarray(num_mean, np.float32)
    std_a = np.asarray(num_std, np.float32)

    # ---------------------------------------------------------------- 4. 逐 split 构块
    meta_splits = {}
    want = [s.strip() for s in args.splits.split(",") if s.strip()]
    for sid, sname in SPLIT_NAMES.items():
        if sname not in want:
            continue
        in_split = split_of_row == sid
        n_s = int(in_split.sum())
        if n_s == 0:
            print(f"[4/5] {sname}: 无数据，跳过")
            continue
        # 块数由「每次住院的医嘱条数」决定，必须先算出来才能开 memmap
        _, cnt = np.unique(hadm[in_split], return_counts=True)
        n_block = int((-(-cnt.astype(np.int64) // L)).sum())
        n_hadm = int(cnt.size)
        del cnt

        n_shard = args.shards or max(1, int(np.ceil(n_s / 6_000_000)))
        print(f"[4/5] {sname}: 行数={n_s}  住院数={n_hadm}  块数={n_block}  分片={n_shard}")

        mm = {
            "tokens": open_memmap(outdir / f"{sname}_tokens.npy", mode="w+",
                                  dtype=tok_dtype, shape=(n_block, L, n_cat)),
            "num": open_memmap(outdir / f"{sname}_num.npy", mode="w+",
                               dtype=np.float32, shape=(n_block, L, n_num)),
            "y13": open_memmap(outdir / f"{sname}_y13.npy", mode="w+",
                               dtype=np.int8, shape=(n_block, L)),
            "censorbin": open_memmap(outdir / f"{sname}_censorbin.npy", mode="w+",
                                     dtype=np.int8, shape=(n_block, L)),
            "key": open_memmap(outdir / f"{sname}_key.npy", mode="w+",
                               dtype=np.int64, shape=(n_block, L)),
        }

        cursor, n_valid = 0, 0
        shard_of_row = (hadm % np.int32(n_shard)).astype(np.int8) if n_shard > 1 else None
        for k in range(n_shard):
            sel = np.flatnonzero(in_split if n_shard == 1 else (in_split & (shard_of_row == k)))
            if sel.size == 0:
                continue
            check_feat()
            ts = time.time()

            # --- 片内排序：主键 hadm，次键 (ordertime, poe_seq)
            h = hadm[sel]
            # 复合次键：ordertime 升序，ordertime 相同时按 poe_seq 升序（key = subject_id*2^24 + poe_seq）
            sec = ((ordertime[sel] - ord_min) << 24) | (lkey[sel] & ((1 << 24) - 1))
            loc = np.lexsort((sec, h))               # 片内局部下标，已排好序
            del sec
            blk_start, blk_len = block_layout(h[loc], L)
            nb = int(blk_start.size)
            del h

            # --- 片内特征矩阵（只为这一片的行建，内存可控）
            frows = sel if frow is None else frow[sel]
            loccat = np.zeros((sel.size, n_cat), tok_dtype)
            for j, f in enumerate(cat_fields):
                v = read_col(p_feat, f)[frows].astype(np.int32, copy=False)
                loccat[:, j] = np.where(v < 0, np.int32(MISSING_ID), v + np.int32(ID_OFFSET))
                del v
            locnum = np.zeros((sel.size, n_num), np.float32)
            for j, f in enumerate(num_fields):
                v = read_col(p_feat, f)[frows].astype(np.float32, copy=False)
                if f in NUM_LOG1P:
                    v = np.log1p(np.maximum(v, np.float32(0.0)))
                locnum[:, j] = (v - mean_a[j]) / std_a[j]
                del v
            for jj, j in enumerate(ind_pos):                     # 指示通道要在 nan_to_num **之前**算
                locnum[:, n_base + jj] = np.isnan(locnum[:, j]).astype(np.float32)
            np.nan_to_num(locnum, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            del frows

            loc_y13, loc_cb, loc_key = y13[sel], censorbin[sel], lkey[sel]
            del sel

            # --- 按块 chunk 连续写盘
            ar = np.arange(L, dtype=np.int64)
            step_b = max(1, args.block_chunk)
            for b0 in range(0, nb, step_b):
                b1 = min(nb, b0 + step_b)
                nbc = b1 - b0
                g = blk_start[b0:b1, None] + ar[None, :]         # (nbc, L) 排序后下标
                valid = ar[None, :] < blk_len[b0:b1, None]
                take = loc[np.where(valid, g, 0).ravel()]        # → 片内原始行下标
                flat = valid.ravel()

                tk = loccat[take].reshape(nbc, L, n_cat)
                tk[~valid] = PAD_ID
                mm["tokens"][cursor + b0:cursor + b1] = tk
                del tk
                nu = locnum[take].reshape(nbc, L, n_num)
                nu[~valid] = 0.0
                mm["num"][cursor + b0:cursor + b1] = nu
                del nu
                yy = np.where(flat, loc_y13[take], np.int8(-1)).reshape(nbc, L)
                mm["y13"][cursor + b0:cursor + b1] = yy
                n_valid += int((yy >= 0).sum())
                mm["censorbin"][cursor + b0:cursor + b1] = \
                    np.where(flat, loc_cb[take], np.int8(-1)).reshape(nbc, L)
                mm["key"][cursor + b0:cursor + b1] = \
                    np.where(flat, loc_key[take], np.int64(-1)).reshape(nbc, L)
                del g, valid, take, flat, yy

            cursor += nb
            del loccat, locnum, loc_y13, loc_cb, loc_key, loc, blk_start, blk_len
            print(f"  片 {k + 1}/{n_shard}: 块 {nb}  累计 {cursor}/{n_block}  用时 {time.time() - ts:.1f}s")

        if cursor != n_block:
            raise RuntimeError(f"{sname} 块数对不上：写了 {cursor}，预期 {n_block}")
        for v in mm.values():
            v.flush()
        del mm, shard_of_row, in_split

        n_pos = n_block * L
        meta_splits[sname] = {
            "n_row": n_s, "n_hadm": n_hadm, "n_block": n_block, "n_pos": n_pos,
            "n_valid_y13": int(n_valid),
            "pad_pct": round(100.0 * (1 - n_s / n_pos), 2) if n_pos else None,
            "valid_pct": round(100.0 * n_valid / n_pos, 2) if n_pos else None,
        }
        print(f"  {sname} 完成：块={n_block} 总位置={n_pos} 有效位置(y13>=0)={n_valid} "
              f"padding={meta_splits[sname]['pad_pct']}%")

    # ---------------------------------------------------------------- 5. meta.json
    meta = {
        "seq_len": L,
        "pad_id": PAD_ID, "missing_id": MISSING_ID, "id_offset": ID_OFFSET,
        "tokens_dtype": np.dtype(tok_dtype).name,
        "cat_fields": cat_fields,
        "cat_vocab": {k: int(v) for k, v in cat_vocab.items()},
        "cat_vocab_list": [int(cat_vocab[f]) for f in cat_fields],
        "num_fields": num_out,
        "num_base_fields": num_fields,
        "num_log1p": sorted(NUM_LOG1P & set(num_fields)),
        "num_mean": [float(x) for x in num_mean],
        "num_std": [float(x) for x in num_std],
        "num_nan_frac": [round(float(x), 6) for x in num_nanfrac],
        "num_indicator_fields": ind_fields,
        "n_cat_field": n_cat, "n_num_field": n_num,
        "splits": meta_splits,
        "note": ("num 的均值方差只用 train 计算；token = code + 2，0=PAD、1=MISSING；"
                 "y13=-1 表示 padding 或不参与竞争风险损失（含 in_cohort=False 与 cause==3）；"
                 "key=-1 表示 padding。是否出预测由 s06 回查 labels.in_cohort 决定。"),
    }
    check_feat()                      # 全程读的是同一版 features，才允许写 meta
    meta["features_stamp"] = {"size": feat_stamp[0], "mtime_ns": feat_stamp[1]}
    (outdir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"→ meta 已写 {outdir / 'meta.json'}")

    if not args.no_metrics:
        C.save_metrics(step, {
            "seq_len": L, "n_cat_field": n_cat, "n_num_field": n_num,
            "tokens_dtype": np.dtype(tok_dtype).name,
            "cat_vocab_total": int(sum(cat_vocab.values())),
            "cat_vocab_max": int(vmax),
            "n_num_indicator": len(ind_fields),
            "splits": meta_splits,
        }, t0)
        C.append_runlog(step, "序列块构建：" + "；".join(
            f"{k} 块={v['n_block']} 有效位置={v['n_valid_y13']} padding={v['pad_pct']}%"
            for k, v in meta_splits.items()))
    print(f"[完成] 用时 {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
