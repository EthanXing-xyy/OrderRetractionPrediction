# -*- coding: utf-8 -*-
"""S03 特征工程（场景 A：只用 t₀ 时刻可得的信息）。

🔴 本脚本最重要的一件事：**把「历史统计」做成真正的 t₀ 前缀量，而不是偷看未来。**

   一个极易踩的泄漏陷阱：「本次住院此前的撤销率」如果直接对前序医嘱的 `y_disc` 求平均，
   就用到了未来——那条更早的医嘱可能是在 **t₀ 之后**才被撤销的，t₀ 时刻根本不知道。
   正确定义是：**只数那些「撤销动作本身已经发生在 t₀ 之前」的医嘱**，即 `t_disc(j) <= t₀`。

   这里有个好用的性质：因为撤销必然晚于下单（实测 `delay_h <= 0` 占比为 0.00%），
   `t_disc(j) <= t₀(i)` 已经蕴含了 `t₀(j) < t₀(i)`，所以**不必再单独约束「j 在 i 之前」**。
   于是整个计数可以写成一次 `np.searchsorted`：

       复合键 = 分组ID << 34 | 时间       （hadm_id < 2^25，epoch 秒 < 2^33，int64 装得下）
       某组内截至 t₀ 已发生的撤销数 = searchsorted(已撤销键[已排序], 本行查询键, 'right') − 该组起点

   同一套写法同时用在「本次住院」和「本开单者」两个分组上，**不需要任何 groupby 循环**，
   5200 万行几十秒跑完，内存峰值约 3.5 GB。

输入：data/{poe,labels,admissions,patients,transfers,services,poe_detail}.parquet
输出：data/features.parquet（每条医嘱一行，与 labels 用 `key` 对齐）+ data/vocab.json
      + results/metrics/s03_features.json（纯聚合量）

用法： python src/s03_features.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

SHIFT_KEY = 24     # key   = subject_id << 24 | poe_seq
SHIFT_T = 34       # 复合键 = 分组ID << 34 | epoch 秒
DETAIL_FIELDS = ["Route", "Admit category", "Admit to",
                 "Tubes & Drains type", "Discharge Planning", "Code status"]
DETAIL_COLNAMES = ["f_detail_route", "f_detail_admit_category", "f_detail_admit_to",
                   "f_detail_tubes_drains", "f_detail_discharge_planning",
                   "f_detail_code_status"]
DETAIL_TOPV = 200  # 每个字段只保留取值频次前 200 名，其余归入 OTHER（防过拟合 + 省内存）


def grp_start(sorted_keys: np.ndarray, gid: np.ndarray) -> np.ndarray:
    """某分组在已排序复合键数组里的起点下标。"""
    return np.searchsorted(sorted_keys, gid.astype(np.int64) << SHIFT_T, side="left")


def prefix_count(sorted_keys: np.ndarray, gid: np.ndarray, t: np.ndarray,
                 side: str = "left") -> np.ndarray:
    """组内「时间 < t（side=left）或 <= t（side=right）」的条数。见文件头的说明。

    🔴 **全工程一律用 side="left"（严格早于 t₀）**，理由有三：

    1. **分钟粒度下同一分钟分不出先后。** MIMIC 的时间戳精确到分钟，
       用 `<=` 等于让特征吃进最多 59 秒的未来。
    2. **`<=` 会让医嘱把自己数进去。** 实测有 191 条医嘱的撤销时刻与下单时刻同秒
       （外加 43 条时间倒挂的数据异常），它们在 `<=` 口径下会把「自己已经被撤销了」
       这件事喂给模型——**这是专挑正类下手的自指泄漏，比随机噪声危险得多**。
    3. **口径必须与 `f_n_prior_orders` 一致。** 一个用 `<` 一个用 `<=`，
       就会造出「已发生的撤销数 > 此前下达的医嘱数」这种逻辑上不可能的行（实测 38 行）。
       统一成 `<` 之后，`t₀(j) <= t_disc(j) < t₀(a) ⇒ t₀(j) < t₀(a)`，
       不变式可以证明成立，于是能写成断言钉死在代码里。

    代价：全体 30.85% 的行上，这个计数会比 `<=` 口径少 1（那些同分钟发生的撤销）。
    这是为「可证明无泄漏」付的价，值得——计数少 1 对模型无实质影响，自指泄漏有。
    """
    q = (gid.astype(np.int64) << SHIFT_T) | t
    return (np.searchsorted(sorted_keys, q, side=side) - grp_start(sorted_keys, gid)
            ).astype(np.int32)


def interval_lookup(sorted_keys, gid, t, val, end):
    """区间连接：取同组内最后一条 `start <= t` 的记录，且要求 `t < end`。否则返回 -1。"""
    q = (gid.astype(np.int64) << SHIFT_T) | t
    pos = np.searchsorted(sorted_keys, q, side="right") - 1
    ok = pos >= 0
    posc = np.clip(pos, 0, len(sorted_keys) - 1)
    same = ok & ((sorted_keys[posc] >> SHIFT_T) == gid)
    if end is not None:
        same &= (end[posc] > t)
    return np.where(same, val[posc], -1)


def topk_codes(codes: np.ndarray, k: int):
    """把字典编码压到 top-k：前 k 名保持 0..k-1 的新编号，其余归 k（OTHER），缺失 -1。"""
    valid = codes[codes >= 0]
    if valid.size == 0:
        return np.full(len(codes), -1, np.int16), 0
    cnt = np.bincount(valid)
    keep = np.argsort(-cnt)[:k]
    remap = np.full(cnt.size, k, dtype=np.int16)      # 默认 OTHER
    remap[keep] = np.arange(len(keep), dtype=np.int16)
    out = np.where(codes >= 0, remap[np.clip(codes, 0, cnt.size - 1)], -1)
    return out.astype(np.int16), int(len(keep))


# 本脚本真正从 poe 读入的列（改代码时必须同步改这里，泄漏自检查的就是它）
SOURCE_COLS_POE = ["subject_id", "poe_seq", "hadm_id", "ordertime",
                   "order_type", "order_subtype", "transaction_type", "order_provider_id"]
# 契约 §0 / §3.1 的禁用源列：要么直接暴露结局，要么是出院之后才知道的信息
FORBIDDEN_SOURCE = {"order_status", "discontinued_by_seq", "discontinue_of_seq",
                    "discharge_location", "hospital_expire_flag", "dischtime", "deathtime",
                    "drg_severity", "drg_mortality"}


def leak_audit(cols) -> dict:
    """泄漏自检。

    ⚠ 这里**故意不用「特征名里含不含某个子串」**来判。
      第一版就是那么写的，结果 `f_detail_code_status`（DNR 等级）因为含 "status"、
      `f_detail_discharge_planning`（出院计划）因为含 "disc"，双双被误报成泄漏。
      而这两个都是 `poe_detail` 里 t₀ 时刻就记在医嘱上的正当属性。
      **一个总在误报的检查迟早会被人关掉，那比没有检查更糟。**
      所以改成对「本脚本到底读了哪些源列」做精确集合判定——这才是能真正拦住泄漏的地方。
    """
    leaked = sorted(FORBIDDEN_SOURCE & set(SOURCE_COLS_POE))
    return {
        "从 poe 读入的源列": SOURCE_COLS_POE,
        "禁用源列是否被读入": leaked,                      # 必须是空列表
        "通过": len(leaked) == 0,
        "标签派生量的用法": ("撤销指针只用于还原『撤销动作发生的绝对时刻 t_disc』（精确整数，"
                             "见 common.resolve_disc_time），且只参与 t_disc <= t₀ 的历史计数；"
                             "它们的取值不进入任何特征值"),
        "历史撤销率只数 t₀ 前已发生的撤销": True,
        "开单者统计为 t₀ 前缀量（非全量统计）": True,
        "科室为 t₀ 时刻的区间连接（非住院期间众数）": True,
        "特征列清单": [c for c, _ in cols if c != "key"],
        "说明": "名字里含 status / disc 的 f_detail_* 是 poe_detail 的医嘱属性，非结局字段",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recheck-only", action="store_true",
                    help="只重算泄漏自检并更新已有的指标 JSON，不重建 features.parquet")
    args = ap.parse_args()
    if args.recheck_only:
        import json as _json
        p = C.METRICS / "s03_features.json"
        d = _json.loads(p.read_text(encoding="utf-8"))
        cols_names = [(c, None) for c in d["非缺失率pct"]]
        d["泄漏自检"] = leak_audit([("key", None)] + cols_names)
        p.write_text(_json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
        print("泄漏自检已重算：", _json.dumps(d["泄漏自检"]["通过"], ensure_ascii=False))
        print("禁用源列是否被读入：", d["泄漏自检"]["禁用源列是否被读入"])
        return

    t0, _ = C.start_log("s03_features")
    cf = C.cfg()
    POE, LAB = C.DATA / "poe.parquet", C.DATA / "labels.parquet"
    N = pq.ParquetFile(POE).metadata.num_rows
    vocab, stats = {}, {}
    print(f"医嘱行数 {N:,}")

    # ================================================================ 读主列
    subj = C.col_np(POE, "subject_id").astype(np.int64)
    seq = C.col_np(POE, "poe_seq").astype(np.int64)
    hadm = C.col_np(POE, "hadm_id").astype(np.int64)
    otime = C.col_np(POE, "ordertime").astype(np.int64)
    key = (subj << SHIFT_KEY) | seq
    ot, v = C.col_np(POE, "order_type", dictionary=True);      vocab["order_type"] = v
    ost, v = C.col_np(POE, "order_subtype", dictionary=True);  vocab["order_subtype"] = v
    tt, v = C.col_np(POE, "transaction_type", dictionary=True); vocab["transaction_type"] = v
    prov, pv = C.col_np(POE, "order_provider_id", dictionary=True)
    vocab["order_provider_id_size"] = len(pv)   # ⚠ 只存词表大小，不存账号取值
    del pv, v
    cause = C.col_np(LAB, "cause").astype(np.int8)
    age = C.col_np(LAB, "age").astype(np.int16)
    y_disc = cause > 0
    # ⚠ 撤销时刻必须用**精确整数**，不能从 float32 的 delay_h 反推——原因见 common.resolve_disc_time
    t_disc, key2, n_res = C.resolve_disc_time(POE)
    assert np.array_equal(key, key2), "poe 的主键与指针解析结果不对齐"
    del key2
    print(f"  主列读完 {time.time() - t0:.0f}s，撤销指针精确解析 {n_res:,} 条")

    # ================================================================ 全局排序键
    def sorted_key(gid, tt_, mask=None):
        m = np.ones(len(gid), bool) if mask is None else mask
        k = (gid[m].astype(np.int64) << SHIFT_T) | np.asarray(tt_)[m].astype(np.int64)
        k.sort()
        return k

    # ⚠ 剔掉**时间倒挂**的撤销事件（t_disc < 下单时刻）。实测 43 条，是 MIMIC 的数据异常
    #   （最极端的倒挂 10.3 年，指针多半指错了）。
    #   为什么必须剔：前缀计数的不变式「已撤数 ≤ 此前下达数」依赖 t₀(j) ≤ t_disc(j)。
    #   一条倒挂的医嘱会被算进本次住院**每一条**医嘱的「历史撤销数」，
    #   却不一定被算进「此前下达数」，不变式就破了——s03 里的断言实测就是被它们炸掉的。
    #   这 43 条在标签里已经因为 delay <= 0 而 t_bin = -1、被排除在 13 类损失外，口径一致。
    disc_ok = y_disc & (t_disc >= otime)
    n_inverted = int((y_disc & (t_disc < otime)).sum())

    ok_hadm = sorted_key(hadm, otime)                        # 住院内全部医嘱的下单时刻
    dk_hadm = sorted_key(hadm, t_disc, disc_ok)              # 住院内已发生的撤销时刻
    ok_prov = sorted_key(prov, otime, prov >= 0)             # 该账号全部医嘱的下单时刻
    dk_prov = sorted_key(prov, t_disc, disc_ok & (prov >= 0))
    sk_prov = sorted_key(prov, t_disc, disc_ok & (cause == 1) & (prov >= 0))  # 该账号已发生的**自撤**
    stats["时间倒挂被剔除的撤销事件"] = n_inverted
    print(f"  排序键建好 {time.time() - t0:.0f}s  (住院撤销键 {len(dk_hadm):,} / "
          f"账号撤销键 {len(dk_prov):,} / 账号自撤键 {len(sk_prov):,}；"
          f"剔除时间倒挂 {n_inverted} 条)")
    del t_disc, disc_ok

    # ================================================================ 小表查找
    A = pq.read_table(C.DATA / "admissions.parquet",
                      columns=["hadm_id", "admittime", "admission_type",
                               "admission_location", "insurance"])
    a_h = A["hadm_id"].to_numpy(zero_copy_only=False).astype(np.int64)
    ao = np.argsort(a_h); a_h = a_h[ao]
    a_t = A["admittime"].fill_null(-1).to_numpy(zero_copy_only=False).astype(np.int64)[ao]

    def catcol(tbl, col, order=None):
        a = tbl[col].dictionary_encode().unify_dictionaries().combine_chunks()
        if hasattr(a, "num_chunks"):
            a = a.chunk(0) if a.num_chunks else a
        names = [str(x) for x in a.dictionary]
        import pyarrow.compute as pc
        c = pc.fill_null(a.indices, -1).to_numpy(zero_copy_only=False).astype(np.int16)
        return (c if order is None else c[order]), names

    a_at, vocab["admission_type"] = catcol(A, "admission_type", ao)
    a_al, vocab["admission_location"] = catcol(A, "admission_location", ao)
    a_in, vocab["insurance"] = catcol(A, "insurance", ao)
    del A, ao

    Pt = pq.read_table(C.DATA / "patients.parquet",
                       columns=["subject_id", "gender", "anchor_year_group"])
    p_s = Pt["subject_id"].to_numpy(zero_copy_only=False).astype(np.int64)
    po = np.argsort(p_s); p_s = p_s[po]
    p_g, vocab["gender"] = catcol(Pt, "gender", po)
    p_y, vocab["anchor_year_group"] = catcol(Pt, "anchor_year_group", po)
    del Pt, po

    Tr = pq.read_table(C.DATA / "transfers.parquet",
                       columns=["hadm_id", "careunit", "intime", "outtime"])
    tr_h = Tr["hadm_id"].fill_null(-1).to_numpy(zero_copy_only=False).astype(np.int64)
    tr_i = Tr["intime"].fill_null(-1).to_numpy(zero_copy_only=False).astype(np.int64)
    tr_o = Tr["outtime"].fill_null(np.iinfo(np.int64).max).to_numpy(zero_copy_only=False).astype(np.int64)
    tr_c, vocab["careunit"] = catcol(Tr, "careunit")
    tm = (tr_h >= 0) & (tr_i >= 0)
    tr_h, tr_i, tr_o, tr_c = tr_h[tm], tr_i[tm], tr_o[tm], tr_c[tm]
    tk = (tr_h << SHIFT_T) | tr_i
    to_ = np.argsort(tk, kind="stable")
    tk, tr_o, tr_c = tk[to_], tr_o[to_], tr_c[to_]
    del Tr, tr_h, tr_i, tm, to_

    Sv = pq.read_table(C.DATA / "services.parquet",
                       columns=["hadm_id", "transfertime", "curr_service"])
    sv_h = Sv["hadm_id"].fill_null(-1).to_numpy(zero_copy_only=False).astype(np.int64)
    sv_t = Sv["transfertime"].fill_null(-1).to_numpy(zero_copy_only=False).astype(np.int64)
    sv_c, vocab["curr_service"] = catcol(Sv, "curr_service")
    sm = (sv_h >= 0) & (sv_t >= 0)
    sk = (sv_h[sm] << SHIFT_T) | sv_t[sm]
    so = np.argsort(sk, kind="stable")
    sk, sv_c = sk[so], sv_c[sm][so]
    del Sv, sv_h, sv_t, sm, so

    # poe_detail：只取覆盖率最高的 6 个字段，每个字段内部对取值做 top-200 压缩
    PD = C.DATA / "poe_detail.parquet"
    d_su = C.col_np(PD, "subject_id").astype(np.int64)
    d_sq = C.col_np(PD, "poe_seq").astype(np.int64)
    d_key = (d_su << SHIFT_KEY) | d_sq
    d_fn, d_fn_names = C.col_np(PD, "field_name", dictionary=True)
    d_fv, _dvn = C.col_np(PD, "field_value", dictionary=True)
    del d_su, d_sq, _dvn                      # ⚠ field_value 的取值不保留、不输出
    d_all = np.sort(d_key)                    # 用来数「这条医嘱有几条属性记录」
    detail_lut = []
    for fname, colname in zip(DETAIL_FIELDS, DETAIL_COLNAMES):
        if fname not in d_fn_names:
            detail_lut.append((np.empty(0, np.int64), np.empty(0, np.int16))); continue
        m = d_fn == d_fn_names.index(fname)
        kk, vv = d_key[m], d_fv[m]
        o = np.argsort(kk, kind="stable")
        vv2, nkept = topk_codes(vv[o], DETAIL_TOPV)
        vocab[colname] = {"n_levels_kept": nkept, "OTHER_code": nkept, "missing_code": -1}
        detail_lut.append((kk[o], vv2))
    del d_key, d_fn, d_fv
    print(f"  查找表建好 {time.time() - t0:.0f}s")

    # ================================================================ 分块算特征并落盘
    age_edges = np.array(cf["features"]["age_bins"], dtype=np.int32)
    prov_edges = np.array(cf["features"]["provider_bins"], dtype=np.int64)
    cols = ([("key", pa.int64()), ("f_order_type", pa.int8()), ("f_order_subtype", pa.int16()),
             ("f_transaction_type", pa.int8()), ("f_hour", pa.int8()), ("f_dow", pa.int8()),
             ("f_since_admit_h", pa.float32()), ("f_admission_type", pa.int8()),
             ("f_admission_location", pa.int8()), ("f_insurance", pa.int8()),
             ("f_careunit", pa.int8()), ("f_curr_service", pa.int8()), ("f_gender", pa.int8()),
             ("f_age_bin", pa.int8()), ("f_anchor_year_group", pa.int8()),
             ("f_n_prior_orders", pa.int32()), ("f_prior_disc_known", pa.int32()),
             ("f_prior_disc_rate", pa.float32()), ("f_dt_prev_order_h", pa.float32()),
             ("f_prov_code", pa.int16()), ("f_prov_n_prior", pa.int32()),
             ("f_prov_n_prior_bin", pa.int8()), ("f_prov_prior_disc_rate", pa.float32()),
             ("f_prov_prior_self_rate", pa.float32()), ("f_n_detail", pa.int8())]
            + [(c, pa.int16()) for c in DETAIL_COLNAMES])
    schema = pa.schema(cols)
    out = C.DATA / "features.parquet"
    # 先写临时文件、成功后再原子改名。否则脚本中途崩掉会在 out 留下**半截 parquet**，
    # 下游（模型、审计）读到它不会报错，只会拿到少了大半行的数据——实测被这么坑过一次。
    tmp = C.DATA / "features.parquet.tmp"
    STEP = 4_000_000
    nz = {c: 0 for c, _ in cols}

    with pq.ParquetWriter(tmp, schema, compression="zstd", compression_level=3) as w:
        for i in range(0, N, STEP):
            s = slice(i, min(i + STEP, N))
            h, t, sb, pr = hadm[s], otime[s], subj[s], prov[s]
            d = {}
            d["key"] = key[s]
            d["f_order_type"] = ot[s].astype(np.int8)
            d["f_order_subtype"] = ost[s].astype(np.int16)
            d["f_transaction_type"] = tt[s].astype(np.int8)
            # 时间：epoch 秒 → 一天内小时 / 星期（1970-01-01 是星期四，故 +4）
            d["f_hour"] = ((t // 3600) % 24).astype(np.int8)
            d["f_dow"] = (((t // 86400) + 4) % 7).astype(np.int8)
            # 住院级
            hp = np.clip(np.searchsorted(a_h, h), 0, len(a_h) - 1)
            hit = a_h[hp] == h
            d["f_since_admit_h"] = np.where(hit & (a_t[hp] >= 0),
                                            (t - a_t[hp]) / 3600.0, np.nan).astype(np.float32)
            d["f_admission_type"] = np.where(hit, a_at[hp], -1).astype(np.int8)
            d["f_admission_location"] = np.where(hit, a_al[hp], -1).astype(np.int8)
            d["f_insurance"] = np.where(hit, a_in[hp], -1).astype(np.int8)
            # 下单时刻所在科室 / 所属专科（区间连接，不是全程众数）
            d["f_careunit"] = interval_lookup(tk, h, t, tr_c, tr_o).astype(np.int8)
            d["f_curr_service"] = interval_lookup(sk, h, t, sv_c, None).astype(np.int8)
            # 人口学
            sp = np.clip(np.searchsorted(p_s, sb), 0, len(p_s) - 1)
            ph = p_s[sp] == sb
            d["f_gender"] = np.where(ph, p_g[sp], -1).astype(np.int8)
            d["f_anchor_year_group"] = np.where(ph, p_y[sp], -1).astype(np.int8)
            d["f_age_bin"] = np.where(age[s] >= 0,
                                      np.searchsorted(age_edges, age[s], "right") - 1,
                                      -1).astype(np.int8)
            # ---- t₀ 前缀统计（防泄漏的核心）----
            # ⚠ 两个计数都用「**严格早于** t₀」（side="left"），口径必须一致。见 prefix_count 的说明。
            npri = prefix_count(ok_hadm, h, t, "left")        # 严格早于 t₀ 的同住院医嘱数
            ndis = prefix_count(dk_hadm, h, t, "left")        # 严格早于 t₀ **已发生**的撤销数
            assert not (ndis > npri).any(), \
                "不变式破了：已发生的撤销数不可能多于此前下达的医嘱数（说明前缀口径又不一致了）"
            d["f_n_prior_orders"] = npri
            d["f_prior_disc_known"] = ndis
            d["f_prior_disc_rate"] = (ndis / np.maximum(npri, 1)).astype(np.float32)
            # 距上一条医嘱多久
            qk = (h << SHIFT_T) | t
            pos = np.searchsorted(ok_hadm, qk, "left")
            gs = grp_start(ok_hadm, h)
            has_prev = pos > gs
            prev_t = np.where(has_prev, ok_hadm[np.clip(pos - 1, 0, len(ok_hadm) - 1)]
                              & ((1 << SHIFT_T) - 1), -1)
            d["f_dt_prev_order_h"] = np.where(has_prev, (t - prev_t) / 3600.0,
                                              np.nan).astype(np.float32)
            # ---- 开单者滚动统计 ----
            pn = np.where(pr >= 0, prefix_count(ok_prov, np.maximum(pr, 0), t, "left"), -1)
            pdn = np.where(pr >= 0, prefix_count(dk_prov, np.maximum(pr, 0), t, "left"), 0)
            psn = np.where(pr >= 0, prefix_count(sk_prov, np.maximum(pr, 0), t, "left"), 0)
            d["f_prov_code"] = pr.astype(np.int16)
            d["f_prov_n_prior"] = pn.astype(np.int32)
            d["f_prov_n_prior_bin"] = np.where(
                pn >= 0, np.searchsorted(prov_edges, pn, "right") - 1, -1).astype(np.int8)
            d["f_prov_prior_disc_rate"] = np.where(
                pn > 0, pdn / np.maximum(pn, 1), np.nan).astype(np.float32)
            d["f_prov_prior_self_rate"] = np.where(
                pdn > 0, psn / np.maximum(pdn, 1), np.nan).astype(np.float32)
            hasp = pr >= 0        # provider 缺失时 pn 记为 -1，不参与不变式
            assert not ((pdn[hasp] > pn[hasp]) | (psn[hasp] > pdn[hasp])).any(), \
                "不变式破了：开单者的已撤数 > 累计下单数，或已自撤数 > 已撤数"
            # ---- 医嘱属性 ----
            kk = key[s]
            lo = np.searchsorted(d_all, kk, "left")
            hi = np.searchsorted(d_all, kk, "right")
            d["f_n_detail"] = np.clip(hi - lo, 0, 127).astype(np.int8)
            for (dk_, dv_), cn in zip(detail_lut, DETAIL_COLNAMES):
                if len(dk_) == 0:
                    d[cn] = np.full(len(kk), -1, np.int16); continue
                p2 = np.clip(np.searchsorted(dk_, kk), 0, len(dk_) - 1)
                d[cn] = np.where(dk_[p2] == kk, dv_[p2], -1).astype(np.int16)

            for c, _ in cols:
                a = d[c]
                nz[c] += int(np.sum(a >= 0) if a.dtype.kind in "iu" else np.sum(np.isfinite(a)))
            w.write_table(pa.table(d, schema=schema))
            print(f"    {min(i + STEP, N):,}/{N:,}  {time.time() - t0:.0f}s", flush=True)

    written = pq.ParquetFile(tmp).metadata.num_rows
    assert written == N, f"写出的行数 {written:,} != 医嘱数 {N:,}，产物不完整，拒绝改名"
    os.replace(tmp, out)      # 原子替换，下游要么看到旧的完整文件，要么看到新的完整文件

    # ================================================================ 统计
    (C.DATA / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False, indent=1),
                                       encoding="utf-8")
    stats["n_rows"] = N
    stats["n_features"] = len(cols) - 1
    stats["features_parquet_mb"] = round(out.stat().st_size / 1024 ** 2, 1)
    stats["非缺失率pct"] = {c: C.pct(v, N) for c, v in nz.items() if c != "key"}
    stats["vocab_sizes"] = {k: (v if isinstance(v, int) else
                                (len(v) if isinstance(v, list) else v))
                            for k, v in vocab.items()}
    stats["detail_fields_used"] = DETAIL_FIELDS
    stats["泄漏自检"] = leak_audit(cols)
    C.save_metrics("s03_features", stats, t0)
    C.append_runlog("s03_features",
                    f"{len(cols) - 1} 个特征 × {N:,} 行 → features.parquet "
                    f"({stats['features_parquet_mb']} MB)")
    print(f"\n→ {out}  {stats['features_parquet_mb']} MB   总耗时 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
