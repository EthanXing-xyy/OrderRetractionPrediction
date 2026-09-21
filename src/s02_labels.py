# -*- coding: utf-8 -*-
"""S02 标签构造：从 hosp.poe 的撤销指针造出 13 类离散时间竞争风险标签。

做什么
------
1. 用 `(subject_id, poe_seq)` 合成 int64 主键，把 `discontinued_by_seq` 指针
   用**排序 + np.searchsorted** 解析到实际行（**不做 pandas 自连接**，52 M 行内存扛不住）。
2. 造出契约 §2 的全部标签列：`y_disc / delay_h / cause / censor_h / t_bin / censor_bin / y13 / in_cohort`。
3. 顺手把方案 §9 里预注册的两项检查跑掉：
   - **R1**：撤销有多少发生在出院前 1 h 内或出院之后（= 出院/转科的**批量清理**，不是「反悔」）；
   - **R9**：`order_status` × `y_disc` 交叉核对（指针与状态字段是否矛盾）。

内存
----
逐列读 parquet、用完即 del，全程 numpy 向量化。目标峰值 ≤ 4 GB（本机常态可用 ~6 GB）。

输出
----
`data/labels.parquet`（患者级，不进 git）+ `results/metrics/s02_labels.json`（纯聚合量）。

用法： python src/s02_labels.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

SHIFT = 24  # key = subject_id << 24 | poe_seq


def bucketize(hours: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """左开右闭分桶。edges=[0,1,6,24,72,168,inf] -> 桶 0..5；<=0 或 NaN 返回 -1。"""
    out = np.searchsorted(edges, hours, side="left") - 1
    out = np.where(np.isnan(hours), -1, out)
    return np.clip(out, -1, len(edges) - 2).astype(np.int8)


def main():
    t0, _ = C.start_log("s02_labels")
    cf = C.cfg()
    POE = C.DATA / "poe.parquet"
    edges = np.array(cf["labels"]["bin_edges_h"] + [np.inf], dtype=np.float64)
    n_bins = cf["labels"]["n_bins"]
    assert len(edges) - 1 == n_bins, "config 里 bin_edges_h 与 n_bins 不一致"

    # ---------------------------------------------------------------- 读列
    N = pq.ParquetFile(POE).metadata.num_rows
    print(f"poe 行数 {N:,}")
    subj = C.col_np(POE, "subject_id").astype(np.int64)
    seq = C.col_np(POE, "poe_seq").astype(np.int64)
    assert seq.min() >= 0 and seq.max() < (1 << SHIFT), \
        f"poe_seq 超出键位宽：max={seq.max()}，2**{SHIFT}={1 << SHIFT}"
    otime = C.col_np(POE, "ordertime", fill=np.iinfo(np.int64).min).astype(np.int64)
    n_otime_null = int((otime == np.iinfo(np.int64).min).sum())
    by_seq = C.col_np(POE, "discontinued_by_seq").astype(np.int64)      # -1 = 无指针
    of_seq = C.col_np(POE, "discontinue_of_seq").astype(np.int64)
    prov, _pnames = C.col_np(POE, "order_provider_id", dictionary=True)  # -1 = 缺失
    n_prov_vocab = len(_pnames)
    del _pnames          # ⚠ provider 账号的取值绝不留、绝不打印，只留词表大小
    hadm = C.col_np(POE, "hadm_id").astype(np.int64)                    # -1 = 无住院号
    print(f"  列读完 {time.time() - t0:.0f}s   开单者账号数={n_prov_vocab:,}  "
          f"provider 缺失={C.pct((prov < 0).sum(), N)}%")

    # ---------------------------------------------------------------- 指针解析
    key = (subj << SHIFT) | seq
    is_sorted = bool(np.all(np.diff(key) > 0))
    if is_sorted:
        key_sorted, order = key, None
    else:
        order = np.argsort(key, kind="stable")
        key_sorted = key[order]
    has_ptr = by_seq >= 0
    tkey = (subj[has_ptr] << SHIFT) | by_seq[has_ptr]
    pos = np.clip(np.searchsorted(key_sorted, tkey), 0, N - 1)
    hit = key_sorted[pos] == tkey
    src = np.flatnonzero(has_ptr)[hit]                 # 被撤销的那条医嘱
    dst = pos[hit] if order is None else order[pos[hit]]   # 执行撤销的那条医嘱
    del key, key_sorted, tkey, pos, hit
    n_ptr = int(has_ptr.sum())
    print(f"  指针非空 {n_ptr:,}  可解析 {len(src):,} "
          f"({C.pct(len(src), n_ptr)}%)  文件已按键有序={is_sorted}  ({time.time() - t0:.0f}s)")

    y_disc = np.zeros(N, dtype=bool)
    y_disc[src] = True

    # ---------------------------------------------------------------- 时延与 cause
    delay_h = np.full(N, np.nan, dtype=np.float32)
    delay_h[src] = (otime[dst] - otime[src]) / 3600.0
    ps, pd_ = prov[src], prov[dst]
    both = (ps >= 0) & (pd_ >= 0)
    cause = np.zeros(N, dtype=np.int8)                 # 0=未撤
    cause[src[both & (ps == pd_)]] = 1                 # 1=自撤（同一账号）
    cause[src[both & (ps != pd_)]] = 2                 # 2=他撤
    cause[src[~both]] = 3                              # 3=任一方 provider 缺失
    del ps, pd_, both

    # ---------------------------------------------------------------- 删失时刻
    A = pq.read_table(C.DATA / "admissions.parquet",
                      columns=["hadm_id", "dischtime", "deathtime"])
    a_h = A["hadm_id"].to_numpy(zero_copy_only=False).astype(np.int64)
    a_di = A["dischtime"].fill_null(np.iinfo(np.int64).max).to_numpy(zero_copy_only=False)
    a_de = A["deathtime"].fill_null(np.iinfo(np.int64).max).to_numpy(zero_copy_only=False)
    a_end = np.minimum(a_di.astype(np.int64), a_de.astype(np.int64))   # min(出院, 死亡)
    ao = np.argsort(a_h)
    a_h, a_end = a_h[ao], a_end[ao]
    del A, a_di, a_de, ao

    hp = np.clip(np.searchsorted(a_h, hadm), 0, len(a_h) - 1)
    matched = (hadm >= 0) & (a_h[hp] == hadm)
    censor_h = np.full(N, np.nan, dtype=np.float32)
    end_v = a_end[hp]
    ok_end = matched & (end_v != np.iinfo(np.int64).max)
    censor_h[ok_end] = (end_v[ok_end] - otime[ok_end]) / 3600.0
    del hp, a_h

    # ---------------------------------------------------------------- 成人过滤
    Pt = pq.read_table(C.DATA / "patients.parquet", columns=["subject_id", "anchor_age"])
    p_s = Pt["subject_id"].to_numpy(zero_copy_only=False).astype(np.int64)
    p_a = Pt["anchor_age"].fill_null(-1).to_numpy(zero_copy_only=False).astype(np.int16)
    po = np.argsort(p_s)
    p_s, p_a = p_s[po], p_a[po]
    sp = np.clip(np.searchsorted(p_s, subj), 0, len(p_s) - 1)
    age = np.where(p_s[sp] == subj, p_a[sp], -1).astype(np.int16)
    del Pt, p_s, p_a, po, sp

    in_cohort = (matched & (age >= cf["cohort"]["min_age"])
                 & np.isfinite(censor_h) & (censor_h > cf["cohort"]["min_censor_h"])
                 & (otime != np.iinfo(np.int64).min))

    # ---------------------------------------------------------------- 分桶 + 13 类
    t_bin = bucketize(delay_h, edges)
    censor_bin = bucketize(censor_h, edges)
    y13 = np.full(N, -1, dtype=np.int8)
    y13[in_cohort & ~y_disc] = 0
    good = in_cohort & y_disc & (t_bin >= 0)
    y13[good & (cause == 1)] = (1 + t_bin[good & (cause == 1)])
    y13[good & (cause == 2)] = (1 + n_bins + t_bin[good & (cause == 2)])
    # cause==3（provider 缺失，不知谁撤）与 delay<=0 的异常样本留在 H1，排除在 13 类损失外

    # ---------------------------------------------------------------- 统计（全聚合）
    ic = in_cohort
    n_ic = int(ic.sum())
    dl = delay_h[y_disc & ic]
    S = {
        "n_orders": N, "poe_id_key_sorted": is_sorted,
        "ordertime_null": n_otime_null,
        "provider_missing_pct": C.pct((prov < 0).sum(), N),
        "provider_vocab": n_prov_vocab,
        "ptr_nonnull": n_ptr, "ptr_resolved": int(len(src)),
        "ptr_resolve_rate_pct": C.pct(len(src), n_ptr),
        "reverse_ptr_nonnull": int((of_seq >= 0).sum()),
        "disc_rate_all_pct": C.pct(y_disc.sum(), N),
        "cohort_flow": {
            "全部医嘱": N,
            "有 hadm_id 且能对上 admissions": int(matched.sum()),
            "成人（anchor_age>=18）": int((matched & (age >= 18)).sum()),
            "删失时刻晚于下单时刻": n_ic,
        },
        "in_cohort": n_ic, "in_cohort_pct": C.pct(n_ic, N),
        "disc_rate_in_cohort_pct": C.pct((y_disc & ic).sum(), n_ic),
        "delay_h_quantiles": {str(q): round(float(np.nanquantile(dl, q)), 2)
                              for q in (.01, .1, .25, .5, .75, .9, .99)},
        "delay_le_0": int((dl <= 0).sum()),
        "delay_le_0_pct": C.pct((dl <= 0).sum(), len(dl)),
        "cause_counts": {"未撤": int((ic & (cause == 0)).sum()),
                         "自撤": C.sup((ic & (cause == 1)).sum()),
                         "他撤": C.sup((ic & (cause == 2)).sum()),
                         "未知(provider缺失)": C.sup((ic & (cause == 3)).sum())},
    }
    ns, no = int((ic & (cause == 1)).sum()), int((ic & (cause == 2)).sum())
    S["self_pct_of_known"] = C.pct(ns, ns + no)
    S["unknown_cause_pct_of_disc"] = C.pct((ic & (cause == 3)).sum(), (ic & y_disc).sum())
    S["y13_counts"] = {str(k): C.sup(int((y13 == k).sum())) for k in range(-1, 1 + 2 * n_bins)}
    S["y13_excluded_pct_of_cohort"] = C.pct((ic & (y13 < 0)).sum(), n_ic)
    S["bin_edges_h"] = cf["labels"]["bin_edges_h"]
    S["t_bin_counts_self"] = {str(b): C.sup(int((ic & (cause == 1) & (t_bin == b)).sum()))
                              for b in range(n_bins)}
    S["t_bin_counts_other"] = {str(b): C.sup(int((ic & (cause == 2) & (t_bin == b)).sum()))
                               for b in range(n_bins)}

    # ---- R1：撤销发生在出院前多久？（批量清理嫌疑）----
    m = ic & y_disc & np.isfinite(censor_h)
    gap = (censor_h[m] - delay_h[m]).astype(np.float64)   # 撤销时刻距出院/死亡的小时数
    lab = ["撤销发生在出院之后", "出院前1h内", "出院前1-6h", "出院前6-24h", "出院前24h以上"]
    be = np.array([-np.inf, 0, 1, 6, 24, np.inf])
    idx = np.clip(np.searchsorted(be, gap, side="right") - 1, 0, 4)
    cnt = np.bincount(idx, minlength=5)
    S["R1_撤销距出院时长分布"] = {k: C.sup(v) for k, v in zip(lab, cnt)}
    S["R1_占比pct"] = {k: C.pct(v, m.sum()) for k, v in zip(lab, cnt)}
    S["R1_批量清理嫌疑pct"] = C.pct(cnt[0] + cnt[1], m.sum())   # 出院后 + 出院前1h内

    # ---- R9：order_status × y_disc ----
    st_codes, st_names = C.col_np(POE, "order_status", dictionary=True)
    S["R9_order_status_x_y_disc"] = {
        (st_names[c] if 0 <= c < len(st_names) else "NULL"): {
            "被撤": C.sup(int((ic & (st_codes == c) & y_disc).sum()),),
            "未撤": C.sup(int((ic & (st_codes == c) & ~y_disc).sum())),
        } for c in np.unique(st_codes)}
    del st_codes

    # ---- 撤销率 by order_type ----
    ot, ot_names = C.col_np(POE, "order_type", dictionary=True)
    rows = {}
    for c in np.unique(ot):
        mm = ic & (ot == c)
        n = int(mm.sum())
        if n == 0:
            continue
        nd = int((mm & y_disc).sum())
        s1 = int((mm & (cause == 1)).sum())
        s2 = int((mm & (cause == 2)).sum())
        rows[ot_names[c] if 0 <= c < len(ot_names) else "NULL"] = {
            "n": C.sup(n), "撤销率pct": C.pct(nd, n), "自撤占已知pct": C.pct(s1, s1 + s2)}
    S["撤销率_by_order_type"] = dict(sorted(
        rows.items(), key=lambda kv: -(kv[1]["n"] if isinstance(kv[1]["n"], int) else 0)))
    del ot

    # ---------------------------------------------------------------- 落盘
    out = C.DATA / "labels.parquet"
    schema = pa.schema([("key", pa.int64()), ("subject_id", pa.int64()), ("hadm_id", pa.int64()),
                        ("ordertime", pa.int64()), ("y_disc", pa.bool_()),
                        ("delay_h", pa.float32()), ("cause", pa.int8()),
                        ("censor_h", pa.float32()), ("t_bin", pa.int8()),
                        ("censor_bin", pa.int8()), ("y13", pa.int8()),
                        ("in_cohort", pa.bool_()), ("age", pa.int16())])
    keyc = (subj << SHIFT) | seq
    with pq.ParquetWriter(out, schema, compression="zstd", compression_level=3) as w:
        step = 4_000_000
        for i in range(0, N, step):
            s = slice(i, i + step)
            w.write_table(pa.table({"key": keyc[s], "subject_id": subj[s], "hadm_id": hadm[s],
                                    "ordertime": otime[s], "y_disc": y_disc[s],
                                    "delay_h": delay_h[s], "cause": cause[s],
                                    "censor_h": censor_h[s], "t_bin": t_bin[s],
                                    "censor_bin": censor_bin[s], "y13": y13[s],
                                    "in_cohort": in_cohort[s], "age": age[s]}, schema=schema))
    S["labels_parquet_mb"] = round(out.stat().st_size / 1024 ** 2, 1)

    C.save_metrics("s02_labels", S, t0)
    C.append_runlog("s02_labels",
                    f"队列 {n_ic:,} 条（占 {S['in_cohort_pct']}%），撤销率 "
                    f"{S['disc_rate_in_cohort_pct']}%，自撤占已知 {S['self_pct_of_known']}%，"
                    f"时延 p50={S['delay_h_quantiles']['0.5']}h，"
                    f"批量清理嫌疑 {S['R1_批量清理嫌疑pct']}%")
    print("\n关键结果：")
    for k in ("cohort_flow", "disc_rate_in_cohort_pct", "self_pct_of_known",
              "delay_h_quantiles", "R1_占比pct", "y13_excluded_pct_of_cohort"):
        print(f"  {k}: {S[k]}")
    print(f"→ {out}  ({S['labels_parquet_mb']} MB)   总耗时 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
