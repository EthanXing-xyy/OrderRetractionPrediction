# -*- coding: utf-8 -*-
"""S09 场景 B：加入执行层（eMAR）事件的 landmark 预测。

为什么要有这一步
----------------
方案里支撑「撤销是两个方向相反的过程、合并建模会互相抵消」的那个实测证据是：

    「该医嘱曾被 Not Given」这个特征，把自撤/他撤的天平
      在 Medications 上推 **−14.51 pp**，在 IV therapy 上推 **+12.02 pp**。

**但 `Not Given` 是执行层事件，发生在 t₀ 之后，根本不在场景 A 的特征集里。**
所以场景 A 上双头赢不赢，其实检验不到这个机制。真正该检验它的地方是这里。

landmark 设计（这是生存分析里处理「时变协变量」的标准做法，不是我们发明的）
------------------------------------------------------------------------
固定一个观测窗 τ（默认 24 h）：

  * **入组条件**：到 t₀+τ 时刻这条医嘱还「活着」——即 `censor_h > τ` 且尚未被撤销。
    已经在 τ 之前被撤销的，不该再问「它接下来会不会被撤」。
  * **预测目标**：在 **(τ, τ+24h]** 这个窗口内被撤销。
  * **可用信息**：场景 A 的全部特征 + **(t₀, t₀+τ] 内**这条医嘱的 eMAR 执行事件。
    eMAR 事件一律只取 `charttime <= t₀+τ` 的，晚于这个时刻的一概不看。

⚠ `Not Given per Sliding Scale`（943,457 行）是**协议规定的不给药**（胰岛素滑动量表），
   不是临床决策，必须从「Not Given」信号里剔除。另有 3 行 `... in Other Location` 同理。

跑哪些模型
----------
每个 τ 跑四个，两两对照，一次回答两个问题：

| 模型 | 有无 eMAR | 头 | 回答 |
|---|---|---|---|
| `lgbmB{τ}_single_noemar` | ✗ | 单头 | 基准 |
| `lgbmB{τ}_single` | ✓ | 单头 | **eMAR 有没有用？** |
| `lgbmB{τ}_3class_noemar` | ✗ | 三分类 | — |
| `lgbmB{τ}_3class` | ✓ | 三分类 | **加了那个反向特征之后，分头建模是否才开始占优？** |

指标一律调 `s07_eval` 里的函数算，**不另写一套尺子**。

用法
----
    python src/s09_scenario_b.py                 # τ=24，全量（按需抽样）
    python src/s09_scenario_b.py --taus 6,12,24
    python src/s09_scenario_b.py --smoke         # 1% 患者，跑通用
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import s07_eval as E  # noqa: E402   复用统一指标，绝不自己再写一份

SHIFT_T = 34
IDX_BITS = 26                      # 52,212,109 < 2**26，够装 poe 行号
# 协议性的「不给药」，不是决策信号，必须剔除
PROTOCOL_NOT_GIVEN = ("Not Given per Sliding Scale",
                      "Not Given per Sliding Scale in Other Location")
SUF = ""          # smoke 跑时置成 "_smoke"，避免与全量产物同名混版


def ev_sorted(idx: np.ndarray, t: np.ndarray, mask: np.ndarray | None = None):
    """把 eMAR 事件压成一个有序的 int64 复合键：poe行号 << 34 | charttime。"""
    m = slice(None) if mask is None else mask
    k = (idx[m].astype(np.int64) << SHIFT_T) | t[m].astype(np.int64)
    o = np.argsort(k, kind="stable")
    return k[o], o


def count_in_window(keys: np.ndarray, idx: np.ndarray, t0: np.ndarray, t1: np.ndarray):
    """这条医嘱在 (t₀, t₁] 内有多少个事件。左开右闭，t₀ 当刻的事件不算（分钟粒度分不出先后）。"""
    base = idx.astype(np.int64) << SHIFT_T
    lo = np.searchsorted(keys, base | t0, side="right")
    hi = np.searchsorted(keys, base | t1, side="right")
    return (hi - lo).astype(np.int32), lo, hi


def build_emar_features(poe_key_sorted, poe_order, n_poe, tau_h, otime):
    """构造 (t₀, t₀+τ] 窗口内的 eMAR 动态特征。返回 dict[str, np.ndarray]。"""
    EM = C.DATA / "emar.parquet"
    e_su = C.col_np(EM, "subject_id").astype(np.int64)
    e_sq = C.col_np(EM, "poe_seq").astype(np.int64)
    e_ct = C.col_np(EM, "charttime", fill=-1).astype(np.int64)
    e_st = C.col_np(EM, "scheduletime", fill=-1).astype(np.int64)
    e_tx, e_names = C.col_np(EM, "event_txt", dictionary=True)

    ek = (e_su << C.SHIFT_KEY) | e_sq
    pos = np.clip(np.searchsorted(poe_key_sorted, ek), 0, n_poe - 1)
    hit = (poe_key_sorted[pos] == ek) & (e_ct >= 0) & (e_sq >= 0)
    e_idx = np.where(hit, pos if poe_order is None else poe_order[pos], -1).astype(np.int64)
    del e_su, e_sq, ek, pos
    ok = e_idx >= 0
    print(f"    eMAR {len(e_idx):,} 行，能连回 poe 的 {int(ok.sum()):,} "
          f"({C.pct(ok.sum(), len(e_idx))}%)")

    adm = ok & np.isin(e_tx, [i for i, n in enumerate(e_names) if n == "Administered"])
    ng_all = [i for i, n in enumerate(e_names) if "not given" in n.lower()]
    ng_proto = [i for i, n in enumerate(e_names) if n in PROTOCOL_NOT_GIVEN]
    ng_real = [i for i in ng_all if i not in ng_proto]
    ngm = ok & np.isin(e_tx, ng_real)
    late = ok & (e_st >= 0) & (e_ct > e_st)          # 实际给药晚于计划时刻
    print(f"    Administered {int(adm.sum()):,} / 真·Not Given {int(ngm.sum()):,} "
          f"（已剔除协议性 {int((ok & np.isin(e_tx, ng_proto)).sum()):,}） / 延迟 {int(late.sum()):,}")

    t1 = otime + int(tau_h * 3600)
    out = {}
    k_all, o_all = ev_sorted(e_idx, e_ct, ok)
    n_all, lo, hi = count_in_window(k_all, np.arange(n_poe), otime, t1)
    out["e_n_events"] = n_all
    # 窗口内最后一个事件的类别（hi-1 落在窗口里才算）
    last = np.where(hi > lo, e_tx[np.flatnonzero(ok)[o_all][np.clip(hi - 1, 0, len(k_all) - 1)]], -1)
    out["e_last_event"] = last.astype(np.int16)
    del k_all, o_all, lo, hi, last

    for name, m in (("e_n_administered", adm), ("e_n_not_given", ngm), ("e_n_late", late)):
        if m.sum() == 0:
            out[name] = np.zeros(n_poe, np.int32); continue
        k, _ = ev_sorted(e_idx, e_ct, m)
        out[name], _, _ = count_in_window(k, np.arange(n_poe), otime, t1)
        del k
    out["e_has_not_given"] = (out["e_n_not_given"] > 0).astype(np.int8)
    with np.errstate(invalid="ignore", divide="ignore"):
        out["e_late_ratio"] = np.where(out["e_n_events"] > 0,
                                       out["e_n_late"] / np.maximum(out["e_n_events"], 1),
                                       np.nan).astype(np.float32)
    return out, e_names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--taus", default="24", help="观测窗（小时），逗号分隔")
    ap.add_argument("--horizon", type=float, default=24.0, help="预测窗口长度（小时）")
    ap.add_argument("--max-train", type=int, default=10_000_000)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    taus = [float(x) for x in a.taus.split(",")]
    global SUF
    SUF = "_smoke" if a.smoke else ""

    t0w, _ = C.start_log("s09_scenario_b" + ("_smoke" if a.smoke else ""))
    import lightgbm as lgb
    cf = C.cfg()
    POE, LAB, FEA, SPL = (C.DATA / "poe.parquet", C.DATA / "labels.parquet",
                          C.DATA / "features.parquet", C.DATA / "splits.parquet")
    for p in (POE, LAB, FEA, SPL):
        if not p.exists():
            sys.exit(f"缺少 {p.name}，请先跑完 s01~s04")

    n_poe = pq.ParquetFile(POE).metadata.num_rows
    subj = C.col_np(POE, "subject_id").astype(np.int64)
    seq = C.col_np(POE, "poe_seq").astype(np.int64)
    otime = C.col_np(POE, "ordertime").astype(np.int64)
    key = (subj << C.SHIFT_KEY) | seq
    poe_order = None if bool(np.all(np.diff(key) > 0)) else np.argsort(key, kind="stable")
    key_sorted = key if poe_order is None else key[poe_order]
    del seq

    y_disc = C.col_np(LAB, "y_disc", fill=False).astype(bool)
    delay_h = C.col_np(LAB, "delay_h", fill=np.nan).astype(np.float32)
    cause = C.col_np(LAB, "cause").astype(np.int8)
    censor_h = C.col_np(LAB, "censor_h", fill=np.nan).astype(np.float32)
    in_co = C.col_np(LAB, "in_cohort", fill=False).astype(bool)

    SP = pq.read_table(SPL)
    s_sub = SP["subject_id"].to_numpy(zero_copy_only=False).astype(np.int64)
    s_spl = SP["split"].to_numpy(zero_copy_only=False).astype(np.int8)
    split = s_spl[np.searchsorted(s_sub, subj)]
    del SP, s_sub, s_spl
    if a.smoke:
        keep_sub = (subj % 100) == 0
        in_co = in_co & keep_sub
        print(f"  --smoke：只留 1% 患者，队列 {int(in_co.sum()):,} 行")

    fcols = [c for c in pq.ParquetFile(FEA).schema.names if c.startswith("f_")]
    CATS = [c for c in fcols if c in (
        "f_order_type", "f_order_subtype", "f_transaction_type", "f_hour", "f_dow",
        "f_admission_type", "f_admission_location", "f_insurance", "f_careunit",
        "f_curr_service", "f_gender", "f_age_bin", "f_anchor_year_group",
        "f_prov_code", "f_prov_n_prior_bin") or c.startswith("f_detail_")]

    report = {"taus": taus, "horizon_h": a.horizon, "smoke": a.smoke,
              "protocol_not_given_excluded": list(PROTOCOL_NOT_GIVEN), "per_tau": {}}

    for tau in taus:
        print(f"\n{'='*70}\nτ = {tau} h   预测窗口 ({tau}, {tau + a.horizon}] h")
        ef, ev_names = build_emar_features(key_sorted, poe_order, n_poe, tau, otime)

        # ---- landmark 入组 + 目标 ----
        alive = in_co & (censor_h > tau) & (~y_disc | (delay_h > tau))
        y_b = alive & y_disc & (delay_h > tau) & (delay_h <= tau + a.horizon)
        cb = np.where(y_b & (cause == 1), 1, np.where(y_b & (cause == 2), 2, 0)).astype(np.int8)
        n_alive = int(alive.sum())
        print(f"  landmark 入组 {n_alive:,}（占队列 {C.pct(n_alive, in_co.sum())}%），"
              f"窗口内被撤 {int(y_b.sum()):,}（{C.pct(y_b.sum(), n_alive)}%）；"
              f"其中自撤 {C.sup((cb == 1).sum())} 他撤 {C.sup((cb == 2).sum())}")
        # 「曾被 Not Given」与后续撤销的关系——就是方案里那个反向效应，这里逐 order_type 复核
        ot, ot_names = C.col_np(POE, "order_type", dictionary=True)
        rev = {}
        for c in np.unique(ot[alive]):
            m = alive & (ot == c)
            if m.sum() < 1000:
                continue
            hasng = m & (ef["e_has_not_given"] > 0)
            non = m & (ef["e_has_not_given"] == 0)
            def self_share(mm):
                s1 = int((mm & (cb == 1)).sum()); s2 = int((mm & (cb == 2)).sum())
                return C.pct(s1, s1 + s2)
            a1, a0 = self_share(hasng), self_share(non)
            rev[ot_names[c]] = {"n_有NotGiven": C.sup(hasng.sum()), "自撤占比pct": a1,
                                "n_无NotGiven": C.sup(non.sum()), "对照自撤占比pct": a0,
                                "Δpp": (round(a1 - a0, 2) if a1 is not None and a0 is not None
                                        else None)}
        report["per_tau"].setdefault(str(tau), {})["NotGiven对自撤占比的影响"] = rev
        print(f"  『曾被 Not Given』对自撤占比的影响（方案里的反向效应）：")
        for k, v in list(rev.items())[:6]:
            print(f"    {k:18s} Δ={v['Δpp']} pp  (有 {v['n_有NotGiven']} / 无 {v['n_无NotGiven']})")

        # ---- 组装训练矩阵 ----
        # ⚠ 不要为全部 5221 万行建矩阵：37 列 × float32 = 7.7 GB，本机必崩（smoke 靠换页才活下来）。
        #   先把 tr/va/te 三个行集合（含训练抽样上限）定下来，**只为这些行建矩阵**。
        tr = alive & (split == 0)
        if tr.sum() > a.max_train:          # 按患者抽样，不按行，避免同一患者跨训练/验证
            step = int(np.ceil(tr.sum() / a.max_train))
            tr = tr & ((subj % step) == 0)
        va, te = alive & (split == 1), alive & (split == 2)
        need = tr | va | te
        idx_need = np.flatnonzero(need)
        pos = np.full(len(need), -1, np.int64)
        pos[idx_need] = np.arange(len(idx_need))
        tr_i, va_i, te_i = (pos[np.flatnonzero(tr)], pos[np.flatnonzero(va)],
                            pos[np.flatnonzero(te)])
        print(f"  训练 {len(tr_i):,} / 验证 {len(va_i):,} / 测试 {len(te_i):,} 行"
              f"（矩阵只建这 {len(idx_need):,} 行）")

        ecols = ["e_n_events", "e_n_administered", "e_n_not_given", "e_n_late",
                 "e_has_not_given", "e_last_event", "e_late_ratio"]
        res_tau = report["per_tau"][str(tau)]
        res_tau["n_train"], res_tau["n_test"] = len(tr_i), len(te_i)
        for use_emar in (False, True):
            cols = fcols + (ecols if use_emar else [])
            cats = CATS + (["e_last_event"] if use_emar else [])
            suffix = "" if use_emar else "_noemar"
            Xall = np.empty((len(idx_need), len(cols)), np.float32)
            for j, c in enumerate(cols):          # 逐列读、逐列子集化，峰值只多一列
                src = ef[c] if c in ef else C.col_np(FEA, c, fill=-1)
                Xall[:, j] = src[idx_need]
                if c not in ef:
                    del src
            for head in ("single", "3class"):
                name = f"lgbmB{int(tau)}_{head}{suffix}"
                t1 = time.time()
                ytr = (y_b if head == "single" else cb)
                params = dict(num_leaves=cf["lgbm"]["num_leaves"],
                              learning_rate=cf["lgbm"]["learning_rate"],
                              min_child_samples=cf["lgbm"]["min_child_samples"],
                              feature_fraction=cf["lgbm"]["feature_fraction"],
                              num_threads=cf["lgbm"]["num_threads"], verbose=-1,
                              objective="binary" if head == "single" else "multiclass",
                              **({} if head == "single" else {"num_class": 3}))
                dtr = lgb.Dataset(Xall[tr_i], label=ytr[tr].astype(np.int32),
                                  feature_name=cols, categorical_feature=cats)
                dva = lgb.Dataset(Xall[va_i], label=ytr[va].astype(np.int32), reference=dtr)
                bst = lgb.train(params, dtr, num_boost_round=400,
                                valid_sets=[dva],
                                callbacks=[lgb.early_stopping(50, verbose=False)])
                pt = bst.predict(Xall[te_i], num_iteration=bst.best_iteration)
                p_disc = pt if head == "single" else (1.0 - pt[:, 0])
                m = {"n_train": int(tr.sum()), "n_test": int(te.sum()),
                     "best_iter": int(bst.best_iteration), "seconds": round(time.time() - t1),
                     "auprc": round(E.auprc(y_b[te], p_disc), 6),
                     "auroc": round(E.auroc(y_b[te], p_disc), 6),
                     "brier": round(E.brier(y_b[te], p_disc), 6)}
                if head == "3class":
                    m["auprc_self"] = round(E.auprc((cb[te] == 1), pt[:, 1]), 6)
                    m["auprc_other"] = round(E.auprc((cb[te] == 2), pt[:, 2]), 6)
                res_tau[name] = m
                # ⚠ smoke 与全量必须写到**不同路径**。原先共用一个文件名，
                #   结果全量跑到一半被掐之后，目录里 2 个是全量的、2 个是 smoke 残留的——
                #   文件都合法、都能读，但混版且无从分辨。这比直接崩掉危险得多。
                np.save(C.DATA / f"pred_{name}{SUF}.npy", p_disc.astype(np.float32))
                print(f"    {name:32s} AUPRC={m['auprc']:.4f} AUROC={m['auroc']:.4f} "
                      f"({m['seconds']}s, {m['best_iter']} 轮)")
                bst.save_model(str(C.ROOT / "checkpoints" / f"{name}.txt"))
                del dtr, dva, bst
            del Xall


        # ---- 两个对照的 bootstrap ----
        te = alive & (split == 2)
        units = subj[te]
        def ld(n):
            return np.load(C.DATA / f"pred_{n}{SUF}.npy")
        for tag, A, B in (("eMAR的贡献(单头)", f"lgbmB{int(tau)}_single",
                           f"lgbmB{int(tau)}_single_noemar"),
                          ("eMAR的贡献(三分类)", f"lgbmB{int(tau)}_3class",
                           f"lgbmB{int(tau)}_3class_noemar"),
                          ("分头vs单头(有eMAR)", f"lgbmB{int(tau)}_3class",
                           f"lgbmB{int(tau)}_single"),
                          ("分头vs单头(无eMAR)", f"lgbmB{int(tau)}_3class_noemar",
                           f"lgbmB{int(tau)}_single_noemar")):
            d = E.bootstrap_delta(ld(A), ld(B), y_b[te], metric="auprc",
                                  units=units, verbose=False)
            res_tau[f"Δ_{tag}"] = {k: (round(v, 6) if isinstance(v, float) else v)
                                   for k, v in d.items()}
            print(f"    Δ {tag:22s} = {d['delta']:+.4f}  "
                  f"95%CI[{d['ci_lo']:+.4f},{d['ci_hi']:+.4f}]  "
                  f"(有效重抽 {d.get('n_boot')}/{d.get('n_boot_requested')})")
        del ef

    C.save_metrics("s09_scenario_b" + ("_smoke" if a.smoke else ""), report, t0w)
    C.append_runlog("s09_scenario_b",
                    f"τ={taus} landmark 场景 B 跑完，四模型对照 + eMAR 贡献的 bootstrap")
    print(f"\n总耗时 {time.time() - t0w:.0f}s")


if __name__ == "__main__":
    main()
