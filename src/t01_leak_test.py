# -*- coding: utf-8 -*-
"""T01 泄漏审计：验证 s03 的「t₀ 前缀统计」真的没有偷看未来。

为什么必须有这个脚本
--------------------
`f_prior_disc_known`（本次住院截至 t₀ 已发生的撤销数）和三个开单者滚动统计，
是全工程唯一会「跨行取信息」的特征。它们一旦写错，模型分数会好看得离谱，
而且**从指标上看不出来**——因为泄漏的方向和标签一致，AUPRC 只会上升不会报警。
所以必须单独验，而且要用**和 s03 完全不同的算法**去验（暴力双重循环），
不能用同一套 searchsorted 再算一遍——那只是把同一个错误复述一遍。

两级验证
--------
A. **合成数据**：手工构造一个已知正确答案的小例子，断言逐格相等。
   重点造一个「陷阱样本」：一条更早下达、但**在 t₀ 之后才被撤销**的医嘱——
   天真实现会把它算进「历史撤销数」，正确实现不会。
B. **真实数据抽样**：随机抽若干次住院，用暴力双重循环重算，与 features.parquet 逐行比对。

用法： python src/t01_leak_test.py [--n-adm 300]
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
import s03_features as F  # noqa: E402   复用被测函数（prefix_count / grp_start）

FAILS = []


def check(name, ok, detail=""):
    print(("  ✅ " if ok else "  ❌ ") + name + (f"   {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


# ============================================================ A. 合成数据
def synthetic():
    print("\n[A] 合成数据：手工构造已知答案")
    # 一次住院（gid=7），5 条医嘱，下单时刻（秒）分别是 0,100,200,300,400
    # 撤销情况：
    #   o0 在 t=150 被撤  -> 对 t0=200 及以后的行「已经发生」
    #   o1 在 t=900 被撤  -> ★陷阱：它比 o2/o3/o4 都早下达，但撤销发生在它们之后，
    #                        天真实现（对前序行的 y_disc 求和）会把它算进去，这是泄漏
    #   o2,o3,o4 未撤
    gid = np.array([7, 7, 7, 7, 7], dtype=np.int64)
    t = np.array([0, 100, 200, 300, 400], dtype=np.int64)
    t_disc = np.array([150, 900, np.inf, np.inf, np.inf], dtype=np.float64)
    y = np.isfinite(t_disc)

    ok_key = np.sort((gid << F.SHIFT_T) | t)
    dk_key = np.sort((gid[y].astype(np.int64) << F.SHIFT_T) | t_disc[y].astype(np.int64))

    n_prior = F.prefix_count(ok_key, gid, t, "left")
    n_disc = F.prefix_count(dk_key, gid, t, "left")

    check("严格早于 t₀ 的同住院医嘱数", list(n_prior) == [0, 1, 2, 3, 4], f"得到 {list(n_prior)}")
    # 正确答案：t=0 →0；t=100 →0（o0 的撤销在 150，还没发生）；t=200 →1（o0 已撤）；
    #           t=300 →1；t=400 →1  （o1 的撤销在 900，对这些时刻都还是未来）
    check("截至 t₀ 已发生的撤销数（正确）", list(n_disc) == [0, 0, 1, 1, 1], f"得到 {list(n_disc)}")

    naive = np.cumsum(np.r_[0, y[:-1]])          # 天真实现：对前序行的 y_disc 累加
    check("★陷阱确实存在：天真实现会高估", list(naive) == [0, 1, 2, 2, 2],
          f"天真={list(naive)} vs 正确={list(n_disc)}")
    check("★正确实现没有被陷阱骗到", not np.array_equal(naive, n_disc))

    # ★陷阱二：同秒撤销的**自指泄漏**。实测真实数据里有 191 条医嘱的撤销时刻与下单时刻同秒。
    #   用 "<=" 口径，这条医嘱会把「我自己已经被撤销了」数进自己的历史特征——专打正类的泄漏。
    g3 = np.array([9, 9, 9], dtype=np.int64)
    t3 = np.array([100, 300, 500], dtype=np.int64)
    td3 = np.array([200, np.inf, 500], dtype=np.float64)     # 第 3 条：下单即撤，同一秒
    y3 = np.isfinite(td3)
    ok3 = np.sort((g3 << F.SHIFT_T) | t3)
    dk3 = np.sort((g3[y3].astype(np.int64) << F.SHIFT_T) | td3[y3].astype(np.int64))
    strict = F.prefix_count(dk3, g3, t3, "left")
    loose = F.prefix_count(dk3, g3, t3, "right")
    check("★同秒撤销：严格口径不把自己数进去", list(strict) == [0, 1, 1], f"得到 {list(strict)}")
    check("★同秒撤销：宽松口径确实会自指（所以不能用）", list(loose) == [0, 1, 2],
          f"得到 {list(loose)}")
    check("★不变式：已撤数 ≤ 此前下达数（严格口径）",
          bool((strict <= F.prefix_count(ok3, g3, t3, "left")).all()))

    # 多分组互不串味
    gid2 = np.array([7, 7, 8, 8], dtype=np.int64)
    t2 = np.array([0, 100, 50, 150], dtype=np.int64)
    td2 = np.array([50, np.inf, np.inf, np.inf], dtype=np.float64)
    y2 = np.isfinite(td2)
    ok2 = np.sort((gid2 << F.SHIFT_T) | t2)
    dk2 = np.sort((gid2[y2].astype(np.int64) << F.SHIFT_T) | td2[y2].astype(np.int64))
    check("跨分组不串味", list(F.prefix_count(dk2, gid2, t2, "left")) == [0, 1, 0, 0],
          f"得到 {list(F.prefix_count(dk2, gid2, t2, 'left'))}")


# ============================================================ B. 真实数据
def on_real(n_adm: int):
    print(f"\n[B] 真实数据：随机抽 {n_adm} 次住院，用暴力双重循环重算")
    FEA, LAB, POE = (C.DATA / "features.parquet", C.DATA / "labels.parquet",
                     C.DATA / "poe.parquet")
    for p in (FEA, LAB, POE):
        if not p.exists():
            print(f"  ⏭ 跳过：{p.name} 还不存在"); return

    hadm = C.col_np(POE, "hadm_id").astype(np.int64)
    otime = C.col_np(POE, "ordertime").astype(np.int64)
    # ⚠ 用精确整数撤销时刻，不从 float32 的 delay_h 反推（反推法正是第一版被本脚本抓出来的缺陷）
    t_disc_all, _, _ = C.resolve_disc_time(POE)
    f_npri = C.col_np(FEA, "f_n_prior_orders").astype(np.int64)
    f_ndis = C.col_np(FEA, "f_prior_disc_known").astype(np.int64)
    fkey = C.col_np(FEA, "key").astype(np.int64)
    lkey = C.col_np(LAB, "key").astype(np.int64)
    aligned = bool(np.array_equal(fkey, lkey))
    check("features 与 labels 的 key 逐行对齐", aligned,
          f"features {len(fkey):,} 行 vs labels {len(lkey):,} 行")
    del fkey, lkey
    if not aligned:
        # 对不齐就别往下跑了：后面的暴力校验会拿错误的下标去索引，报出来的 IndexError
        # 会掩盖掉真正的原因（多半是上一次 s03 崩在半路、留下了不完整的 parquet）。
        print("  ⏭ key 未对齐，跳过后续校验。请先重跑 s03_features.py 生成完整的 features.parquet")
        return {"中止": "features 与 labels 行数/顺序不一致"}

    rng = np.random.default_rng(C.cfg()["seed"])
    uh = np.unique(hadm)
    pick = rng.choice(uh, size=min(n_adm, len(uh)), replace=False)
    sel = np.isin(hadm, pick)
    idx = np.flatnonzero(sel)
    print(f"  抽中 {len(pick):,} 次住院，共 {len(idx):,} 条医嘱")

    h_s, t_s = hadm[idx], otime[idx]
    td_s = t_disc_all[idx]          # 未撤销者为 C.T_NEVER（int64 最大值），恒不满足 <= t₀
    bad_n = bad_d = 0
    for hh in pick:                      # 暴力：组内 O(n²)，仅用于校验
        m = np.flatnonzero(h_s == hh)
        if m.size == 0:
            continue
        tt, dd = t_s[m], td_s[m]
        for a in range(m.size):
            bad_n += int(f_npri[idx[m[a]]] != int((tt < tt[a]).sum()))
            bad_d += int(f_ndis[idx[m[a]]] != int((dd < tt[a]).sum()))   # 严格早于，见 s03.prefix_count
    check("f_n_prior_orders 与暴力法一致", bad_n == 0, f"不一致 {bad_n} 行")
    check("f_prior_disc_known 与暴力法一致", bad_d == 0, f"不一致 {bad_d} 行")

    # 全局硬约束：已发生的撤销数不可能超过此前下达的医嘱数
    over = int((f_ndis > f_npri).sum())
    check("已撤数 ≤ 此前下达数（全 5200 万行）", over == 0, f"违反 {over} 行")

    # 天真实现如果被误用，会在多少行上产生差异？（量化这个坑的规模，写进报告）
    diff_rate = C.pct(int((f_ndis > 0).sum()), len(f_ndis))
    print(f"  参考：f_prior_disc_known > 0 的行占 {diff_rate}%")
    return {"抽样住院数": int(len(pick)), "抽样医嘱数": int(len(idx)),
            "f_n_prior_orders不一致行": bad_n, "f_prior_disc_known不一致行": bad_d,
            "已撤数超过此前下达数的行": over, "有历史撤销记录的行占比pct": diff_rate}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-adm", type=int, default=300)
    a = ap.parse_args()
    t0, _ = C.start_log("t01_leak_test")
    synthetic()
    real = on_real(a.n_adm)
    S = {"合成用例": "全部通过" if not FAILS else FAILS, "真实数据抽样": real,
         "结论": "未发现泄漏" if not FAILS else "发现问题，见 失败项"}
    C.save_metrics("t01_leak_test", S, t0)
    C.append_runlog("t01_leak_test", S["结论"] + f"（失败 {len(FAILS)} 项）")
    print(f"\n{'='*60}\n泄漏审计：{'全部通过 ✅' if not FAILS else '失败 ' + str(FAILS)}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
