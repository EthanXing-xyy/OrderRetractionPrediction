#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""s07_eval.py —— 统一评测（库 + 命令行两用）

**这是全工程唯一的一把尺子。** 所有模型（baserate / logreg / lgbm_single / lgbm_dual /
gru / transformer_*）只负责按 `docs/01_数据契约.md` §7 吐出
`results/preds/<模型名>_<split>.parquet`，指标一律由本文件计算，
模型脚本**不许自己算指标**（否则不同模型的 AUPRC 就不可比了）。

三种用法
--------
```
python src/s07_eval.py --model lgbm_single --split test
python src/s07_eval.py --compare lgbm_dual lgbm_single --split test   # 预注册主实验
python src/s07_eval.py --selftest                                     # 合成数据自测
python src/s07_eval.py --smoke                                        # 读真 labels 打几个聚合计数
```

设计要点
--------
1. **指标函数全是纯函数**（输入 numpy，输出 float / dict），方便单测，见 §B。
2. **bootstrap 按 `subject_id` 重抽**（契约 §6），不是按行——同一患者的所有医嘱一起进出，
   否则 CI 会假窄。实现见 §C：把「重抽」变成「给每一行一个整数权重」，
   于是每次重抽只要 1 次 gather + 2 次 cumsum，全程不排序（排序在预处理里做一次）。
3. **AUPRC 是主指标**（正类约 20%，AUROC 会虚高）。
4. **隐私**：只输出聚合量；n < 10 的计数一律走 `C.sup()`；
   绝不打印 subject_id / order_provider_id / 绝对时间戳的取值。
5. 常数（bootstrap 次数、校准桶数、判据阈值）从 `C.cfg()["eval"]` 读，不在本文件硬编码。

⚠ 本脚本**纯 CPU**：进程一进来就把 CUDA_VISIBLE_DEVICES 置空，
  免得 `common.env_fingerprint()` 里的 `torch.cuda.is_available()` 去抢那张 6 GB 卡
  （同一时刻只许一个 GPU 作业，见 AGENTS.md §4.5）。代价是环境指纹里不再记录 GPU 型号。
"""
from __future__ import annotations

import os

# 必须在 import common（进而可能 import torch）之前设置
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

PREDS_DIR = C.RESULTS / "preds"
LABELS_PATH = C.DATA / "labels.parquet"

# 契约 §7 登记在案的模型名（不在表里只是提醒，不拦截）
KNOWN_MODELS = ("baserate", "logreg", "lgbm_single", "lgbm_dual",
                "gru", "transformer_v0", "transformer_cr")

# 契约 §7 的预测列
P_REQUIRED = ("key", "subject_id", "p_disc", "p_self", "p_other", "p_disc_24h")
P13_COLS = tuple(f"p13_{i}" for i in range(13))

# 契约 §2.1 的桶边界（小时）。24h 对应前 3 个桶（bin 0/1/2）。
BINS_UPTO_24H = (0, 1, 2)

_EVAL_DEFAULTS = {"bootstrap_n": 1000, "bootstrap_unit": "subject_id",
                  "calibration_bins": 10, "prereg_delta_auprc": 0.01}


def ecfg(key: str):
    """读 `configs/config.yaml` 的 eval 段。缺键就用文档化的默认值并告警。"""
    try:
        v = C.cfg().get("eval", {}).get(key, None)
    except Exception as e:                                  # 配置读不到不该让自测挂掉
        print(f"[warn] 读 config.yaml 失败（{e}），eval.{key} 用默认值")
        v = None
    if v is None:
        print(f"[warn] config.yaml 缺 eval.{key}，用默认值 {_EVAL_DEFAULTS[key]}")
        return _EVAL_DEFAULTS[key]
    return v


def n_threads(user: int | None = None) -> int:
    """CPU 线程数上限 16（另一个 agent 在占 GPU，CPU 也别吃死）。"""
    return max(1, min(16, user or (os.cpu_count() or 4)))


# ============================================================================ §A
# 曲线内核：一次预处理 + 带整数权重的 tp/fp 曲线
# ----------------------------------------------------------------------------
# 为什么要这么写：bootstrap 要重复算 1000 次 AUPRC。朴素做法是每次 gather 出
# 一百万行再 argsort，单次 ~150 ms，1000 次就是两分半 × 两个模型。
# 这里改成：**分数的排序只做一次**，每次重抽只是给每一行换一个整数权重 w
# （w = 这一行所属患者这次被抽中了几次），于是曲线 = cumsum(w*y) / cumsum(w)。
# 单次降到 ~15 ms，而且结果与「真的 gather 出那些行再算」逐位相等
# （自测 T18 就是拿这两条路径对拍的）。
# ============================================================================

def _prep_curve(y: np.ndarray, p: np.ndarray, units: np.ndarray | None = None) -> dict:
    """把 (y, p) 预处理成按分数降序排好的「曲线规格」，供加权指标反复复用。

    返回 dict：
      y_ord   int8，按 p 降序排列后的标签
      p_ord   float64，同序的分数
      u_ord   int32，同序的「重抽单元编号」（患者编号）；units=None 时为 None
      tie_end int64 索引数组：每个不同分数值的**最后一个**位置；无并列时为 None
      n_unit  单元个数
    """
    y = np.asarray(y).astype(np.int8, copy=False)
    p = np.asarray(p, dtype=np.float64)
    if y.shape != p.shape:
        raise ValueError(f"y 与 p 形状不一致：{y.shape} vs {p.shape}")
    if not np.isfinite(p).all():
        raise ValueError("分数里有 NaN/Inf——上游模型没按契约 §7 出干净的概率")

    order = np.argsort(-p, kind="stable")
    p_ord = p[order]
    y_ord = y[order]
    diff = np.flatnonzero(p_ord[1:] != p_ord[:-1])
    n = p_ord.size
    tie_end = None if diff.size == n - 1 else np.append(diff, n - 1)

    u_ord, n_unit = None, 0
    if units is not None:
        codes = np.asarray(units)
        if codes.dtype.kind != "i" or codes.min(initial=0) < 0:
            _, codes = np.unique(codes, return_inverse=True)
        codes = codes.astype(np.int32, copy=False)
        n_unit = int(codes.max()) + 1 if codes.size else 0
        u_ord = codes[order]
    return {"y_ord": y_ord, "p_ord": p_ord, "u_ord": u_ord,
            "tie_end": tie_end, "n": n, "n_unit": n_unit}


def _tpfp(spec: dict, w: np.ndarray | None):
    """累计真阳/假阳曲线（阈值由高到低）。w 是每行的整数权重，None 表示全 1。

    ⚠ **踩过的坑**（端到端演练时 1000 次 bootstrap 里 569 次返回 nan）：
      按患者重抽时约 1/e 的患者权重为 0，于是分数最高的那几行很可能整段权重为 0，
      累计分母 tp+fp 在开头是 0，precision 变成 0/0 = nan，
      再乘上 Δrecall=0 仍是 nan，**整条 AP 就被污染成 nan**。
      修法：累计权重 tot 是非降的，前面那段 tot==0 直接切掉——
      这与「真的 gather 出被抽中的行再算」逐位等价（那些行本来就不在样本里）。
    """
    y_ord, tie_end = spec["y_ord"], spec["tie_end"]
    if w is None:
        tp = np.cumsum(y_ord, dtype=np.int64)
        tot = np.arange(1, spec["n"] + 1, dtype=np.int64)
    else:
        tp = np.cumsum(w * y_ord, dtype=np.int64)
        tot = np.cumsum(w, dtype=np.int64)
    if tie_end is not None:                 # 并列分数必须并成一个阈值点，否则 AUC 会偏
        tp = tp[tie_end]
        tot = tot[tie_end]
    if w is not None:                       # tot 非降；切掉开头那段累计权重为 0 的位置
        k = int(np.searchsorted(tot, 1))
        if k:
            tp, tot = tp[k:], tot[k:]
    return tp, tot - tp


def _ap_from(tps: np.ndarray, fps: np.ndarray) -> float:
    """average precision = Σ (R_i − R_{i−1}) · P_i（与 sklearn 同一约定）。"""
    if tps.size == 0:
        return float("nan")
    pos = tps[-1]
    if pos <= 0:
        return float("nan")
    prec = tps / (tps + fps)
    rec = tps / pos
    return float(np.sum(np.diff(rec, prepend=0.0) * prec))


def _auc_from(tps: np.ndarray, fps: np.ndarray) -> float:
    """ROC 曲线下面积（梯形法，(0,0) 补在最前）。"""
    if tps.size == 0:
        return float("nan")
    pos, neg = tps[-1], fps[-1]
    if pos <= 0 or neg <= 0:
        return float("nan")
    tpr = np.concatenate(([0.0], tps / pos))
    fpr = np.concatenate(([0.0], fps / neg))
    return float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) * 0.5))


# ============================================================================ §B
# 指标函数（纯函数，输入 numpy）
# ============================================================================

def auroc(y: np.ndarray, p: np.ndarray) -> float:
    """ROC AUC。并列分数按「并成一个阈值点」处理（等价于 mid-rank）。

    正类或负类为空时返回 nan（不是 0.5，也不是 0——那会骗人）。
    """
    return _auc_from(*_tpfp(_prep_curve(y, p), None))


def auprc(y: np.ndarray, p: np.ndarray) -> float:
    """PR AUC（average precision）。**本工程主指标**：正类约 20%，AUROC 会虚高。"""
    return _ap_from(*_tpfp(_prep_curve(y, p), None))


def brier(y: np.ndarray, p: np.ndarray) -> float:
    """Brier score = mean((p − y)²)，越小越好。"""
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    return float(np.mean((p - y) ** 2))


def reliability_curve(y: np.ndarray, p: np.ndarray, n_bins: int = 10,
                      strategy: str = "uniform") -> dict:
    """可靠性曲线（校准曲线）的桶数据，直接给 `s08_figures.py` 画图用。

    strategy="uniform"  等宽分桶（[0,0.1),[0.1,0.2),…）——ECE 的标准定义
    strategy="quantile" 等频分桶——预测高度偏斜时更有信息量

    返回 dict：bin_edges / bin_center / mean_pred / frac_pos / count / n / strategy
    空桶的 mean_pred 与 frac_pos 为 nan，count 为 0。**本函数返回原始计数，
    小格子抑制在写 JSON 的时候做**（见 `_sup_list`）。
    """
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    n = p.size
    if strategy == "quantile":
        edges = np.unique(np.quantile(p, np.linspace(0.0, 1.0, n_bins + 1)))
        if edges.size < 2:
            edges = np.array([0.0, 1.0])
        edges[0], edges[-1] = min(edges[0], 0.0), max(edges[-1], 1.0)
    elif strategy == "uniform":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    else:
        raise ValueError(f"未知的分桶策略 {strategy!r}")

    nb = edges.size - 1
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, nb - 1)
    cnt = np.bincount(idx, minlength=nb).astype(np.int64)
    s_p = np.bincount(idx, weights=p, minlength=nb)
    s_y = np.bincount(idx, weights=y, minlength=nb)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_pred = np.where(cnt > 0, s_p / np.maximum(cnt, 1), np.nan)
        frac_pos = np.where(cnt > 0, s_y / np.maximum(cnt, 1), np.nan)
    return {"strategy": strategy,
            "bin_edges": [float(x) for x in edges],
            "bin_center": [float(x) for x in (edges[:-1] + edges[1:]) / 2.0],
            "mean_pred": [float(x) for x in mean_pred],
            "frac_pos": [float(x) for x in frac_pos],
            "count": [int(x) for x in cnt],
            "n": int(n)}


def ece(y: np.ndarray, p: np.ndarray, n_bins: int = 10, strategy: str = "uniform") -> float:
    """Expected Calibration Error = Σ (n_b/n)·|实际率_b − 平均预测_b|。"""
    rc = reliability_curve(y, p, n_bins=n_bins, strategy=strategy)
    cnt = np.asarray(rc["count"], dtype=np.float64)
    gap = np.abs(np.asarray(rc["frac_pos"]) - np.asarray(rc["mean_pred"]))
    m = cnt > 0
    if not m.any():
        return float("nan")
    return float(np.sum(cnt[m] * gap[m]) / cnt.sum())


def auprc_at_24h(y_disc: np.ndarray, delay_h: np.ndarray, p_disc_24h: np.ndarray,
                 horizon_h: float = 24.0) -> float:
    """契约 §6 的 `auprc@24h`：真值 = `y_disc & (delay_h <= 24)`，分数 = `p_disc_24h`。

    未撤销的行 `delay_h` 是 NaN，`NaN <= 24` 为 False，正好落在负类，不必特判。
    """
    y_disc = np.asarray(y_disc).astype(bool, copy=False)
    d = np.asarray(delay_h, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        truth = y_disc & (d <= horizon_h)
    return auprc(truth.astype(np.int8), p_disc_24h)


def cindex_td(y13: np.ndarray, censor_bin: np.ndarray, risk: np.ndarray, cause: int,
              subject: np.ndarray | None = None, n_pairs: int = 2_000_000,
              seed: int = 0) -> dict:
    """竞争风险版的时间依赖 C-index（**抽样估计，不是全量枚举**）。

    可比对子的定义（本工程自定，写死在这里，报告里要照抄）：
      设 i 在桶 b 因 cause=k 发生事件，则 j 与 i 可比当且仅当
        ① j 到桶 b 为止仍未发生任何事件，且随访覆盖到桶 b（`censor_bin[j] >= b`）；**或**
        ② j 因**另一个** cause 发生事件（任意时刻）——那 j 就永远不会因 k 发生了。
      一致 = `risk[i] > risk[j]` 记 1，`risk[i] == risk[j]` 记 0.5，否则 0。

    ⚠ **对子是抽样的。** 5200 万行（test 也有百万行）不可能枚举 O(n²) 个对子。
      抽样方案：i 从「cause=k 的事件行」里均匀有放回抽，j 从全部行里均匀有放回抽，
      **丢掉同一患者的对子**（同患者内部的排序不是我们要评的东西），再筛可比。
      这是对「(事件行)×(全部行)」这个乘积空间的均匀抽样，而全部可比对子是它的子集，
      所以「可比对子中的一致比例」是无偏的。抽了多少对、其中多少可比，都写进返回值，
      **不许对外声称是全量**。

    `y13 == -1` 的行（契约 §2.2：cause 缺失 / delay_h<=0 的异常样本）整体排除。
    """
    y13 = np.asarray(y13, dtype=np.int16)
    cb = np.asarray(censor_bin, dtype=np.int16)
    risk = np.asarray(risk, dtype=np.float64)
    n = y13.size
    if subject is None:
        subject = np.arange(n, dtype=np.int64)       # 无患者信息时退化成「每行一个患者」
    subject = np.asarray(subject)

    usable = (y13 >= 0) & np.isfinite(risk)
    ev_cause = np.zeros(n, dtype=np.int8)
    ev_bin = np.full(n, np.iinfo(np.int16).max, dtype=np.int16)
    m1 = usable & (y13 >= 1) & (y13 <= 6)
    m2 = usable & (y13 >= 7) & (y13 <= 12)
    ev_cause[m1], ev_bin[m1] = 1, y13[m1] - 1
    ev_cause[m2], ev_bin[m2] = 2, y13[m2] - 7

    idx_i_pool = np.flatnonzero(usable & (ev_cause == cause))
    idx_j_pool = np.flatnonzero(usable)
    out = {"cause": int(cause), "n_events": int(idx_i_pool.size),
           "n_pairs_drawn": 0, "n_pairs_comparable": 0, "pairs_sampled": True,
           "pair_scheme": "i~事件行均匀, j~全部行均匀, 剔除同患者对子", "cindex": float("nan")}
    if idx_i_pool.size == 0 or idx_j_pool.size == 0:
        return out

    rng = np.random.default_rng(seed)
    n_pairs = int(n_pairs)
    ii = idx_i_pool[rng.integers(0, idx_i_pool.size, n_pairs)]
    jj = idx_j_pool[rng.integers(0, idx_j_pool.size, n_pairs)]

    keep = subject[ii] != subject[jj]
    ii, jj = ii[keep], jj[keep]
    b = ev_bin[ii]
    cond1 = (ev_bin[jj] > b) & (cb[jj] >= b)
    cond2 = (ev_cause[jj] != 0) & (ev_cause[jj] != cause)
    comp = cond1 | cond2
    ii, jj = ii[comp], jj[comp]

    out["n_pairs_drawn"] = int(n_pairs)
    out["n_pairs_comparable"] = int(ii.size)
    if ii.size == 0:
        return out
    ri, rj = risk[ii], risk[jj]
    out["cindex"] = float((np.count_nonzero(ri > rj) + 0.5 * np.count_nonzero(ri == rj)) / ii.size)
    return out


# ============================================================================ §C
# bootstrap：按 subject_id 重抽
# ============================================================================

def _unit_offsets(units: np.ndarray):
    """把行按重抽单元分组，返回 (codes, row_sorted, starts, sizes)。

    `row_sorted` 是按单元排好的行号，`starts/sizes` 是每个单元在其中的区间。
    慢路径（任意 metric callable）用它把「抽中的单元」展开成真实行号。
    """
    _, codes = np.unique(np.asarray(units), return_inverse=True)
    codes = codes.astype(np.int64, copy=False)
    n_unit = int(codes.max()) + 1 if codes.size else 0
    row_sorted = np.argsort(codes, kind="stable")
    sizes = np.bincount(codes, minlength=n_unit).astype(np.int64)
    starts = np.zeros(n_unit, dtype=np.int64)
    np.cumsum(sizes[:-1], out=starts[1:])
    return codes, row_sorted, starts, sizes


def _expand(row_sorted: np.ndarray, starts: np.ndarray, sizes: np.ndarray,
            picked: np.ndarray) -> np.ndarray:
    """把「抽中的单元编号」展开成行号（向量化的 ragged repeat，不写 python 循环）。"""
    cnt = sizes[picked]
    tot = int(cnt.sum())
    if tot == 0:
        return np.empty(0, dtype=np.int64)
    step = np.ones(tot, dtype=np.int64)
    step[0] = starts[picked[0]]
    if picked.size > 1:
        pos = np.cumsum(cnt)[:-1]
        step[pos] = starts[picked[1:]] - (starts[picked[:-1]] + cnt[:-1]) + 1
    return row_sorted[np.cumsum(step)]


def _metric_weighted(spec: dict, w: np.ndarray | None, metric: str) -> float:
    """在「每行带整数权重」的意义下算指标。w=None 表示全 1（即原样本）。"""
    if metric == "brier":
        y = spec["y_ord"].astype(np.float64)
        p = spec["p_ord"]
        if w is None:
            return float(np.mean((p - y) ** 2))
        ws = w.astype(np.float64)
        tot = ws.sum()
        return float(np.sum(ws * (p - y) ** 2) / tot) if tot > 0 else float("nan")
    tps, fps = _tpfp(spec, w)
    if metric == "auprc":
        return _ap_from(tps, fps)
    if metric == "auroc":
        return _auc_from(tps, fps)
    raise ValueError(f"快路径不支持的 metric {metric!r}")


def _boot_chunk(seed: int, reps: int, spec_a: dict, spec_b: dict,
                n_unit: int, metric: str) -> np.ndarray:
    """一个 worker 干 reps 次重抽，返回 reps 个 Δ。**必须是模块级函数**（loky 要 pickle）。"""
    rng = np.random.default_rng(seed)
    out = np.empty(reps, dtype=np.float64)
    ua, ub = spec_a["u_ord"], spec_b["u_ord"]
    for r in range(reps):
        mult = np.bincount(rng.integers(0, n_unit, n_unit), minlength=n_unit).astype(np.int32)
        out[r] = _metric_weighted(spec_a, mult[ua], metric) - \
            _metric_weighted(spec_b, mult[ub], metric)
    return out


def bootstrap_delta(pred_a, pred_b, labels, metric="auprc", n=None, unit="subject_id",
                    units=None, n_jobs=None, seed=None, verbose=True) -> dict:
    """两个模型在**同一批行**上的指标差 Δ = metric(A) − metric(B) 及其 bootstrap 95% CI。

    参数
    ----
    pred_a, pred_b : (n_rows,) 两个模型对同一批行的分数
    labels         : (n_rows,) 0/1 真值
    metric         : "auprc"（默认，主指标）/ "auroc" / "brier"；也可传 callable f(y,p)->float
                     （callable 走慢路径：每次真的 gather 出那些行，正确但慢得多）
    n              : bootstrap 次数；None 则读 `config.yaml` 的 `eval.bootstrap_n`
    unit           : "subject_id" 按患者重抽（契约 §6 要求）；"row" 按行重抽**只供对照**，
                     它会让 CI 假窄，不许用来报结果
    units          : (n_rows,) 每行的 subject_id；unit=="subject_id" 时必填

    返回
    ----
    {delta, ci_lo, ci_hi, p_boot, n_boot, n_subjects, ...}
    `n_boot` 是**真跑了多少次**，不是请求了多少次（AGENTS.md §5：不许写 1000 只跑 200）。
    """
    y = np.asarray(labels).astype(np.int8, copy=False)
    a = np.asarray(pred_a, dtype=np.float64)
    b = np.asarray(pred_b, dtype=np.float64)
    if not (y.shape == a.shape == b.shape):
        raise ValueError(f"形状不一致：labels{y.shape} a{a.shape} b{b.shape}")
    n_boot = int(n if n is not None else ecfg("bootstrap_n"))
    n_jobs = n_threads(n_jobs)
    seed = int(seed if seed is not None else C.cfg().get("seed", 0))

    if unit == "row":
        units_arr = np.arange(y.size, dtype=np.int64)
    elif unit == "subject_id":
        if units is None:
            raise ValueError("unit='subject_id' 必须传 units=（每行的 subject_id）")
        units_arr = np.asarray(units)
    else:
        raise ValueError(f"未知的重抽单位 {unit!r}")

    fast = isinstance(metric, str)
    codes, row_sorted, starts, sizes = (None, None, None, None)
    if fast:
        _, codes = np.unique(units_arr, return_inverse=True)
        codes = codes.astype(np.int32, copy=False)
        n_unit = int(codes.max()) + 1 if codes.size else 0
        spec_a = _prep_curve(y, a, units=codes)
        spec_b = _prep_curve(y, b, units=codes)
        point = _metric_weighted(spec_a, None, metric) - _metric_weighted(spec_b, None, metric)
        val_a = _metric_weighted(spec_a, None, metric)
        val_b = _metric_weighted(spec_b, None, metric)
    else:
        codes, row_sorted, starts, sizes = _unit_offsets(units_arr)
        n_unit = sizes.size
        val_a, val_b = float(metric(y, a)), float(metric(y, b))
        point = val_a - val_b

    t0 = time.time()
    if fast:
        # 把 n_boot 次均分给 n_jobs 个 worker，每个 worker 只启动一次
        per = [n_boot // n_jobs + (1 if i < n_boot % n_jobs else 0) for i in range(n_jobs)]
        per = [x for x in per if x > 0]
        if len(per) <= 1 or n_boot < 32:
            deltas = _boot_chunk(seed, n_boot, spec_a, spec_b, n_unit, metric)
        else:
            from joblib import Parallel, delayed
            chunks = Parallel(n_jobs=len(per), verbose=0)(
                delayed(_boot_chunk)(seed + 1000 * i, per[i], spec_a, spec_b, n_unit, metric)
                for i in range(len(per)))
            deltas = np.concatenate(chunks)
    else:
        rng = np.random.default_rng(seed)
        deltas = np.empty(n_boot, dtype=np.float64)
        for r in range(n_boot):
            idx = _expand(row_sorted, starts, sizes, rng.integers(0, n_unit, n_unit))
            deltas[r] = float(metric(y[idx], a[idx])) - float(metric(y[idx], b[idx]))
    elapsed = time.time() - t0

    good = deltas[np.isfinite(deltas)]
    n_used = int(good.size)
    if n_used == 0:
        return {"delta": point, "ci_lo": float("nan"), "ci_hi": float("nan"),
                "p_boot": float("nan"), "n_boot": 0, "n_boot_requested": n_boot,
                "n_subjects": int(n_unit), "n_rows": int(y.size),
                "metric": metric if isinstance(metric, str) else "callable",
                "bootstrap_unit": unit, "value_a": val_a, "value_b": val_b,
                "elapsed_s": round(elapsed, 1), "note": "全部重抽都算不出指标（正类为空？）"}
    lo, hi = np.percentile(good, [2.5, 97.5])
    p_boot = 2.0 * min(float(np.mean(good <= 0.0)), float(np.mean(good >= 0.0)))
    p_boot = min(1.0, max(p_boot, 1.0 / n_used))
    if verbose:
        print(f"  bootstrap：{n_used} 次 / 单位={unit} / {n_unit} 个患者 / "
              f"{y.size} 行 / {n_jobs} 线程 / 实耗 {elapsed:.1f}s")
    return {"delta": float(point), "ci_lo": float(lo), "ci_hi": float(hi),
            "p_boot": float(p_boot), "n_boot": n_used, "n_boot_requested": n_boot,
            "n_subjects": int(n_unit), "n_rows": int(y.size),
            "metric": metric if isinstance(metric, str) else "callable",
            "bootstrap_unit": unit, "value_a": float(val_a), "value_b": float(val_b),
            "boot_sd": float(np.std(good, ddof=1)) if n_used > 1 else float("nan"),
            "n_boot_nonfinite": int(n_boot - n_used),
            "elapsed_s": round(elapsed, 1)}


# ============================================================================ §D
# 契约 §7 自检
# ============================================================================

def check_pred_contract(cols: dict, tol: float = 1e-3) -> dict:
    """检查一份预测是否符合契约 §7。返回**违规计数**（聚合量），不返回违规的行。

    查：概率越界 / NaN / key 重复 / p13 不和为 1 /
       p_disc≠1−p13_0 / p_self≠Σp13_1..6 / p_other≠Σp13_7..12 / p_disc_24h≠Σp13_{1,2,3,7,8,9}
    """
    v, n = {}, len(cols["key"])
    v["n_rows"] = n
    v["n_key_duplicated"] = int(n - np.unique(cols["key"]).size)
    for c in ("p_disc", "p_self", "p_other", "p_disc_24h"):
        if c not in cols:
            v[f"{c}_missing"] = True
            continue
        x = np.asarray(cols[c], dtype=np.float64)
        v[f"n_{c}_nan"] = int(np.count_nonzero(~np.isfinite(x)))
        with np.errstate(invalid="ignore"):
            v[f"n_{c}_out_of_range"] = int(np.count_nonzero((x < -tol) | (x > 1 + tol)))
    has13 = all(c in cols for c in P13_COLS)
    v["has_p13"] = bool(has13)
    if has13:
        m = np.stack([np.asarray(cols[c], dtype=np.float64) for c in P13_COLS], axis=1)
        v["n_p13_sum_bad"] = int(np.count_nonzero(np.abs(m.sum(axis=1) - 1.0) > tol))
        pairs = {"p_disc": 1.0 - m[:, 0],
                 "p_self": m[:, 1:7].sum(axis=1),
                 "p_other": m[:, 7:13].sum(axis=1),
                 "p_disc_24h": m[:, [1, 2, 3]].sum(axis=1) + m[:, [7, 8, 9]].sum(axis=1)}
        for c, want in pairs.items():
            if c in cols:
                d = np.abs(np.asarray(cols[c], dtype=np.float64) - want)
                v[f"n_{c}_inconsistent"] = int(np.count_nonzero(d > tol))
                v[f"max_{c}_gap"] = float(np.nanmax(d)) if n else 0.0
    bad = [k for k, x in v.items()
           if (k.endswith("_missing")) or
           (k.startswith("n_") and k != "n_rows" and isinstance(x, int) and x != 0)]
    v["ok"] = not bad
    v["violations"] = bad
    return v


# ============================================================================ §E
# 数据装载：preds ←→ labels 对齐
# ============================================================================

def _need(path: Path, what: str):
    if not path.exists():
        print(f"[错误] 找不到{what}：{path}")
        print("       —— 上游脚本还没跑完。评测不造假数据，这里直接退出。")
        raise SystemExit(2)


def load_preds(model: str, split: str) -> dict:
    """读 `results/preds/<model>_<split>.parquet`，返回 {列名: numpy 数组}。"""
    import pyarrow.parquet as pq
    p = PREDS_DIR / f"{model}_{split}.parquet"
    _need(p, f"预测文件（模型 {model} / split {split}）")
    pf = pq.ParquetFile(p)
    have = set(pf.schema_arrow.names)
    miss = [c for c in P_REQUIRED if c not in have]
    if miss:
        raise SystemExit(f"[错误] {p.name} 缺契约 §7 必需列：{miss}")
    cols = list(P_REQUIRED) + [c for c in P13_COLS if c in have]
    t = pq.read_table(p, columns=cols)
    out = {c: t[c].to_numpy(zero_copy_only=False) for c in cols}
    out["key"] = out["key"].astype(np.int64, copy=False)
    out["subject_id"] = out["subject_id"].astype(np.int64, copy=False)
    print(f"  读到 {p.name}：{len(out['key'])} 行，"
          f"{'带' if len(cols) > len(P_REQUIRED) else '不带'} p13_*")
    return out


def _match_keys(lab_key: np.ndarray, pkey_sorted: np.ndarray, chunk: int = 4_000_000):
    """在 5200 万行的 labels 里定位这一批预测的 key。

    ⚠ labels 的 key **不是**排好序的（实测），但我们不能为它做一次 52M 的 argsort
      （那是 417 MB，本机可用内存只有 ~6 GB）。改成：对**小的**预测 key 排序，
      再分块对 labels 做 searchsorted，峰值只多 chunk×8 字节。
    返回 (labels 行号, 对应的 pkey_sorted 下标)。
    """
    m = pkey_sorted.size
    lab_rows, p_idx = [], []
    for s in range(0, lab_key.size, chunk):
        blk = lab_key[s:s + chunk]
        j = np.searchsorted(pkey_sorted, blk)
        np.minimum(j, m - 1, out=j)
        hit = pkey_sorted[j] == blk
        w = np.flatnonzero(hit)
        if w.size:
            lab_rows.append(w + s)
            p_idx.append(j[w])
    if not lab_rows:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    return np.concatenate(lab_rows), np.concatenate(p_idx)


LABEL_COLS = {"in_cohort": False, "y_disc": False, "delay_h": np.nan,
              "cause": -1, "y13": -1, "censor_bin": -1}


def attach_labels(pred_keys: np.ndarray, labels_path: Path = LABELS_PATH) -> dict:
    """按 `key` 把 labels 的标签列取出来，顺序与 `pred_keys` 一致。"""
    _need(labels_path, "labels.parquet")
    order = np.argsort(pred_keys, kind="stable")
    pkey_sorted = pred_keys[order]
    if np.any(pkey_sorted[1:] == pkey_sorted[:-1]):
        raise SystemExit("[错误] 预测文件里 key 有重复——契约 §7 要求每条医嘱一行")

    lab_key = C.col_np(labels_path, "key", fill=-1).astype(np.int64, copy=False)
    lab_rows, p_idx = _match_keys(lab_key, pkey_sorted)
    del lab_key
    m = pred_keys.size
    if lab_rows.size != m:
        raise SystemExit(f"[错误] {m} 条预测里只有 {lab_rows.size} 条能在 labels 里找到 key，"
                         f"对不上 {m - lab_rows.size} 条。模型和 labels 不是同一版，先查上游。")

    inv = np.empty(m, dtype=np.int64)
    inv[order] = np.arange(m)                     # pkey_sorted 下标 -> 原始行序
    dest = inv[p_idx]
    out = {}
    for name, fill in LABEL_COLS.items():
        col = C.col_np(labels_path, name, fill=fill)
        buf = np.empty(m, dtype=col.dtype)
        buf[dest] = col[lab_rows]
        out[name] = buf
        del col
    return out


def load_aligned(model: str, split: str) -> dict:
    """读一个模型的预测 + 对齐标签 + 只留 `in_cohort==True`，返回一个扁平 dict。"""
    pr = load_preds(model, split)
    lb = attach_labels(pr["key"])
    keep = lb["in_cohort"].astype(bool)
    n_drop = int(keep.size - np.count_nonzero(keep))
    if n_drop:
        print(f"  [注意] 有 {n_drop} 行 in_cohort=False，已剔除（契约 §7：只在队列内出预测）")
    d = {k: v[keep] for k, v in pr.items()}
    d.update({k: v[keep] for k, v in lb.items()})
    d["_n_dropped_out_of_cohort"] = n_drop
    return d


# ============================================================================ §F
# 评测编排
# ============================================================================

def _sup_list(xs):
    """列表里的计数逐个走小格子抑制。"""
    return [C.sup(int(x)) for x in xs]


def _r(x, nd=6):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), nd)


def evaluate_model(model: str, split: str, n_pairs: int = 2_000_000,
                   n_bins: int | None = None) -> dict:
    """单模型全套指标（契约 §6 那张表）。"""
    n_bins = int(n_bins if n_bins is not None else ecfg("calibration_bins"))
    d = load_aligned(model, split)
    y = d["y_disc"].astype(np.int8)
    p = np.asarray(d["p_disc"], dtype=np.float64)
    n = y.size
    subj = d["subject_id"]
    n_subj = int(np.unique(subj).size)
    n_pos = int(y.sum())
    prev = n_pos / n if n else float("nan")

    print(f"\n[{model} / {split}] 队列内 {n} 行、{n_subj} 位患者、"
          f"正类率 {C.pct(n_pos, n)}%")

    contract = check_pred_contract(d)
    if not contract.get("ok", False):
        bad = {k: v for k, v in contract.items()
               if k.startswith("n_") and k != "n_rows" and v}
        print(f"  [警告] 契约 §7 自检不通过：{bad}")

    res = {"model": model, "split": split,
           "n_rows": C.sup(n), "n_subjects": C.sup(n_subj),
           "n_pos": C.sup(n_pos), "prevalence": _r(prev),
           "n_dropped_out_of_cohort": C.sup(d["_n_dropped_out_of_cohort"]),
           "contract_check": contract}

    # --- H1 判别 ---
    res["auroc"] = _r(auroc(y, p))
    res["auprc"] = _r(auprc(y, p))
    res["auprc_baserate"] = _r(prev)          # 随机模型的 AUPRC 参考线 = 正类率
    print(f"  AUROC={res['auroc']}  AUPRC={res['auprc']}（随机线 {res['auprc_baserate']}）")

    # --- 逐因 AUPRC：cause==3（provider 缺失，契约 §2.2）整体排除 ---
    cause = d["cause"].astype(np.int16)
    known = cause != 3
    n_unk = int(np.count_nonzero(~known))
    res["n_excluded_cause_unknown"] = C.sup(n_unk)
    for nm, k, sc in (("self", 1, "p_self"), ("other", 2, "p_other")):
        yk = (cause[known] == k).astype(np.int8)
        res[f"auprc_{nm}"] = _r(auprc(yk, np.asarray(d[sc], dtype=np.float64)[known]))
        res[f"n_pos_{nm}"] = C.sup(int(yk.sum()))
    print(f"  AUPRC_self={res['auprc_self']}  AUPRC_other={res['auprc_other']}"
          f"（排除 cause=3 的 {res['n_excluded_cause_unknown']} 行）")

    # --- 校准 ---
    res["brier"] = _r(brier(y, p))
    res["calibration_bins"] = n_bins
    res["ece"] = _r(ece(y, p, n_bins=n_bins, strategy="uniform"))
    res["ece_quantile"] = _r(ece(y, p, n_bins=n_bins, strategy="quantile"))
    for tag, strat in (("reliability", "uniform"), ("reliability_quantile", "quantile")):
        rc = reliability_curve(y, p, n_bins=n_bins, strategy=strat)
        rc["count"] = _sup_list(rc["count"])            # 小格子抑制
        rc["mean_pred"] = [_r(x) for x in rc["mean_pred"]]
        rc["frac_pos"] = [_r(x) for x in rc["frac_pos"]]
        rc["bin_edges"] = [_r(x) for x in rc["bin_edges"]]
        rc["bin_center"] = [_r(x) for x in rc["bin_center"]]
        rc["n"] = C.sup(rc["n"])
        res[tag] = rc
    print(f"  Brier={res['brier']}  ECE(等宽{n_bins}桶)={res['ece']}  "
          f"ECE(等频)={res['ece_quantile']}")

    # --- 时间依赖 C-index（竞争风险版，抽样对子）---
    y13, cb = d["y13"].astype(np.int16), d["censor_bin"].astype(np.int16)
    seed = int(C.cfg().get("seed", 0))
    ci_s = cindex_td(y13, cb, np.asarray(d["p_self"], dtype=np.float64), 1,
                     subject=subj, n_pairs=n_pairs, seed=seed)
    ci_o = cindex_td(y13, cb, np.asarray(d["p_other"], dtype=np.float64), 2,
                     subject=subj, n_pairs=n_pairs, seed=seed + 1)
    res["cindex_td_self"] = _r(ci_s["cindex"])
    res["cindex_td_other"] = _r(ci_o["cindex"])
    wgt = np.array([ci_s["n_pairs_comparable"], ci_o["n_pairs_comparable"]], dtype=np.float64)
    vals = np.array([ci_s["cindex"], ci_o["cindex"]], dtype=np.float64)
    ok = np.isfinite(vals) & (wgt > 0)
    res["cindex_td"] = _r(float(np.sum(vals[ok] * wgt[ok]) / wgt[ok].sum()) if ok.any() else np.nan)
    res["cindex_detail"] = {
        "self": {**{k: v for k, v in ci_s.items() if k != "cindex"},
                 "n_events": C.sup(ci_s["n_events"]),
                 "n_pairs_comparable": C.sup(ci_s["n_pairs_comparable"])},
        "other": {**{k: v for k, v in ci_o.items() if k != "cindex"},
                  "n_events": C.sup(ci_o["n_events"]),
                  "n_pairs_comparable": C.sup(ci_o["n_pairs_comparable"])},
        "备注": "对子是抽样的，不是全量枚举；抽样规模见 n_pairs_drawn"}
    print(f"  C-index(td)：self={res['cindex_td_self']} other={res['cindex_td_other']} "
          f"加权={res['cindex_td']}（各抽 {n_pairs} 对，可比 "
          f"{res['cindex_detail']['self']['n_pairs_comparable']}/"
          f"{res['cindex_detail']['other']['n_pairs_comparable']}）")

    # --- 24h 内撤销 ---
    d24 = np.asarray(d["delay_h"], dtype=np.float64)
    with np.errstate(invalid="ignore"):
        y24 = (y.astype(bool) & (d24 <= 24.0))
    v24 = auprc_at_24h(y.astype(bool), d24, np.asarray(d["p_disc_24h"], dtype=np.float64))
    res["auprc_at_24h"] = _r(v24)
    res["auprc@24h"] = res["auprc_at_24h"]          # 契约 §6 的原名，给下游两种写法都留着
    res["n_pos_24h"] = C.sup(int(y24.sum()))
    res["auprc_at_24h_baserate"] = _r(float(y24.mean()) if n else np.nan)
    print(f"  AUPRC@24h={res['auprc_at_24h']}（随机线 {res['auprc_at_24h_baserate']}）")
    return res


def compare_models(model_a: str, model_b: str, split: str, metric: str = "auprc",
                   n_boot: int | None = None, n_jobs: int | None = None) -> dict:
    """预注册主实验（契约 §7 末尾）：Δ = metric(A) − metric(B)，按患者 bootstrap。"""
    da, db = load_aligned(model_a, split), load_aligned(model_b, split)
    ka, kb = da["key"], db["key"]
    if ka.size != kb.size or not np.array_equal(np.sort(ka), np.sort(kb)):
        common = np.intersect1d(ka, kb, assume_unique=True)
        print(f"  [警告] 两个模型的行不完全一致（A={ka.size} B={kb.size}），"
              f"取交集 {common.size} 行比较")
        sa = np.searchsorted(np.sort(ka), common)
        ia = np.argsort(ka, kind="stable")[sa]
        sb = np.searchsorted(np.sort(kb), common)
        ib = np.argsort(kb, kind="stable")[sb]
        n_inter = int(common.size)
    else:
        ia = np.argsort(ka, kind="stable")
        ib = np.argsort(kb, kind="stable")
        n_inter = int(ka.size)

    y = da["y_disc"].astype(np.int8)[ia]
    subj = da["subject_id"][ia]
    pa = np.asarray(da["p_disc"], dtype=np.float64)[ia]
    pb = np.asarray(db["p_disc"], dtype=np.float64)[ib]
    assert np.array_equal(ka[ia], kb[ib]), "对齐后 key 仍不一致——这是 bug，不要往下算"

    thr = float(ecfg("prereg_delta_auprc"))
    unit = str(ecfg("bootstrap_unit"))
    print(f"\n[预注册主实验] {model_a} vs {model_b} @ {split}，指标={metric}，"
          f"{n_inter} 行 / {np.unique(subj).size} 位患者")
    r = bootstrap_delta(pa, pb, y, metric=metric, n=n_boot, unit=unit,
                        units=subj, n_jobs=n_jobs)

    ci_excl0 = np.isfinite(r["ci_lo"]) and np.isfinite(r["ci_hi"]) and \
        (r["ci_lo"] > 0 or r["ci_hi"] < 0)
    met = bool(r["delta"] >= thr and ci_excl0 and r["delta"] > 0)
    reasons = []
    if r["delta"] < thr:
        reasons.append(f"Δ={r['delta']:.4f} < 阈值 {thr}")
    if not ci_excl0:
        reasons.append(f"95% CI [{r['ci_lo']:.4f}, {r['ci_hi']:.4f}] 含 0")
    out = {"model_a": model_a, "model_b": model_b, "split": split, "metric": metric,
           "n_rows": C.sup(n_inter), **{k: v for k, v in r.items() if k != "n_rows"},
           f"delta_{metric}": r["delta"],
           "prereg_threshold": thr,
           "prereg_criterion": f"Δ{metric.upper()} >= {thr} 且按 {unit} 重抽的 95% CI 不含 0",
           "prereg_met": met,
           "prereg_reason": "达成" if met else "；".join(reasons) or "未达成"}
    out["n_subjects"] = C.sup(r["n_subjects"])
    print(f"  {metric}(A)={r['value_a']:.6f}  {metric}(B)={r['value_b']:.6f}")
    print(f"  Δ={r['delta']:+.6f}  95% CI [{r['ci_lo']:+.6f}, {r['ci_hi']:+.6f}]  "
          f"p_boot={r['p_boot']:.4f}  （真跑 {r['n_boot']} 次）")
    print(f"  判据：{out['prereg_criterion']}")
    print(f"  ★ 预注册判据：{'达成' if met else '未达成'}"
          f"{'' if met else '（' + out['prereg_reason'] + '）'}")
    return out


def smoke_labels() -> dict:
    """冒烟验证：真的把 labels 读进来，打几个聚合计数（不依赖任何模型预测）。"""
    _need(LABELS_PATH, "labels.parquet")
    t0 = time.time()
    inc = C.col_np(LABELS_PATH, "in_cohort", fill=False).astype(bool)
    n = inc.size
    n_in = int(np.count_nonzero(inc))
    out = {"n_rows": C.sup(n), "n_in_cohort": C.sup(n_in),
           "pct_in_cohort": C.pct(n_in, n)}
    yd = C.col_np(LABELS_PATH, "y_disc", fill=False).astype(bool)[inc]
    out["n_disc_in_cohort"] = C.sup(int(yd.sum()))
    out["pct_disc_in_cohort"] = C.pct(int(yd.sum()), n_in)
    del yd
    cz = C.col_np(LABELS_PATH, "cause", fill=-1)[inc]
    out["cause_hist"] = {f"cause_{k}": C.sup(int(np.count_nonzero(cz == k))) for k in (0, 1, 2, 3)}
    del cz
    y13 = C.col_np(LABELS_PATH, "y13", fill=-1)[inc]
    out["y13_hist"] = {f"y13_{k}": C.sup(int(np.count_nonzero(y13 == k)))
                       for k in range(-1, 13)}
    del y13
    cb = C.col_np(LABELS_PATH, "censor_bin", fill=-1)[inc]
    out["censor_bin_hist"] = {f"bin_{k}": C.sup(int(np.count_nonzero(cb == k)))
                              for k in range(-1, 6)}
    del cb
    out["_read_s"] = round(time.time() - t0, 1)
    print("\n[冒烟] labels.parquet 聚合计数（只有计数，没有任何患者级取值）：")
    for k, v in out.items():
        print(f"  {k} = {v}")
    return out


# ============================================================================ §G
# --selftest：合成数据自测，完全不碰真实数据
# ============================================================================

class _T:
    def __init__(self):
        self.n = 0
        self.fail = []

    def ok(self, cond, name, detail=""):
        self.n += 1
        flag = "OK  " if cond else "FAIL"
        print(f"  [{self.n:2d}] {flag} {name}" + (f"   {detail}" if detail else ""))
        if not cond:
            self.fail.append(name)
        return bool(cond)


def selftest() -> int:
    """合成数据自测。返回失败条数（0 = 全过）。"""
    t = _T()
    rng = np.random.default_rng(20260921)
    print("\n=== s07_eval 自测（合成数据，不读 data/、不读 results/preds/）===")

    # ---------- 1. AUROC / AUPRC 的已知答案 ----------
    y = np.r_[np.zeros(500, np.int8), np.ones(500, np.int8)]
    p_sep = np.r_[rng.uniform(0.0, 0.4, 500), rng.uniform(0.6, 1.0, 500)]
    t.ok(abs(auroc(y, p_sep) - 1.0) < 1e-12, "完全可分 ⇒ AUROC=1.0", f"{auroc(y, p_sep):.6f}")
    t.ok(abs(auroc(y, -p_sep) - 0.0) < 1e-12, "完全反着分 ⇒ AUROC=0.0", f"{auroc(y, -p_sep):.6f}")
    t.ok(abs(auprc(y, p_sep) - 1.0) < 1e-12, "完全可分 ⇒ AUPRC=1.0", f"{auprc(y, p_sep):.6f}")
    t.ok(abs(auroc(y, np.full(1000, 0.3)) - 0.5) < 1e-12,
         "分数全并列 ⇒ AUROC=0.5", f"{auroc(y, np.full(1000, 0.3)):.6f}")

    n = 200_000
    yr = (rng.random(n) < 0.2).astype(np.int8)
    pr = rng.random(n)
    t.ok(abs(auroc(yr, pr) - 0.5) < 0.01, "随机分数 ⇒ AUROC≈0.5", f"{auroc(yr, pr):.4f}")
    t.ok(abs(auprc(yr, pr) - 0.2) < 0.01, "随机分数 ⇒ AUPRC≈正类率 0.2", f"{auprc(yr, pr):.4f}")

    # ---------- 2. 与 sklearn 对拍（含大量并列的场景）----------
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score
        ys = (rng.random(20_000) < 0.2).astype(np.int8)
        ps = np.round(rng.random(20_000), 2)          # 只有 101 个不同取值 ⇒ 大量并列
        t.ok(abs(auroc(ys, ps) - roc_auc_score(ys, ps)) < 1e-10,
             "AUROC 与 sklearn 一致（含并列）", f"Δ={abs(auroc(ys, ps) - roc_auc_score(ys, ps)):.2e}")
        t.ok(abs(auprc(ys, ps) - average_precision_score(ys, ps)) < 1e-10,
             "AUPRC 与 sklearn 一致（含并列）",
             f"Δ={abs(auprc(ys, ps) - average_precision_score(ys, ps)):.2e}")
    except ImportError:
        print("  [--] SKIP sklearn 对拍（未安装 scikit-learn）")

    # ---------- 3. Brier / ECE / 可靠性曲线 ----------
    pc_ = rng.random(200_000)
    yc = (rng.random(200_000) < pc_).astype(np.int8)          # 完美校准
    t.ok(ece(yc, pc_, 10) < 0.01, "完美校准 ⇒ ECE≈0", f"{ece(yc, pc_, 10):.5f}")
    pm = np.clip(pc_ + 0.2, 0, 1)                             # 系统性高估 0.2
    t.ok(ece(yc, pm, 10) > 0.1, "系统性高估 0.2 ⇒ ECE 明显 >0.1", f"{ece(yc, pm, 10):.5f}")
    t.ok(abs(brier(yc, pc_) - np.mean((pc_ - yc) ** 2)) < 1e-12, "Brier = mean((p−y)²)")
    rc = reliability_curve(yc, pc_, 10)
    t.ok(sum(rc["count"]) == 200_000 and len(rc["bin_center"]) == 10,
         "可靠性曲线：10 桶、计数求和 = n")
    t.ok(np.all(np.diff([x for x in rc["frac_pos"] if np.isfinite(x)]) > 0),
         "可靠性曲线：完美校准下实际率随桶单调上升")

    # ---------- 4. 契约 §7 一致性检查能抓到被人为破坏的输入 ----------
    nn = 5000
    raw = rng.random((nn, 13)) ** 3
    m13 = raw / raw.sum(axis=1, keepdims=True)
    good = {"key": np.arange(nn, dtype=np.int64),
            "p_disc": 1.0 - m13[:, 0],
            "p_self": m13[:, 1:7].sum(1),
            "p_other": m13[:, 7:13].sum(1),
            "p_disc_24h": m13[:, [1, 2, 3]].sum(1) + m13[:, [7, 8, 9]].sum(1)}
    good.update({f"p13_{i}": m13[:, i] for i in range(13)})
    v = check_pred_contract(good)
    t.ok(v["ok"] and v["n_p13_sum_bad"] == 0, "合法的 13 类矩阵 ⇒ 契约自检通过")

    bad1 = dict(good, p_disc=good["p_disc"] + 0.05)
    t.ok(check_pred_contract(bad1)["n_p_disc_inconsistent"] == nn,
         "破坏 p_disc = 1 − p13_0 ⇒ 被抓到")
    bad2 = dict(good)
    bad2["p13_5"] = bad2["p13_5"] + 0.3
    t.ok(check_pred_contract(bad2)["n_p13_sum_bad"] == nn, "破坏 Σp13 = 1 ⇒ 被抓到")
    bad3 = dict(good, p_self=good["p_self"] * 0.5)
    t.ok(check_pred_contract(bad3)["n_p_self_inconsistent"] > nn * 0.9,
         "破坏 p_self = Σp13_1..6 ⇒ 被抓到")
    bad4 = dict(good, p_other=good["p_other"] + 2.0)
    t.ok(check_pred_contract(bad4)["n_p_other_out_of_range"] == nn, "概率越界 ⇒ 被抓到")
    bad5 = dict(good, p_disc_24h=good["p_disc_24h"] * 0.0 + 0.5)
    t.ok(check_pred_contract(bad5)["n_p_disc_24h_inconsistent"] > nn * 0.9,
         "破坏 p_disc_24h = Σp13_{1,2,3,7,8,9} ⇒ 被抓到")

    # ---------- 5. 加权快路径 ≡ 真 gather 慢路径（bootstrap 的正确性地基）----------
    nr, ns = 30_000, 3_000
    subj = rng.integers(0, ns, nr)
    yw = (rng.random(nr) < 0.2).astype(np.int8)
    pw = np.clip(0.2 + 0.5 * yw + rng.normal(0, 0.3, nr), 0, 1)
    codes = np.unique(subj, return_inverse=True)[1].astype(np.int32)
    nu = codes.max() + 1
    draw = np.random.default_rng(7).integers(0, nu, nu)
    mult = np.bincount(draw, minlength=nu).astype(np.int32)
    spec = _prep_curve(yw, pw, units=codes)
    fast_ap = _metric_weighted(spec, mult[spec["u_ord"]], "auprc")
    _, rs, st, sz = _unit_offsets(subj)
    idx = _expand(rs, st, sz, draw)
    slow_ap = auprc(yw[idx], pw[idx])
    t.ok(abs(fast_ap - slow_ap) < 1e-10, "加权快路径 ≡ gather 慢路径（AUPRC）",
         f"Δ={abs(fast_ap - slow_ap):.2e}")
    fast_auc = _metric_weighted(spec, mult[spec["u_ord"]], "auroc")
    t.ok(abs(fast_auc - auroc(yw[idx], pw[idx])) < 1e-10, "加权快路径 ≡ gather 慢路径（AUROC）")
    t.ok(idx.size == int(sz[draw].sum()), "单元展开的行数 = 抽中单元的行数之和")

    # 回归测试：**分数最高的那一段权重全为 0**。
    # 这是端到端演练里真实咬人的那个 bug —— precision 开头 0/0=nan，
    # 乘 Δrecall=0 还是 nan，整条 AP 被污染，1000 次 bootstrap 有 569 次返回 nan。
    top_units = np.unique(spec["u_ord"][:max(1, nr // 50)])   # 分数最高的 2% 行所属的患者
    adv_draw = np.setdiff1d(np.arange(nu), top_units)
    adv_draw = np.repeat(adv_draw, 2)                          # 把这些患者整体踢出重抽样本
    adv_mult = np.bincount(adv_draw, minlength=nu).astype(np.int32)
    adv_fast = _metric_weighted(spec, adv_mult[spec["u_ord"]], "auprc")
    adv_idx = _expand(rs, st, sz, adv_draw)
    adv_slow = auprc(yw[adv_idx], pw[adv_idx])
    t.ok(np.isfinite(adv_fast) and abs(adv_fast - adv_slow) < 1e-10,
         "最高分那段权重全为 0 时仍算得出 AUPRC，且 ≡ gather 路径",
         f"fast={adv_fast:.6f} slow={adv_slow:.6f}")
    adv_fast_auc = _metric_weighted(spec, adv_mult[spec["u_ord"]], "auroc")
    t.ok(np.isfinite(adv_fast_auc) and abs(adv_fast_auc - auroc(yw[adv_idx], pw[adv_idx])) < 1e-10,
         "最高分那段权重全为 0 时 AUROC 也 ≡ gather 路径")

    # ---------- 6. bootstrap_delta ----------
    nr, ns = 60_000, 4_000
    subj = rng.integers(0, ns, nr)
    yb = (rng.random(nr) < 0.2).astype(np.int8)
    strong = np.clip(0.1 + 0.7 * yb + rng.normal(0, 0.25, nr), 0, 1)   # A 明显强
    weak = np.clip(0.1 + 0.7 * yb + rng.normal(0, 1.20, nr), 0, 1)     # B 明显弱
    rA = bootstrap_delta(strong, weak, yb, metric="auprc", n=200,
                         unit="subject_id", units=subj, n_jobs=8, seed=1, verbose=False)
    t.ok(rA["delta"] > 0 and rA["ci_lo"] > 0,
         "A 明显强于 B ⇒ Δ>0 且 95% CI 不含 0",
         f"Δ={rA['delta']:.4f} CI[{rA['ci_lo']:.4f},{rA['ci_hi']:.4f}] p={rA['p_boot']:.3f}")
    rB = bootstrap_delta(strong, strong.copy(), yb, metric="auprc", n=200,
                         unit="subject_id", units=subj, n_jobs=8, seed=1, verbose=False)
    t.ok(rB["delta"] == 0.0 and rB["ci_lo"] <= 0 <= rB["ci_hi"],
         "A ≡ B ⇒ Δ=0 且 95% CI 含 0",
         f"Δ={rB['delta']:.6f} CI[{rB['ci_lo']:.6f},{rB['ci_hi']:.6f}]")
    noise = np.clip(0.2 + rng.normal(0, 1.0, nr), 0, 1)
    rC = bootstrap_delta(noise, np.clip(0.2 + rng.normal(0, 1.0, nr), 0, 1), yb,
                         metric="auprc", n=300, unit="subject_id", units=subj,
                         n_jobs=8, seed=3, verbose=False)
    t.ok(rC["ci_lo"] <= 0 <= rC["ci_hi"], "两个纯噪声模型 ⇒ 95% CI 含 0",
         f"CI[{rC['ci_lo']:.4f},{rC['ci_hi']:.4f}]")
    t.ok(rA["n_boot"] == 200 and rA["n_subjects"] == ns,
         "返回值里的 n_boot / n_subjects 是真实值", f"{rA['n_boot']}, {rA['n_subjects']}")
    t.ok(rA["n_boot_nonfinite"] == 0 and rC["n_boot_nonfinite"] == 0,
         "没有任何一次重抽算出 nan（CI 是全部 n_boot 次的分位数，不是残存那几次的）",
         f"{rA['n_boot_nonfinite']} / {rC['n_boot_nonfinite']}")
    rD = bootstrap_delta(strong, weak, yb, metric=lambda yy, pp: auprc(yy, pp), n=40,
                         unit="subject_id", units=subj, n_jobs=4, seed=1, verbose=False)
    t.ok(abs(rD["delta"] - rA["delta"]) < 1e-10,
         "callable metric 慢路径与快路径的点估计一致")
    r_row = bootstrap_delta(strong, weak, yb, metric="auprc", n=200, unit="row",
                            n_jobs=8, seed=1, verbose=False)
    t.ok((r_row["ci_hi"] - r_row["ci_lo"]) < (rA["ci_hi"] - rA["ci_lo"]),
         "按行重抽的 CI 比按患者重抽更窄（所以必须按患者，否则假窄）",
         f"行 {r_row['ci_hi'] - r_row['ci_lo']:.4f} < 患者 {rA['ci_hi'] - rA['ci_lo']:.4f}")

    # ---------- 7. cindex_td ----------
    nsub, per = 400, 3
    nn = nsub * per
    sub_c = np.repeat(np.arange(nsub), per)
    y13c = np.zeros(nn, np.int16)
    y13c[0::3] = rng.integers(1, 7, nsub)        # 自撤，桶 0..5
    y13c[1::3] = rng.integers(7, 13, nsub)       # 他撤
    y13c[2::3] = 0                               # 未撤
    cbc = np.full(nn, 5, np.int16)
    ev_bin = np.where((y13c >= 1) & (y13c <= 6), y13c - 1,
                      np.where(y13c >= 7, y13c - 7, 99))
    risk_perfect = np.where((y13c >= 1) & (y13c <= 6), -ev_bin.astype(float), -100.0)
    c1 = cindex_td(y13c, cbc, risk_perfect, 1, subject=sub_c, n_pairs=200_000, seed=0)
    t.ok(abs(c1["cindex"] - 1.0) < 1e-12, "风险完美排序 ⇒ C-index=1.0",
         f"{c1['cindex']:.6f}，可比 {c1['n_pairs_comparable']} 对")
    c2 = cindex_td(y13c, cbc, -risk_perfect, 1, subject=sub_c, n_pairs=200_000, seed=0)
    t.ok(abs(c2["cindex"] - 0.0) < 1e-12, "风险完全反着排 ⇒ C-index=0.0", f"{c2['cindex']:.6f}")
    c3 = cindex_td(y13c, cbc, rng.random(nn), 1, subject=sub_c, n_pairs=400_000, seed=0)
    t.ok(abs(c3["cindex"] - 0.5) < 0.02, "随机风险 ⇒ C-index≈0.5", f"{c3['cindex']:.4f}")
    c4 = cindex_td(y13c, cbc, np.zeros(nn), 1, subject=sub_c, n_pairs=100_000, seed=0)
    t.ok(abs(c4["cindex"] - 0.5) < 1e-12, "风险全并列 ⇒ C-index=0.5（并列计 0.5）")
    t.ok(c1["pairs_sampled"] is True and c1["n_pairs_drawn"] == 200_000,
         "C-index 的抽样规模如实写进返回值（不谎称全量）")
    cbe = np.zeros(nn, np.int16)            # 所有人在桶 0 就删失
    c5 = cindex_td(y13c, cbe, risk_perfect, 1, subject=sub_c, n_pairs=50_000, seed=0)
    t.ok(c5["n_pairs_comparable"] < c1["n_pairs_comparable"],
         "随访被砍到桶 0 ⇒ 可比对子数显著减少（删失被正确尊重）",
         f"{c5['n_pairs_comparable']} < {c1['n_pairs_comparable']}")

    # ---------- 8. auprc_at_24h ----------
    nn = 20_000
    yd = (rng.random(nn) < 0.2)
    dh = np.where(yd, rng.exponential(48.0, nn), np.nan)
    with np.errstate(invalid="ignore"):
        truth = yd & (dh <= 24.0)
    t.ok(abs(auprc_at_24h(yd, dh, truth.astype(float)) - 1.0) < 1e-12,
         "auprc@24h：真值当分数 ⇒ 1.0")
    t.ok(abs(auprc_at_24h(yd, dh, rng.random(nn)) - truth.mean()) < 0.02,
         "auprc@24h：随机分数 ⇒ ≈24h 内撤销率", f"{truth.mean():.4f}")
    t.ok(not np.isnan(auprc_at_24h(yd, dh, rng.random(nn))),
         "auprc@24h：未撤销行的 delay_h=NaN 被当作负类，不产生 NaN")

    # ---------- 收尾 ----------
    print()
    if t.fail:
        print(f"SELFTEST FAILED {t.n - len(t.fail)}/{t.n}  失败项：{t.fail}")
    else:
        print(f"SELFTEST OK {t.n}/{t.n}")
    return len(t.fail)


# ============================================================================
# main
# ============================================================================

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="BN5212 统一评测：所有模型走同一套指标函数（docs/01_数据契约.md §6/§7）")
    ap.add_argument("--model", help=f"单模型评测，模型名如 {'/'.join(KNOWN_MODELS[:4])}")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"),
                    help="预注册主实验：比较 A 与 B 的 ΔAUPRC（Δ = A − B）")
    ap.add_argument("--split", default="test", choices=["val", "test"],
                    help="契约 §7：训练集不出预测文件")
    ap.add_argument("--selftest", action="store_true", help="合成数据自测，不碰真实数据")
    ap.add_argument("--smoke", action="store_true", help="读 labels.parquet 打几个聚合计数")
    ap.add_argument("--metric", default="auprc", choices=["auprc", "auroc", "brier"],
                    help="--compare 用的指标，默认主指标 auprc")
    ap.add_argument("--n-boot", type=int, default=None,
                    help="bootstrap 次数，默认读 config.yaml 的 eval.bootstrap_n")
    ap.add_argument("--n-jobs", type=int, default=None, help="并行线程数，上限 16")
    ap.add_argument("--n-pairs", type=int, default=2_000_000,
                    help="C-index 抽样的对子数（不是全量枚举）")
    a = ap.parse_args(argv)

    if not (a.selftest or a.smoke or a.model or a.compare):
        ap.print_help()
        return 1

    if a.selftest:
        t0, _ = C.start_log("s07_eval_selftest")
        nfail = selftest()
        C.save_metrics("s07_eval_selftest",
                       {"passed": nfail == 0, "n_failed": nfail,
                        "note": "合成数据自测，不读真实数据"}, t0)
        C.append_runlog("s07_eval_selftest",
                        "全过" if nfail == 0 else f"**失败 {nfail} 条**")
        return 0 if nfail == 0 else 1

    if a.smoke:
        t0, _ = C.start_log("s07_eval_smoke")
        out = smoke_labels()
        C.save_metrics("s07_eval_smoke", out, t0)
        C.append_runlog("s07_eval_smoke",
                        f"labels 队列内 {out['n_in_cohort']} 行、撤销率 {out['pct_disc_in_cohort']}%")
        return 0

    if a.model:
        if a.model not in KNOWN_MODELS:
            print(f"[提示] 模型名 {a.model!r} 不在契约 §7 的登记表里：{KNOWN_MODELS}")
        step = f"s07_eval_{a.model}_{a.split}"
        t0, _ = C.start_log(step)
        res = evaluate_model(a.model, a.split, n_pairs=a.n_pairs)
        C.save_metrics(step, res, t0)
        C.append_runlog(step, f"AUPRC={res['auprc']} AUROC={res['auroc']} "
                              f"Brier={res['brier']} ECE={res['ece']} "
                              f"C-index={res['cindex_td']} AUPRC@24h={res['auprc_at_24h']}")
        return 0

    A, B = a.compare
    step = f"s07_compare_{A}_vs_{B}_{a.split}"
    t0, _ = C.start_log(step)
    res = compare_models(A, B, a.split, metric=a.metric, n_boot=a.n_boot, n_jobs=a.n_jobs)
    C.save_metrics(step, res, t0)
    C.append_runlog(step, f"Δ{a.metric}={res['delta']:+.4f} "
                          f"CI[{res['ci_lo']:+.4f},{res['ci_hi']:+.4f}] "
                          f"n_boot={res['n_boot']} 预注册判据"
                          f"{'达成' if res['prereg_met'] else '未达成'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
