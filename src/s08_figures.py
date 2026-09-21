#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""s08_figures.py —— 出图（只读，不产生任何新的中间数据）。

依赖的接口（全部只读，写码前已逐条核对过真实字段名）：
  - `results/metrics/s02_labels.json`  （契约 §2：撤销率_by_order_type / t_bin_counts_self·other /
    cohort_flow / in_cohort）
  - `data/labels.parquet`              （契约 §2：本脚本只读 `delay_h` / `in_cohort` 两列算直方图，
    绝不整表读入，绝不打印/落盘任何一行原始记录）
  - `results/metrics/s07_eval_<model>_<split>.json`  （契约 §6：reliability 校准曲线、ece）
  - `results/metrics/s07_compare_<A>_vs_<B>_<split>.json`  （契约 §7 末尾：预注册主实验）
  - `results/metrics/s05_baseline.json`  （models.<name>.top30_gain：LightGBM 特征重要性）
  - `results/preds/<model>_<split>.parquet`  （契约 §7：模型预测，唯一稳定接口）

第二组（F6/F7/F8/F9）依赖别的脚本（s05/s06/s07）的产出，**现在很可能还没跑完**。
设计原则：每张图一个函数，函数开头检查自己需要的输入在不在/齐不齐，
不齐就打印"跳过 Fx：缺 xxx"并把这一条记进 manifest，绝不让一张图的缺失拖垮整个脚本。

⚠ F6（PR 曲线）刻意**不读** `s07_eval_*.json`：那份文件目前只存标量 AUPRC/AUROC，
  不存曲线点位（读了源码确认过，`evaluate_model()` 没有把 precision/recall 数组写进 JSON）。
  改成直接读 `results/preds/<model>_<split>.parquet`（契约 §7，稳定接口）+
  `data/labels.parquet` 的 `key`/`y_disc` 两列，自己按分数排序算曲线——
  这样只要任何一个模型出了预测文件就能画，不必等别人改 s07_eval.py 的输出 schema
  （改那份文件不是本脚本的职责，s01–s07 都有别人在写，本脚本只准写这一个文件）。

用法
----
```
python src/s08_figures.py            # 画全部 9 张（能画的）
python src/s08_figures.py --only F1,F3
```

画图规矩（AGENTS.md + 派单要求）：
  - Windows 中文字体：Microsoft YaHei / SimHei，负号不用 unicode minus。
  - matplotlib + Agg 后端，不用 seaborn，不用交互式窗口。
  - 150 dpi、白底、bbox_inches="tight"，存到 `C.FIGURES`（不自己拼路径）。
  - 配色用 dataviz skill 校验过（`scripts/validate_palette.js`，adjacent 配对，light 模式全过）：
    自撤/他撤用 diverging 蓝/红两极；多模型对比按契约 §7 登记顺序固定分配类别色，
    颜色跟着"模型是谁"走、不跟着"这一轮实际画了几个"走；单序列量级图统一一个蓝色调。
    所有"自撤 vs 他撤"的图都额外加 hatch/marker 做第二编码，保证黑白打印也能区分。
  - 隐私：只画聚合量；任何 n<10 的格子/桶一律隐去（本脚本自己计算直方图时同样执行这条）。
  - 内存：`labels.parquet` 5200 万行，只读需要的那一列（`C.col_np`），不整表读入。
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import tempfile
from pathlib import Path

# 必须在 import common（进而可能 import torch）之前设置——本脚本纯 CPU，
# 不能让 common.env_fingerprint() 里的 torch.cuda.is_available() 去碰那张 6 GB 卡
# （同一时刻只许一个 GPU 作业，见 AGENTS.md §4.5；s07_eval.py 用的同一招）。
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

DPI = 150
SPLIT = "test"                       # F6/F7/F8 统一看 test 集（契约 §7：训练集不出预测）
PREDS_DIR = C.RESULTS / "preds"
LABELS_PATH = C.DATA / "labels.parquet"
TOP_N_ORDER_TYPE = 10
BUCKET_LABELS = ["≤1h", "1-6h", "6-24h", "24-72h", "72h-7d", ">7d"]   # 契约 §2.1

# --------------------------------------------------------------------------- 配色
# 已用 dataviz skill 的 scripts/validate_palette.js 校验（light 模式，adjacent 配对全过，
# 见派单记录）。自撤/他撤用 diverging 蓝/红两极——呼应项目主张"两个方向相反的过程"；
# 其余单序列量级图统一一个蓝色调（sequential=一个色相），不给"排第几"发不同颜色。
COLOR_SELF = "#2a78d6"     # blue
COLOR_OTHER = "#e34948"    # red
SEQ_BLUE = "#2a78d6"
ORDINAL_BLUE_RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab"]   # F5 漏斗，浅→深
STATUS_GOOD = "#0ca30c"
STATUS_CRITICAL = "#d03b3b"
GRID_COLOR = "#e1e0d9"
HATCH_SELF = ""
HATCH_OTHER = "///"

# 契约 §7 登记的模型名（docs/01_数据契约.md）。颜色按这个固定顺序分配，
# 不随"这一轮实际有哪些模型跑完"变化——同一个模型在 F6/F7 两张图里颜色一致。
MODEL_ORDER = ["baserate", "logreg", "lgbm_single", "lgbm_dual",
               "gru", "transformer_v0", "transformer_cr"]
_CAT_HEX = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]
_CAT_MARK = ["o", "s", "^", "D", "v", "P", "X"]
MODEL_COLORS = dict(zip(MODEL_ORDER, _CAT_HEX))
MODEL_MARKERS = dict(zip(MODEL_ORDER, _CAT_MARK))


# =============================================================================
# 小工具
# =============================================================================

def _num(x, default=0.0):
    """从 metrics JSON 里取数：遇到隐私抑制过的 "<10" 之类字符串就退回 default，不崩。"""
    if isinstance(x, bool):
        return default
    if isinstance(x, (int, float)) and np.isfinite(x):
        return x
    return default


def _read_metrics(name: str):
    """读 `results/metrics/<name>.json`；文件不存在返回 None（由调用方决定怎么跳过）。"""
    p = C.METRICS / f"{name}.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:                          # noqa: BLE001
        print(f"  [警告] {p.name} 解析失败（{type(e).__name__}: {e}），当作不存在处理")
        return None


def _savefig(fig, stem: str, note: str, manifest: list):
    path = C.FIGURES / f"{stem}.png"
    fig.savefig(path, dpi=DPI, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    rel = str(path.relative_to(C.ROOT)).replace("\\", "/")
    manifest.append({"file": rel, "note": note, "status": "ok"})
    print(f"  已画 {rel}：{note}")


def _skip(stem: str, reason: str, manifest: list):
    manifest.append({"file": f"results/figures/{stem}.png", "note": reason, "status": "skipped"})
    print(f"跳过 {stem}：{reason}")


def font_smoke_test() -> Path:
    """画一张带中文的小图存到系统临时目录，人工确认没有方块字/缺字。

    存到临时目录而不是 results/figures/：那个目录只放 F1..F9 交付物，QA 图不掺进去。
    """
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    ax.set_title("中文字体自测：撤销率 22.1%，负号-5.0")
    ax.text(0.5, 0.5, "自撤 / 他撤\n医嘱撤销时延分布", ha="center", va="center", fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    p = Path(tempfile.gettempdir()) / "s08_font_smoke_test.png"
    fig.savefig(p, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[字体自测] 已存 {p}（不进 results/figures/），请人工确认无方块字后再看正式图")
    return p


# =============================================================================
# 第一组：数据事实（s02_labels.json / labels.parquet，现在就能画）
# =============================================================================

def fig_f1(manifest: list) -> None:
    """F1 撤销率 by order_type：横向条形图，按医嘱量排序，前 10 类 + 其它，标 n 和撤销率。"""
    stem = "F1_revocation_rate_by_order_type"
    d = _read_metrics("s02_labels")
    if d is None or "撤销率_by_order_type" not in d:
        _skip(stem, "缺 results/metrics/s02_labels.json 或其中的 撤销率_by_order_type 字段", manifest)
        return

    rows = [{"name": k, "n": _num(v.get("n")), "rate": _num(v.get("撤销率pct"))}
            for k, v in d["撤销率_by_order_type"].items()]
    rows = [r for r in rows if r["n"] > 0]
    if not rows:
        _skip(stem, "撤销率_by_order_type 里没有可用的数值（都被小格子抑制了？）", manifest)
        return
    rows.sort(key=lambda r: -r["n"])
    top = rows[:TOP_N_ORDER_TYPE]
    rest = rows[TOP_N_ORDER_TYPE:]
    if rest:
        n_rest = sum(r["n"] for r in rest)
        revoked_rest = sum(r["n"] * r["rate"] / 100.0 for r in rest)
        rate_rest = 100.0 * revoked_rest / n_rest if n_rest else float("nan")
        top.append({"name": f"其它（{len(rest)}类合并）", "n": n_rest, "rate": rate_rest})
    top.sort(key=lambda r: -r["n"])            # "其它"合并后可能比某些 top10 单类还大，重排一次

    names = [r["name"] for r in top][::-1]     # barh 从下往上画，反转让最大的在最上面
    ns = [r["n"] for r in top][::-1]
    rates = [r["rate"] for r in top][::-1]
    n_total = sum(ns)

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.set_axisbelow(True)
    ax.grid(axis="x", color=GRID_COLOR, linewidth=0.8)
    ax.barh(names, rates, color=SEQ_BLUE, edgecolor="white", linewidth=0.4)
    xmax = max(rates) if rates else 1.0
    for y, (n, rate) in enumerate(zip(ns, rates)):
        ax.text(rate + xmax * 0.015, y, f"{rate:.1f}%  (n={int(n):,})", va="center", fontsize=8)
    ax.set_xlim(0, xmax * 1.30)
    ax.set_xlabel("撤销率 (%)")
    ax.set_title(f"各医嘱类型的撤销率（按医嘱量排序，前{TOP_N_ORDER_TYPE}类+其它，覆盖 n={int(n_total):,}）")
    _savefig(fig, stem, f"前{TOP_N_ORDER_TYPE}类+其它的撤销率，n 已标注；覆盖 n={int(n_total):,}", manifest)


def fig_f2(manifest: list) -> None:
    """F2 撤销时延分布：对数横轴直方图 + p50 参考线。自己从 labels.parquet 算，只出聚合直方图。"""
    stem = "F2_delay_distribution"
    if not LABELS_PATH.exists():
        _skip(stem, "缺 data/labels.parquet", manifest)
        return

    delay = C.col_np(LABELS_PATH, "delay_h", fill=np.nan).astype(np.float64)
    in_cohort = C.col_np(LABELS_PATH, "in_cohort", fill=False)
    mask = np.isfinite(delay) & in_cohort & (delay > 0)
    x = delay[mask]
    del delay, in_cohort, mask
    gc.collect()

    n = int(x.size)
    if n == 0:
        _skip(stem, "队列内没有 delay_h>0 的撤销记录", manifest)
        return

    p50 = float(np.median(x))
    vmin = float(x.min())
    vmax_full = float(x.max())
    vmax = float(np.quantile(x, 0.999))
    vmax = max(vmax, vmin * 1.001)
    n_trimmed = int(np.count_nonzero(x > vmax))

    n_bins = 60
    edges = np.logspace(np.log10(vmin), np.log10(vmax), n_bins + 1)
    counts, edges = np.histogram(x, bins=edges)
    n_suppressed_bins = int(np.sum((counts > 0) & (counts < 10)))
    n_suppressed_orders = int(counts[(counts > 0) & (counts < 10)].sum())
    counts_disp = np.where(counts < 10, 0, counts).astype(np.float64)   # 小格子抑制：n<10 不画
    del x
    gc.collect()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8)
    ax.bar(edges[:-1], counts_disp, width=np.diff(edges), align="edge",
           color=SEQ_BLUE, edgecolor="white", linewidth=0.2)
    ax.set_xscale("log")
    ax.axvline(p50, color=COLOR_OTHER, linestyle="--", linewidth=1.6,
               label=f"中位数 p50={p50:.2f}h")
    ax.set_xlabel("撤销时延（下单到撤销，小时，对数轴）")
    ax.set_ylabel("医嘱数")
    note_bits = []
    if n_trimmed:
        note_bits.append(f"右侧 {n_trimmed:,} 条（{C.pct(n_trimmed, n)}%，最长到 {vmax_full:.0f}h）"
                          f"超出 p99.9 未画出")
    if n_suppressed_bins:
        note_bits.append(f"另有 {n_suppressed_bins} 个长尾桶因 n<10 被隐去")
    note = ("；" + "；".join(note_bits)) if note_bits else ""
    ax.set_title(f"撤销时延分布，n={n:,}{note}")
    ax.legend(loc="upper right", fontsize=9)
    _savefig(fig, stem, f"撤销时延对数直方图，n={n}，p50={p50:.2f}h", manifest)


def fig_f3(manifest: list) -> None:
    """F3（核心图）自撤 vs 他撤 的时延桶分布，各占自己总量的百分比。"""
    stem = "F3_hazard_shape_self_vs_other"
    d = _read_metrics("s02_labels")
    if d is None or "t_bin_counts_self" not in d or "t_bin_counts_other" not in d:
        _skip(stem, "缺 s02_labels.json 或其中的 t_bin_counts_self/other 字段", manifest)
        return

    cs, co = d["t_bin_counts_self"], d["t_bin_counts_other"]
    self_counts = [_num(cs.get(str(i))) for i in range(6)]
    other_counts = [_num(co.get(str(i))) for i in range(6)]
    n_self, n_other = sum(self_counts), sum(other_counts)
    if n_self <= 0 or n_other <= 0:
        _skip(stem, "t_bin_counts_self/other 求和为 0（都被抑制或缺失）", manifest)
        return
    self_pct = [100.0 * c / n_self for c in self_counts]
    other_pct = [100.0 * c / n_other for c in other_counts]

    x = np.arange(6)
    w = 0.36
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8)
    b1 = ax.bar(x - w / 2, self_pct, width=w, label=f"自撤（n={int(n_self):,}）",
                color=COLOR_SELF, hatch=HATCH_SELF, edgecolor="black", linewidth=0.8)
    b2 = ax.bar(x + w / 2, other_pct, width=w, label=f"他撤（n={int(n_other):,}）",
                color=COLOR_OTHER, hatch=HATCH_OTHER, edgecolor="black", linewidth=0.8)
    for rect, v in zip(list(b1) + list(b2), self_pct + other_pct):
        ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.6,
                f"{v:.1f}%", ha="center", va="bottom", fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels(BUCKET_LABELS)
    ax.set_xlabel("下单到撤销的时延分桶")
    ax.set_ylabel("占该原因总撤销数的比例 (%)")
    ax.set_ylim(0, max(self_pct + other_pct) * 1.22)
    ax.set_title("自撤集中在下单后 1 小时内、他撤峰值在 6-24 小时——两个过程的 hazard 形状不同")
    ax.legend(fontsize=9)
    _savefig(fig, stem,
             f"自撤/他撤各自的时延桶分布（占各自总量%），n_self={int(n_self)}, n_other={int(n_other)}",
             manifest)


def fig_f4(manifest: list) -> None:
    """F4 自撤占比 by order_type：全部 16 类按自撤占比降序排列，标 n。"""
    stem = "F4_self_share_by_order_type"
    d = _read_metrics("s02_labels")
    if d is None or "撤销率_by_order_type" not in d:
        _skip(stem, "缺 s02_labels.json 或 撤销率_by_order_type 字段", manifest)
        return

    rows = [{"name": k, "n": _num(v.get("n")), "self_pct": _num(v.get("自撤占已知pct"), None)}
            for k, v in d["撤销率_by_order_type"].items()]
    rows = [r for r in rows if r["n"] > 0 and r["self_pct"] is not None]
    if not rows:
        _skip(stem, "撤销率_by_order_type 里没有可用的 自撤占已知pct", manifest)
        return
    rows.sort(key=lambda r: -r["self_pct"])

    names = [r["name"] for r in rows][::-1]
    vals = [r["self_pct"] for r in rows][::-1]
    ns = [r["n"] for r in rows][::-1]

    fig, ax = plt.subplots(figsize=(8.5, 6.8))
    ax.set_axisbelow(True)
    ax.grid(axis="x", color=GRID_COLOR, linewidth=0.8)
    ax.barh(names, vals, color=COLOR_SELF, edgecolor="black", linewidth=0.5)
    ax.axvline(50, color="#52514e", linestyle="--", linewidth=1, label="50% 参考线")
    for y, (n, v) in enumerate(zip(ns, vals)):
        ax.text(v + 1.0, y, f"{v:.1f}% (n={int(n):,})", va="center", fontsize=7.6)
    ax.set_xlim(0, 105)
    ax.set_xlabel("自撤占「已知撤销原因」医嘱的比例 (%)")
    ax.set_title("自撤占比因医嘱类型而异（Medications 最低、TPN/Hemodialysis 最高）——不是同一个过程")
    ax.legend(loc="lower right", fontsize=8)
    _savefig(fig, stem, f"{len(rows)} 个 order_type 的自撤占比（降序），n 已标注", manifest)


def fig_f5(manifest: list) -> None:
    """F5 队列流图：从全部医嘱到入队列的筛选漏斗（cohort_flow 原始顺序，不重排）。"""
    stem = "F5_cohort_flow"
    d = _read_metrics("s02_labels")
    if d is None or "cohort_flow" not in d:
        _skip(stem, "缺 s02_labels.json 或 cohort_flow 字段", manifest)
        return

    flow = [(k, _num(v)) for k, v in d["cohort_flow"].items()]
    flow = [(k, v) for k, v in flow if v > 0]
    if not flow:
        _skip(stem, "cohort_flow 里没有可用的数值", manifest)
        return
    names = [f[0] for f in flow]
    vals = [f[1] for f in flow]
    n0 = vals[0]

    ic = _num(d.get("in_cohort"), None)
    if ic is not None and abs(ic - vals[-1]) > max(1.0, 0.0001 * n0):
        print(f"  [F5 提示] cohort_flow 最后一步 {vals[-1]:,.0f} 与 in_cohort 字段 {ic:,.0f} "
              f"不一致，仍按 cohort_flow 原始记录画图")

    colors = [ORDINAL_BLUE_RAMP[min(i, len(ORDINAL_BLUE_RAMP) - 1)] for i in range(len(flow))]
    y = np.arange(len(flow))[::-1]     # 第一阶段画在最上面

    fig, ax = plt.subplots(figsize=(8.5, max(3.2, 0.7 * len(flow) + 1.2)))
    ax.set_axisbelow(True)
    ax.grid(axis="x", color=GRID_COLOR, linewidth=0.8)
    ax.barh(y, vals, color=colors, edgecolor="black", height=0.55)
    for yi, n in zip(y, vals):
        pct = 100.0 * n / n0 if n0 else float("nan")
        ax.text(n + n0 * 0.01, yi, f"{n:,.0f}（{pct:.2f}%）", va="center", fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlim(0, n0 * 1.32)
    ax.set_xlabel("医嘱数")
    ax.set_title("队列筛选流程：从全部医嘱到入队列（括号内为占第一阶段的百分比）")
    _savefig(fig, stem, f"队列流：{names[0]} n={n0:,.0f} → 最终 n={vals[-1]:,.0f}", manifest)


# =============================================================================
# 第二组：模型结果（依赖 s05/s06/s07，现在很可能还没有）
# =============================================================================

def _load_pred_key_score(model: str, split: str):
    """只读 `results/preds/<model>_<split>.parquet` 的 key/p_disc 两列。"""
    import pyarrow.parquet as pq
    p = PREDS_DIR / f"{model}_{split}.parquet"
    t = pq.read_table(p, columns=["key", "p_disc"])
    pk = t["key"].to_numpy(zero_copy_only=False).astype(np.int64)
    ps = t["p_disc"].to_numpy(zero_copy_only=False).astype(np.float64)
    return pk, ps


def fig_f6(manifest: list) -> None:
    """F6 PR 曲线对比：各模型在 test 上的 precision-recall 曲线（H1，AUPRC 是主指标）。

    直接读 `results/preds/`（契约 §7）而不是 `s07_eval_*.json`——见文件头的说明。
    """
    stem = "F6_pr_curve"
    present = [m for m in MODEL_ORDER if (PREDS_DIR / f"{m}_{SPLIT}.parquet").exists()]
    if not present:
        _skip(stem, f"results/preds/ 下没有任何模型的 {SPLIT} 预测文件（契约 §7），"
                     "上游模型脚本还没产出预测", manifest)
        return
    if not LABELS_PATH.exists():
        _skip(stem, "缺 data/labels.parquet，无法给预测对齐真值", manifest)
        return

    lab_key = C.col_np(LABELS_PATH, "key", fill=-1).astype(np.int64)
    lab_y = C.col_np(LABELS_PATH, "y_disc", fill=False).astype(np.int8)

    fig, ax = plt.subplots(figsize=(6.6, 6.6))
    ax.set_axisbelow(True)
    ax.grid(color=GRID_COLOR, linewidth=0.8)
    drawn = []
    for m in present:
        try:
            pk, ps = _load_pred_key_score(m, SPLIT)
            order = np.argsort(pk, kind="stable")
            pk_sorted, ps_sorted = pk[order], ps[order]
            j = np.searchsorted(pk_sorted, lab_key)
            jc = np.clip(j, 0, pk_sorted.size - 1)
            hit = pk_sorted[jc] == lab_key
            y = lab_y[hit]
            n = int(y.size)
            if n == 0 or y.sum() == 0:
                print(f"  [F6] {m}：匹配到 0 行或没有正类，跳过这条线")
                del pk, ps, order, pk_sorted, ps_sorted, j, jc, hit, y
                gc.collect()
                continue
            score = ps_sorted[jc[hit]]
            so = np.argsort(-score, kind="stable")
            y_ord = y[so]
            tp = np.cumsum(y_ord)
            fp = np.cumsum(1 - y_ord)
            precision = tp / (tp + fp)
            recall = tp / tp[-1]
            ap = float(np.sum(np.diff(recall, prepend=0.0) * precision))
            if recall.size > 2000:
                idx = np.unique(np.linspace(0, recall.size - 1, 2000).astype(np.int64))
                r_plot, p_plot = recall[idx], precision[idx]
            else:
                r_plot, p_plot = recall, precision
            ax.plot(r_plot, p_plot, color=MODEL_COLORS.get(m, SEQ_BLUE),
                     marker=MODEL_MARKERS.get(m, "o"), markevery=max(len(r_plot) // 12, 1),
                     markersize=5, linewidth=2.0,
                     label=f"{m}（AUPRC={ap:.3f}, n={n:,}）")
            drawn.append(m)
            del pk, ps, order, pk_sorted, ps_sorted, j, jc, hit, y, score, so, y_ord, tp, fp
        except Exception as e:                       # noqa: BLE001
            print(f"  [F6] {m} 画线失败（{type(e).__name__}: {e}），跳过这一条")
        gc.collect()
    del lab_key, lab_y
    gc.collect()

    if not drawn:
        plt.close(fig)
        _skip(stem, "候选模型的预测文件都在，但没有一条能画出曲线（见上方日志）", manifest)
        return
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Recall（召回率）")
    ax.set_ylabel("Precision（精确率）")
    ax.set_title(f"各模型 {SPLIT} 集 PR 曲线对比（H1：是否会被撤销）")
    ax.legend(loc="upper right", fontsize=8)
    _savefig(fig, stem, f"{len(drawn)} 个模型的 PR 曲线：{', '.join(drawn)}", manifest)


def fig_f7(manifest: list) -> None:
    """F7 校准曲线：10 桶可靠性图 + 对角线，读 s07_eval_<model>_<split>.json 的 reliability 字段。"""
    stem = "F7_calibration"
    curves, eces, nrows = {}, {}, {}
    for m in MODEL_ORDER:
        d = _read_metrics(f"s07_eval_{m}_{SPLIT}")
        if d and isinstance(d.get("reliability"), dict):
            curves[m] = d["reliability"]
            eces[m] = d.get("ece")
            nrows[m] = _num(d.get("n_rows"), None)
    if not curves:
        _skip(stem, f"没有任何 results/metrics/s07_eval_*_{SPLIT}.json 含 reliability 字段", manifest)
        return

    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    ax.set_axisbelow(True)
    ax.grid(color=GRID_COLOR, linewidth=0.8)
    ax.plot([0, 1], [0, 1], linestyle="--", color="#898781", linewidth=1.5, label="完美校准")
    drawn = []
    for m, rc in curves.items():
        mp = np.asarray(rc.get("mean_pred", []), dtype=np.float64)
        fpz = np.asarray(rc.get("frac_pos", []), dtype=np.float64)
        ok = np.isfinite(mp) & np.isfinite(fpz)
        if not ok.any():
            continue
        lbl = m
        if isinstance(eces.get(m), (int, float)):
            lbl += f"（ECE={eces[m]:.4f}"
            lbl += f", n={int(nrows[m]):,}）" if nrows.get(m) else "）"
        ax.plot(mp[ok], fpz[ok], marker=MODEL_MARKERS.get(m, "o"),
                 color=MODEL_COLORS.get(m, SEQ_BLUE), linewidth=2.0, markersize=6.5, label=lbl)
        drawn.append(m)
    if not drawn:
        plt.close(fig)
        _skip(stem, "reliability 字段都是空/全 NaN 桶", manifest)
        return
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("模型预测概率（桶内均值）")
    ax.set_ylabel("实际撤销发生率（桶内）")
    ax.set_title(f"校准曲线（{SPLIT} 集，等宽 10 桶；越贴近对角线越好）")
    ax.legend(loc="lower right", fontsize=8)
    _savefig(fig, stem, f"{len(drawn)} 个模型的校准曲线：{', '.join(drawn)}", manifest)


def fig_f8(manifest: list) -> None:
    """F8 预注册主实验：lgbm_dual vs lgbm_single 的 ΔAUPRC 点估计 + bootstrap 95% CI。"""
    stem = "F8_prereg_delta_auprc"
    name_fwd = f"s07_compare_lgbm_dual_vs_lgbm_single_{SPLIT}"
    name_rev = f"s07_compare_lgbm_single_vs_lgbm_dual_{SPLIT}"
    d = _read_metrics(name_fwd)
    flip = False
    if d is None:
        d = _read_metrics(name_rev)
        flip = d is not None
    if d is None:
        _skip(stem, f"缺 results/metrics/{name_fwd}.json（预注册主实验还没跑）", manifest)
        return

    delta, lo, hi = d.get("delta"), d.get("ci_lo"), d.get("ci_hi")
    if delta is None or lo is None or hi is None:
        _skip(stem, f"{name_fwd if not flip else name_rev}.json 缺 delta/ci_lo/ci_hi 字段", manifest)
        return
    if flip:                     # 若两模型顺序反了，统一换算成"dual − single"再画
        delta, lo, hi = -delta, -hi, -lo

    metric = str(d.get("metric", "auprc"))
    thr = d.get("prereg_threshold")
    if thr is None:
        thr = float(C.cfg().get("eval", {}).get("prereg_delta_auprc", 0.01))
    ci_excl0 = (lo > 0) or (hi < 0)
    met = bool(delta >= thr and ci_excl0 and delta > 0)     # 本地重算，不直接信 flip 后的旧字段

    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    color = STATUS_GOOD if met else STATUS_CRITICAL
    ax.errorbar([delta], [0], xerr=[[delta - lo], [hi - delta]], fmt="o", color=color,
                markersize=11, capsize=6, linewidth=2.2, elinewidth=2.2, zorder=3)
    ax.axvline(0, color="#898781", linewidth=1, linestyle="-", zorder=1)
    ax.axvline(thr, color="#52514e", linewidth=1.3, linestyle="--", zorder=1)
    ax.text(thr, 0.42, f"预注册判据 Δ≥{thr:g}", ha="center", fontsize=8.5, color="#52514e")
    ax.text(delta, -0.55, f"Δ{metric.upper()}={delta:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]\n"
                          f"{'达成' if met else '未达成'}预注册判据",
            ha="center", fontsize=9.5, color=color, fontweight="bold")
    ax.set_yticks([])
    ax.set_ylim(-1, 1)
    span = max(abs(hi), abs(lo), abs(thr), 1e-6) * 1.5
    ax.set_xlim(-span, span)
    ax.set_xlabel(f"Δ{metric.upper()} = {metric.upper()}(lgbm_dual) − {metric.upper()}(lgbm_single)")
    n_boot, n_subj = d.get("n_boot"), d.get("n_subjects")
    ax.set_title(f"预注册主实验：双头 vs 单头（按患者 bootstrap {n_boot} 次，n_subjects={n_subj}）")
    _savefig(fig, stem,
             f"Δ{metric.upper()}={delta:+.4f} CI[{lo:+.4f},{hi:+.4f}] "
             f"预注册判据{'达成' if met else '未达成'}", manifest)


def fig_f9(manifest: list) -> None:
    """F9 特征重要性：LightGBM top-20（读 s05_baseline.json 的 models.<name>.top30_gain）。"""
    stem = "F9_feature_importance"
    d = _read_metrics("s05_baseline")
    if d is None:
        _skip(stem, "缺 results/metrics/s05_baseline.json", manifest)
        return
    models = d.get("models", {})
    priority = ["lgbm_single", "lgbm_dual", "lgbm_self_head", "lgbm_other_head", "lgbm_13"]
    chosen = None
    for name in priority:
        m = models.get(name)
        if isinstance(m, dict) and isinstance(m.get("top30_gain"), list) and m["top30_gain"]:
            chosen = (name, m["top30_gain"])
            break
    if chosen is None:
        _skip(stem, "s05_baseline.json 的 models.*.top30_gain 都不存在/为空", manifest)
        return

    model_name, rows = chosen
    top = [r for r in rows if isinstance(r, dict) and "feature" in r][:20]
    if not top:
        _skip(stem, f"{model_name}.top30_gain 里没有带 feature 字段的条目", manifest)
        return
    use_pct = "gain_pct" in top[0]
    feats = [r["feature"] for r in top][::-1]
    vals = [r.get("gain_pct", r.get("gain", 0.0)) if use_pct else r.get("gain", 0.0)
            for r in top][::-1]
    unit = "gain 占比 (%)" if use_pct else "gain（原始值）"

    cohort = d.get("cohort", {}) if isinstance(d.get("cohort"), dict) else {}
    n_tr = _num(cohort.get("n_train_rows_used"), None)
    n_subj = _num(cohort.get("n_train_subjects"), None)
    size_note = ""
    if n_tr:
        size_note = f"，训练 n_rows={int(n_tr):,}" + (f"、n_subjects={int(n_subj):,}" if n_subj else "")

    fig, ax = plt.subplots(figsize=(8.5, max(5.0, 0.32 * len(top) + 1.5)))
    ax.set_axisbelow(True)
    ax.grid(axis="x", color=GRID_COLOR, linewidth=0.8)
    ax.barh(feats, vals, color=SEQ_BLUE, edgecolor="white", linewidth=0.4)
    xmax = max(vals) if vals else 1.0
    for y, v in enumerate(vals):
        suffix = "%" if use_pct else ""
        ax.text(v + xmax * 0.015, y, f"{v:.2f}{suffix}", va="center", fontsize=7.8)
    ax.set_xlim(0, xmax * 1.18)
    ax.set_xlabel(unit)
    ax.set_title(f"特征重要性 top {len(top)}（{model_name}，LightGBM gain{size_note}）")
    _savefig(fig, stem, f"{model_name} 的 top{len(top)} 特征重要性（gain）{size_note}", manifest)


# =============================================================================
# main
# =============================================================================

FIG_FUNCS = {
    "F1": fig_f1, "F2": fig_f2, "F3": fig_f3, "F4": fig_f4, "F5": fig_f5,
    "F6": fig_f6, "F7": fig_f7, "F8": fig_f8, "F9": fig_f9,
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="BN5212 出图：第一组数据事实 + 第二组模型结果（缺就跳过）")
    ap.add_argument("--only", default=None, help="只画指定的图，逗号分隔，如 F1,F3")
    a = ap.parse_args(argv)

    t0, _ = C.start_log("s08_figures")
    try:
        font_smoke_test()
    except Exception as e:                           # noqa: BLE001
        print(f"[警告] 字体自测图失败：{type(e).__name__}: {e}（继续画正式图，请事后自查中文有没有方块）")

    todo = list(FIG_FUNCS)
    if a.only:
        todo = [x.strip().upper() for x in a.only.split(",") if x.strip()]

    manifest = []
    for name in todo:
        fn = FIG_FUNCS.get(name)
        if fn is None:
            print(f"[警告] 未知的图号 {name!r}，跳过（可选：{', '.join(FIG_FUNCS)}）")
            continue
        print(f"\n--- {name} ---")
        try:
            fn(manifest)
        except Exception as e:                       # noqa: BLE001
            print(f"[错误] {name} 画图异常：{type(e).__name__}: {e}，跳过，不影响其它图")
            manifest.append({"file": f"results/figures/{name}_*.png",
                              "note": f"异常：{type(e).__name__}: {e}", "status": "error"})

    n_ok = sum(1 for r in manifest if r["status"] == "ok")
    n_bad = len(manifest) - n_ok
    C.save_metrics("s08_figures", {"figures": manifest, "n_ok": n_ok, "n_skipped_or_error": n_bad}, t0)
    C.append_runlog("s08_figures", f"画出 {n_ok} 张，跳过/异常 {n_bad} 张")
    print(f"\n完成：{n_ok} 张已画，{n_bad} 张跳过/异常")
    return 0


if __name__ == "__main__":
    sys.exit(main())
