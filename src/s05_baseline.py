# -*- coding: utf-8 -*-
"""s05_baseline.py —— 基线模型 + 预注册主实验（单头 vs 双头）。

本脚本只负责**吐预测**，不负责算最终指标（指标由 src/s07_eval.py 统一算，
保证所有模型用同一把尺子）。脚本内部会打印一个 sklearn 的粗略 AUPRC 做自查，
**打印和 JSON 里都标了「仅自查」**，正式结果以 s07_eval 为准。

模型清单（全部按 docs/01_数据契约.md §7 吐 preds）
-------------------------------------------------
baserate      常数预测 = 训练集撤销率（AUPRC 的下限基准）
logreg        逻辑回归（类别 one-hot + 数值标准化，SGD log_loss）
lgbm_single   **单头**：直接训二分类 y_disc
lgbm_dual     **双头**：头1 训 cause==1（自撤）、头2 训 cause==2（他撤），合成 p_disc
lgbm_13       （可选，--with13）13 类 multiclass，直接给出完整 p13_* 向量

预注册主张（docs/00_项目方案.md §3.1，AGENTS.md §1）
--------------------------------------------------
撤销是「自撤」「他撤」两个方向相反的过程的混合，分头建模再合成优于单头。
判据：test 上 ΔAUPRC = AUPRC(lgbm_dual) − AUPRC(lgbm_single) >= 0.01，
且按 subject_id 重抽的 bootstrap 95% CI 不含 0（CI 由 s07_eval 算）。
**不达标就如实报告，负结果照样是结果。**

双头如何合成 p_disc（关键，别写错）
----------------------------------
自撤与他撤是**互斥**的竞争事件（一条医嘱只会被撤一次，撤它的人要么是本人要么是别人），
两个头都在**全体样本**上训、各自的正类是互斥子集，所以
    p_disc = p_self + p_other                      ← **主结果**
而不是 1-(1-p_self)(1-p_other)（那个假设两事件独立共存，在这里是错的）。
两种都算、都报，JSON 里 `composition.primary` 写明哪个是主结果。

⚠ cause==3（provider 缺失、不知谁撤，约占撤销的 1.07%）：
它确实被撤了，所以在 lgbm_single 的标签里是**正类**，但在双头里两个头都算**负类**
—— 这会系统性压低 p_disc。这是双头的一个**真实劣势**，已写进 JSON 的
`cause3_penalty` 与报告。**不许**把这些样本从 single 里也删掉来「公平」——那是在改题目。
正确做法：主结果用全部样本（诚实地让双头吃这个亏），另出一份「排除 cause==3 后」的
敏感性分析（preds 后缀 `_nc3`，**模型不重训，只在评测行集合上排除**），两个都报。

隐私（AGENTS.md §2）
-------------------
只输出聚合量；n<10 一律 C.sup()；**绝不打印 order_provider_id / field_value 的取值**。
特征重要性只报**列名**与重要性数值，不报具体取值。

用法
----
    python src/s05_baseline.py --synth 60000      # 合成数据跑通整条路径（不碰 data/ 与 results/）
    python src/s05_baseline.py --smoke            # 真数据、抽 1% subject，产物落 *_smoke
    python src/s05_baseline.py                    # 全量
    python src/s05_baseline.py --with13           # 额外训 13 类 multiclass（慢）
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

import common as C

# --------------------------------------------------------------------------- 常量

# 泄漏铁律（契约 §3.1）：这些子串一出现就**立刻停机**，不自己删了继续跑。
LEAK_HARD = ("order_status", "discontinued", "discontinue_of", "_by_seq", "y_disc", "y13", "cause",
             "discharge_location", "hospital_expire", "drg")
# 契约 §3 明文允许的「名字里带 disc 但合法」的滚动前缀统计（t₀ 之前的累积量）。
# f_prior_disc_known 是 s03 加的、契约 §3 没逐字列出的一列。
# ⚠ 它**曾经真的泄漏过**（2026-09-21 20:00 由 src/t01_leak_test.py 抓到）：
#   前缀统计用「撤销时刻 <= t₀」而 f_n_prior_orders 用「严格早于 t₀」，口径不一致，
#   导致 191 条「撤销与下单同秒」的医嘱把**自己已被撤销**这件事数进了自己的历史特征
#   —— 专挑正类下手的自指泄漏，并造出 38 行「已撤数 > 此前下达数」的逻辑不可能。
#   s03 已统一成「严格早于 t₀」。**本脚本不信任口头修复**：见 check_feature_invariants()，
#   每次训练前用训练矩阵机器复核这三条不变式，违反就停机。
# f_detail_discharge_planning 里的 disc 是 **discharge（出院）不是 discontinue（撤销）**：
#   它是 poe_detail 的一个 field_name（「出院计划」这张单子上有没有填这一栏），t₀ 时刻就定了。
#   实测只有两个取值（-1=该栏缺失 / 0=有该栏），撤销率 20.37% vs 24.41%，单列 AUC=0.5014 ⇒ 无泄漏。
#   注意这**不是**契约 §3.1 禁止的 discharge_location（那个是出院后才知道的去向，已进 LEAK_HARD）。
LEAK_ALLOW = ("f_prior_disc_rate", "f_prov_prior_disc_rate", "f_prior_disc_known",
              "f_detail_discharge_planning")

# 数值特征（其余整数编码列一律当 categorical 交给 LightGBM）。
NUMERIC_EXACT = {
    "f_since_admit_h", "f_n_prior_orders", "f_prior_disc_rate", "f_dt_prev_order_h",
    "f_prov_n_prior", "f_prov_prior_disc_rate", "f_prov_prior_self_rate",
    "f_prior_disc_known",
}
NUMERIC_PREFIX = ("f_n_",)                     # 计数类：f_n_prior_orders / f_n_detail
NUMERIC_SUFFIX = ("_h", "_rate", "_frac", "_ratio", "_std", "_mean")

# 方向分析的两个子群（名字来自 order_type 词表，聚合级、非患者级）
SUBGROUPS = ("Medications", "IV therapy")

SALT_SAMPLE = 0xB15212  # 抽样用的独立盐；与契约 §4 的切分哈希无关，不会改变 split


# --------------------------------------------------------------------------- 小工具

def splitmix64(x: np.ndarray, salt: int) -> np.ndarray:
    """向量化 splitmix64：subject_id -> 均匀的 uint64。确定性、可复现、与 split 正交。"""
    with np.errstate(over="ignore"):
        z = (x.astype(np.uint64) ^ np.uint64(salt)) + np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z = z ^ (z >> np.uint64(31))
    return z


def read_col(path: Path, name: str):
    """读 parquet 的一列成 numpy。浮点列 null->NaN；整数/布尔列 null->-1/False。

    不用 C.col_np 是因为它对浮点列的 fill_null(-1) 会把缺失变成真实取值 -1。
    """
    a = pq.read_table(path, columns=[name])[name]
    if pa.types.is_floating(a.type):
        a = a.combine_chunks()
        if hasattr(a, "num_chunks"):
            a = a.chunk(0) if a.num_chunks else a
        return a.to_numpy(zero_copy_only=False)
    fill = False if pa.types.is_boolean(a.type) else -1
    a = pc.fill_null(a, fill).combine_chunks()
    if hasattr(a, "num_chunks"):
        a = a.chunk(0) if a.num_chunks else a
    return a.to_numpy(zero_copy_only=False)


def classify_cols(schema: pa.Schema) -> tuple[list[str], list[str], list[str], set[str]]:
    """features.parquet 的列 -> (全部 f_ 列, 数值列, 类别列, 整数型数值列)。

    整数型数值列要单独拎出来：它们的缺失在 read_col 里被填成 -1 哨兵，
    后面要还原成 NaN；浮点列本来就带 NaN，**不能**做这个替换
    （f_since_admit_h 实测最小值 -81381 h，-1 是合法取值）。
    """
    feats = [n for n in schema.names if n.startswith("f_")]
    num, cat, num_int = [], [], set()
    for n in feats:
        t = schema.field(n).type
        is_num = (n in NUMERIC_EXACT) or n.endswith(NUMERIC_SUFFIX) \
            or n.startswith(NUMERIC_PREFIX) or pa.types.is_floating(t)
        if is_num:
            num.append(n)
            if not pa.types.is_floating(t):
                num_int.add(n)
        else:
            cat.append(n)
    return feats, num, cat, num_int


def leak_check(all_cols: list[str]) -> dict:
    """契约 §3.1 的泄漏自查。命中硬名单 -> 抛异常停机（不自己删列继续跑）。"""
    hard, soft = [], []
    for n in all_cols:
        low = n.lower()
        if n in LEAK_ALLOW:
            soft.append(n)
            continue
        if any(k in low for k in LEAK_HARD):
            hard.append(n)
        elif "disc" in low:          # 不在白名单里、名字又带 disc 的，一律当泄漏处理
            hard.append(n)
    if hard:
        raise RuntimeError(
            f"🔴 泄漏自查失败：features 里出现疑似泄漏列 {hard}。"
            f"按派单要求**立刻停机并报告**，不自行删列继续跑。"
            f"（契约 §3.1 白名单只有 {list(LEAK_ALLOW)}）")
    return {"hard_hits": hard, "allowlisted_disc_cols": soft}


def check_feature_invariants(X: np.ndarray, names: list[str]) -> dict:
    """训练前的机器复核：前缀统计的三条不变式。违反就**停机**，不训。

    这道门是 2026-09-21 那次自指泄漏的产物：s03 曾把「撤销时刻 <= t₀」当成历史，
    于是「与下单同秒被撤」的医嘱把自己的结局数进了自己的特征。
    口头说修好了不算数，每次训练前用真实训练矩阵量一遍：
      ① f_prior_disc_known <= f_n_prior_orders     （已撤数不可能多于此前下达数）
      ② f_prior_disc_rate ∈ [0, 1]                  （是个比例）
      ③ f_prov_*_rate     ∈ [0, 1]
    """
    rep, bad = {}, []

    def g(c):
        return X[:, names.index(c)] if c in names else None

    a, b = g("f_prior_disc_known"), g("f_n_prior_orders")
    if a is not None and b is not None:
        v = int(np.nansum(a > b))
        rep["f_prior_disc_known>f_n_prior_orders"] = v
        if v:
            bad.append(f"{v} 行『已撤数 > 此前下达数』（自指泄漏的指纹）")
        z = int(np.nansum((b == 0) & (a > 0)))
        rep["第一条医嘱就已有撤销历史"] = z
        if z:
            bad.append(f"{z} 行在本次住院第一条医嘱上就带非零撤销历史")
    for c in ("f_prior_disc_rate", "f_prov_prior_disc_rate", "f_prov_prior_self_rate"):
        v = g(c)
        if v is None:
            continue
        n = int(np.nansum((v < 0) | (v > 1)))
        rep[f"{c}_越界[0,1]"] = n
        if n:
            bad.append(f"{c} 有 {n} 行不在 [0,1]（{np.nanmin(v):.3f}~{np.nanmax(v):.3f}）")
    if bad:
        raise RuntimeError("🔴 特征不变式校验失败，拒绝训练：" + "；".join(bad) +
                           "。这是 t01_leak_test 抓过的那类前缀统计口径问题，请先修 s03。")
    print(f"✓ 特征不变式校验通过（训练矩阵 {X.shape[0]:,} 行）：{rep}")
    return rep


class TimeBudget:
    """LightGBM 回调：超过墙钟预算就干净地停下（防跑飞，不是早停）。"""

    def __init__(self, seconds: float, name: str):
        self.t0, self.s, self.name, self.hit = time.time(), seconds, name, False

    def __call__(self, env):
        if time.time() - self.t0 > self.s:
            self.hit = True
            print(f"  [{self.name}] 触达时间预算 {self.s:.0f}s，在第 {env.iteration + 1} 轮停止")
            import lightgbm as lgb
            raise lgb.callback.EarlyStopException(env.iteration, env.evaluation_result_list)


def ap_selfcheck(y: np.ndarray, p: np.ndarray) -> float | None:
    """粗略 AUPRC，**仅自查**，正式指标见 s07_eval。"""
    from sklearn.metrics import average_precision_score
    y = np.asarray(y).astype(np.int8)
    if y.sum() == 0 or y.sum() == len(y):
        return None
    return round(float(average_precision_score(y, np.asarray(p, dtype=np.float64))), 6)


# --------------------------------------------------------------------------- 合成数据（dry run）

def make_synth(out: Path, n_subj: int, seed: int) -> None:
    """造结构与契约严格一致的假数据，把整条路径先跑通。**只写到 --data-dir，不碰 data/**。

    故意埋两个东西：① 一个真实可学的信号（让 AUPRC 明显高于 baserate）；
    ② 同一个特征在 Medications / IV therapy 上对「自撤 vs 他撤」**方向相反**，
    用来验证子群方向分析这段代码真的能把符号翻转找出来。
    """
    rng = np.random.default_rng(seed)
    out.mkdir(parents=True, exist_ok=True)
    n_ord = rng.integers(1, 40, size=n_subj)
    subj = np.repeat(np.arange(1000, 1000 + n_subj, dtype=np.int64), n_ord)
    n = subj.size
    seq = np.concatenate([np.arange(k, dtype=np.int64) + 1 for k in n_ord])
    key = subj * (2 ** 24) + seq

    ot_names = ["Medications", "Lab", "General Care", "ADT orders", "IV therapy",
                "Nutrition", "Radiology", "Consults", "Blood Bank", "Respiratory",
                "Cardiology", "TPN", "Critical Care", "Hemodialysis", "Neurology", "OB"]
    f_ot = rng.integers(0, len(ot_names), n).astype(np.int16)
    cols = {
        "f_order_type": f_ot,
        "f_order_subtype": rng.integers(0, 200, n).astype(np.int16),
        "f_transaction_type": rng.integers(0, 4, n).astype(np.int16),
        "f_hour": rng.integers(0, 24, n).astype(np.int16),
        "f_dow": rng.integers(0, 7, n).astype(np.int16),
        "f_admission_type": rng.integers(0, 9, n).astype(np.int16),
        "f_admission_location": rng.integers(0, 12, n).astype(np.int16),
        "f_insurance": rng.integers(0, 5, n).astype(np.int16),
        "f_careunit": rng.integers(0, 40, n).astype(np.int16),
        "f_curr_service": rng.integers(0, 20, n).astype(np.int16),
        "f_gender": rng.integers(0, 2, n).astype(np.int16),
        "f_age_bin": rng.integers(0, 6, n).astype(np.int16),
        "f_anchor_year_group": rng.integers(0, 5, n).astype(np.int16),
        "f_detail_dose": rng.integers(-1, 50, n).astype(np.int16),
        "f_detail_route": rng.integers(-1, 20, n).astype(np.int16),
        "f_detail_frequency": rng.integers(-1, 30, n).astype(np.int16),
        "f_since_admit_h": rng.exponential(48, n).astype(np.float32),
        "f_n_prior_orders": (npo := rng.integers(0, 500, n).astype(np.int32)),
        # 前缀统计的不变式要在合成数据上也成立，否则 check_feature_invariants 测不到
        "f_prior_disc_known": (npo * rng.random(n)).astype(np.int32),
        "f_prior_disc_rate": rng.beta(2, 8, n).astype(np.float32),
        "f_dt_prev_order_h": rng.exponential(3, n).astype(np.float32),
        "f_prov_n_prior": rng.integers(0, 6, n).astype(np.int16),
        "f_prov_prior_disc_rate": rng.beta(2, 8, n).astype(np.float32),
        "f_prov_prior_self_rate": rng.beta(4, 6, n).astype(np.float32),
    }
    z = (-1.6 + 1.5 * cols["f_prior_disc_rate"] + 0.004 * np.minimum(cols["f_n_prior_orders"], 300)
         + 0.010 * np.minimum(cols["f_since_admit_h"], 200) - 0.05 * cols["f_transaction_type"])
    p_disc = 1 / (1 + np.exp(-z))
    y_disc = rng.random(n) < p_disc

    # 方向相反：f_prov_prior_self_rate 在 Medications 上压低自撤占比、在 IV therapy 上抬高
    sgn = np.where(f_ot == 0, -1.0, np.where(f_ot == 4, +1.0, 0.0))
    zs = 0.2 + 3.0 * sgn * (cols["f_prov_prior_self_rate"] - 0.4) + 0.5 * (cols["f_hour"] > 12)
    is_self = rng.random(n) < 1 / (1 + np.exp(-zs))
    cause = np.where(~y_disc, 0, np.where(rng.random(n) < 0.011, 3,
                                          np.where(is_self, 1, 2))).astype(np.int8)
    delay = np.where(y_disc, rng.exponential(24, n), np.nan).astype(np.float32)
    edges = np.array(C.cfg()["labels"]["bin_edges_h"] + [np.inf], dtype=np.float64)
    t_bin = np.where(y_disc, np.clip(np.digitize(np.nan_to_num(delay, nan=-1.0), edges) - 1, 0, 5),
                     -1).astype(np.int8)
    y13 = np.where(cause == 0, 0, np.where(cause == 1, 1 + t_bin,
                                           np.where(cause == 2, 7 + t_bin, -1))).astype(np.int8)
    lab = {
        "key": key, "subject_id": subj, "hadm_id": subj * 10,
        "ordertime": np.int64(1_500_000_000) + seq * 3600,
        "y_disc": y_disc, "delay_h": delay, "cause": cause,
        "censor_h": rng.exponential(200, n).astype(np.float32),
        "t_bin": t_bin, "censor_bin": np.full(n, 5, np.int8), "y13": y13,
        "in_cohort": rng.random(n) > 0.002,
    }
    pq.write_table(pa.table(lab), out / "labels.parquet")
    pq.write_table(pa.table({"key": key, **cols}), out / "features.parquet")
    usub = np.unique(subj)
    h = splitmix64(usub, 7) % np.uint64(100)
    sp = np.where(h < 80, 0, np.where(h < 90, 1, 2)).astype(np.int8)
    pq.write_table(pa.table({"subject_id": usub, "split": sp,
                             "split_time": (h % np.uint64(2)).astype(np.int8)}),
                   out / "splits.parquet")
    (out / "vocab.json").write_text(json.dumps({"f_order_type": ot_names}, ensure_ascii=False),
                                    encoding="utf-8")
    print(f"[synth] 已生成 {n:,} 行 / {n_subj:,} 患者 -> {out}")


# --------------------------------------------------------------------------- 数据装载

def load_index(data_dir: Path, smoke: bool) -> dict:
    """读 labels + splits，算出每行的 split 与行掩码。只留 in_cohort。"""
    lp, sp = data_dir / "labels.parquet", data_dir / "splits.parquet"
    key = read_col(lp, "key").astype(np.int64)
    subj = read_col(lp, "subject_id").astype(np.int64)
    in_co = read_col(lp, "in_cohort").astype(bool)
    n = key.size
    print(f"labels 行数 {n:,}；in_cohort {in_co.sum():,}（{C.pct(in_co.sum(), n)}%）")

    s_sub = read_col(sp, "subject_id").astype(np.int64)
    s_spl = read_col(sp, "split").astype(np.int8)
    o = np.argsort(s_sub, kind="stable")
    s_sub, s_spl = s_sub[o], s_spl[o]
    pos = np.searchsorted(s_sub, subj)
    pos = np.clip(pos, 0, s_sub.size - 1)
    ok = s_sub[pos] == subj
    split = np.where(ok, s_spl[pos], -1).astype(np.int8)
    if (~ok).any():
        print(f"⚠ 有 {int((~ok).sum()):,} 行的 subject_id 不在 splits.parquet 里，已排除")
    del s_sub, s_spl, o, pos, ok
    valid = in_co & (split >= 0)

    hs = splitmix64(subj, SALT_SAMPLE)
    if smoke:
        valid &= (hs % np.uint64(1000)) < np.uint64(10)     # 1% subject
        print(f"[smoke] 抽 1% subject -> {int(valid.sum()):,} 行")
    return {"key": key, "subject_id": subj, "split": split, "valid": valid,
            "hash": hs, "n_rows": n}


def build_matrix(fpath: Path, cols: list[str], mask: np.ndarray, tag: str,
                 num_int: set[str] | None = None) -> np.ndarray:
    """按列读 features.parquet，只保留 mask 选中的行，拼成 float32 矩阵。

    逐列读是为了省内存：任一时刻只有一整列（~0.2 GB）在内存里。
    **整数型数值列**的缺失（-1 哨兵）转成 NaN 交给 LightGBM 当 missing；
    类别列保留 -1（LightGBM 自己把负值当 missing）；浮点列本来就是 NaN，不动。
    """
    num_int = num_int or set()
    nsel = int(mask.sum())
    X = np.empty((nsel, len(cols)), dtype=np.float32)
    t0 = time.time()
    for j, c in enumerate(cols):
        v = read_col(fpath, c)
        X[:, j] = v[mask]
        if c in num_int:
            np.putmask(X[:, j], X[:, j] == -1, np.nan)
        del v
        if (j + 1) % 10 == 0:
            print(f"  [{tag}] 已读 {j + 1}/{len(cols)} 列 ({time.time() - t0:.0f}s)")
    gc.collect()
    print(f"  [{tag}] 矩阵 {X.shape}，{X.nbytes / 2 ** 30:.2f} GiB，{time.time() - t0:.0f}s")
    return X


def pick_train_rows(idx: dict, max_rows: int) -> tuple[np.ndarray, float]:
    """训练集按 **subject** 抽样到 max_rows 以内（同一患者整体进出，避免患者内泄漏）。

    用哈希桶的累计行数精确选阈值，而不是猜比例——每个患者的医嘱条数差异极大。
    """
    m = idx["valid"] & (idx["split"] == 0)
    n_all = int(m.sum())
    if n_all <= max_rows:
        return m, 1.0
    b = (idx["hash"][m] % np.uint64(1024)).astype(np.int64)
    cum = np.cumsum(np.bincount(b, minlength=1024))
    k = int(np.searchsorted(cum, max_rows, side="right"))
    k = max(k, 1)
    sel = m & ((idx["hash"] % np.uint64(1024)) < np.uint64(k))
    frac = int(sel.sum()) / n_all
    print(f"训练集 {n_all:,} 行 -> 抽 {int(sel.sum()):,} 行（{frac * 100:.2f}%，"
          f"按 subject 哈希 {k}/1024 桶；内存约束，抽样率已写进 JSON）")
    return sel, frac


# --------------------------------------------------------------------------- LightGBM

def lgb_params(cfg: dict, objective: str, n_class: int = 0) -> dict:
    g = cfg["lgbm"]
    p = {
        "objective": objective, "learning_rate": g["learning_rate"],
        "num_leaves": g["num_leaves"], "min_child_samples": g["min_child_samples"],
        "feature_fraction": g["feature_fraction"], "bagging_fraction": g["bagging_fraction"],
        "bagging_freq": g["bagging_freq"], "num_threads": g["num_threads"],
        "verbosity": -1, "seed": cfg["seed"], "device_type": "cpu",   # GPU 归 s06 独占
        "max_bin": 255, "force_row_wise": True,
    }
    if objective == "multiclass":
        p.update(num_class=n_class, metric="multi_logloss")
    else:
        # 不用 is_unbalance / scale_pos_weight：那会破坏概率校准，
        # 而双头合成 p_self+p_other 正需要两个头的概率处在同一个尺度上。
        p.update(metric=["average_precision", "auc"])
    return p


def train_lgb(name, X, y, Xv, yv, cat_idx, feat_names, params, rounds, es, budget, ckpt=None):
    """训一个 LightGBM，返回 (booster, info)。X 由调用方持有，Dataset 用完即弃。"""
    import lightgbm as lgb
    t0 = time.time()
    dtr = lgb.Dataset(X, label=y, feature_name=feat_names, categorical_feature=cat_idx,
                      free_raw_data=True)
    dva = lgb.Dataset(Xv, label=yv, reference=dtr, free_raw_data=True)
    tb = TimeBudget(budget, name)
    multi = params.get("objective") == "multiclass"
    print(f"[{name}] 开训：{X.shape[0]:,}×{X.shape[1]} " +
          (f"{params['num_class']} 类" if multi else f"正类率 {float(np.mean(y)) * 100:.3f}%"))
    bst = lgb.train(params, dtr, num_boost_round=rounds, valid_sets=[dva], valid_names=["valid"],
                    callbacks=[lgb.early_stopping(es, verbose=False),
                               lgb.log_evaluation(100), tb])
    dt = time.time() - t0
    best = {k: round(float(v), 6) for k, v in bst.best_score.get("valid", {}).items()}
    info = {"train_s": round(dt, 1), "best_iteration": int(bst.best_iteration or bst.num_trees()),
            "n_trees": int(bst.num_trees()),
            "pos_rate_train": None if multi else round(float(np.mean(y)), 6),
            "valid_best_score_内部早停用": best, "hit_time_budget": tb.hit}
    if ckpt:
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        bst.save_model(str(ckpt), num_iteration=bst.best_iteration or None)
        info["checkpoint"] = str(ckpt.relative_to(C.ROOT)) if str(ckpt).startswith(str(C.ROOT)) \
            else ckpt.name
    print(f"[{name}] 完成 {dt:.0f}s，best_iter={info['best_iteration']}，{best}")
    del dtr, dva
    gc.collect()
    return bst, info


def predict_chunked(bst, X, chunk=1_000_000, n_class=1):
    """分块预测，避免 LightGBM 内部缓冲一次吃掉几个 GB。"""
    out = np.empty((X.shape[0], n_class) if n_class > 1 else (X.shape[0],), dtype=np.float32)
    ni = bst.best_iteration or None
    for a in range(0, X.shape[0], chunk):
        b = min(a + chunk, X.shape[0])
        out[a:b] = bst.predict(X[a:b], num_iteration=ni)
    return out


def top_gain(bst, names, k=30):
    g = bst.feature_importance("gain")
    o = np.argsort(-g)[:k]
    tot = float(g.sum()) or 1.0
    return [{"feature": names[i], "gain": round(float(g[i]), 2),
             "gain_pct": round(100 * float(g[i]) / tot, 3)} for i in o]


# --------------------------------------------------------------------------- 逻辑回归

class LogregPipe:
    """类别 one-hot（target-free，高基数截到 max_categories）+ 数值标准化 + SGD log_loss。

    不用 target/label 编码：那会引入标签信息，在这种「按患者切分」的设定下极易泄漏。
    """

    def __init__(self, cat_idx, num_idx, seed, max_cat=64, min_freq=1e-4):
        from sklearn.preprocessing import OneHotEncoder
        self.ci, self.ni, self.seed = np.array(cat_idx, int), np.array(num_idx, int), seed
        self.ohe = OneHotEncoder(handle_unknown="infrequent_if_exist", sparse_output=True,
                                 dtype=np.float32, max_categories=max_cat, min_frequency=min_freq)
        self.mu = self.sd = self.med = None

    def fit_transform(self, X):
        A = self.ohe.fit_transform(X[:, self.ci].astype(np.int32))
        N = X[:, self.ni].astype(np.float32)
        self.med = np.nanmedian(N, axis=0)
        self.med = np.where(np.isfinite(self.med), self.med, 0.0).astype(np.float32)
        N = np.where(np.isfinite(N), N, self.med)
        self.mu = N.mean(0)
        self.sd = np.where(N.std(0) > 1e-9, N.std(0), 1.0).astype(np.float32)
        return self._stack(A, (N - self.mu) / self.sd)

    def transform(self, X):
        A = self.ohe.transform(X[:, self.ci].astype(np.int32))
        N = X[:, self.ni].astype(np.float32)
        N = np.where(np.isfinite(N), N, self.med)
        return self._stack(A, (N - self.mu) / self.sd)

    @staticmethod
    def _stack(A, N):
        import scipy.sparse as sp
        return sp.hstack([A, sp.csr_matrix(N.astype(np.float32))], format="csr")

    def n_out(self):
        return len(self.ohe.get_feature_names_out()) + len(self.ni)


def fit_sgd(Z, y, seed, max_iter=12):
    from sklearn.linear_model import SGDClassifier
    m = SGDClassifier(loss="log_loss", penalty="l2", alpha=1e-6, max_iter=max_iter,
                      tol=1e-4, random_state=seed, average=True)
    m.fit(Z, y.astype(np.int8))
    return m


# --------------------------------------------------------------------------- 预测输出

def write_preds(out_dir: Path, model: str, split_name: str, key, subj,
                p_disc, p_self, p_other, p24, p13=None) -> dict:
    """按契约 §7 落 results/preds/<模型名>_<split>.parquet，并做 schema 自检。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    clip = float(np.mean(p_disc > 1.0))
    d = {"key": np.asarray(key, np.int64), "subject_id": np.asarray(subj, np.int64),
         "p_disc": np.clip(p_disc, 0, 1).astype(np.float32),
         "p_self": np.clip(p_self, 0, 1).astype(np.float32),
         "p_other": np.clip(p_other, 0, 1).astype(np.float32),
         "p_disc_24h": np.clip(p24, 0, 1).astype(np.float32)}
    if p13 is not None:
        for c in range(13):
            d[f"p13_{c}"] = p13[:, c].astype(np.float32)
        s = p13.sum(1)
        assert np.abs(s - 1).max() < 1e-3, f"p13_* 和不为 1（max dev {np.abs(s - 1).max():.2e}）"
    for k, v in d.items():
        assert v.shape[0] == len(key), f"{model}/{split_name} 列 {k} 长度不一致"
    p = out_dir / f"{model}_{split_name}.parquet"
    pq.write_table(pa.table(d), p, compression="zstd")
    print(f"  → preds {p.name}  {len(key):,} 行  p_disc 截断率 {clip * 100:.4f}%")
    return {"rows": len(key), "clip_gt1_pct": round(clip * 100, 4),
            "p_disc_mean": round(float(d["p_disc"].mean()), 6)}


# --------------------------------------------------------------------------- 子群方向分析

def subgroup_direction(Xtr, cause_tr, ot_col_idx, ot_vocab, cat_idx, names, params,
                       cfg, budget, ckpt_dir, rng_seed, max_rows=1_500_000, rounds=300):
    """「方向相反」的证据：在 Medications / IV therapy 上分别训自撤头与他撤头，
    报 top 特征 + 带符号的方向，并给一份**不依赖模型**的 pp 级佐证。

    两条证据都只输出聚合量：特征**列名** + 统计量，绝不输出任何取值。
    """
    import lightgbm as lgb
    out = {"说明": "自撤头/他撤头分别在子群上重训（轮数与行数都更小），"
                   "方向 = 特征值与其 SHAP 贡献的 Pearson 相关符号；"
                   "类别特征的整数编码无序，其方向仅供参考（code_order_meaningless=true）。"
                   "⚠ 这里的 90/10 只是**按行序**切的（行序≈按患者成块，非严格患者切分），"
                   "仅用于这段辅助分析的早停与 SHAP 取样，**不产生任何上报指标**；"
                   "正式指标一律走 s04 的患者级切分 + s07_eval。"
                   "model_free_pp 是不依赖模型的佐证：在**已知 cause 的撤销单**里，"
                   "按该特征的四分位看「自撤占比」，高四分位 − 低四分位的百分点差（pp）。",
           "subgroups": {}, "flip_table": [], "model_free_pp": {}}
    if ot_col_idx is None or not ot_vocab:
        out["skipped"] = "缺 f_order_type 列或 data/vocab.json，无法定位子群"
        return out
    name2code = {n: i for i, n in enumerate(ot_vocab)}
    ot = Xtr[:, ot_col_idx]
    numeric_set = {j for j in range(len(names)) if j not in set(cat_idx)}
    dirs = {}

    for g in SUBGROUPS:
        if g not in name2code:
            out["subgroups"][g] = {"skipped": f"词表里没有 {g}"}
            continue
        m = ot == float(name2code[g])
        ng = int(m.sum())
        if ng < 20_000:
            out["subgroups"][g] = {"n_train_rows": C.sup(ng), "skipped": "子群样本过少"}
            continue
        ridx = np.flatnonzero(m)
        if ridx.size > max_rows:
            ridx = np.random.default_rng(rng_seed).choice(ridx, max_rows, replace=False)
            ridx.sort()
        Xg, cg = Xtr[ridx], cause_tr[ridx]
        cut = int(0.9 * Xg.shape[0])
        rec = {"n_train_rows": ng, "n_used": int(Xg.shape[0]), "heads": {}}
        shap, boosters = {}, {}
        for head, pos in (("self", 1), ("other", 2)):
            y = (cg == pos).astype(np.int8)
            if y[:cut].sum() < 500 or y[cut:].sum() < 100:
                rec["heads"][head] = {"skipped": "子群内正类过少"}
                continue
            b, info = train_lgb(f"sub_{g}_{head}", Xg[:cut], y[:cut], Xg[cut:], y[cut:],
                                cat_idx, names, params, rounds, 50, budget,
                                ckpt_dir / f"lgbm_sub_{g.replace(' ', '_')}_{head}.txt")
            boosters[head] = b
            rec["heads"][head] = {"top_gain": top_gain(b, names, 15), **info}
            ns = min(80_000, Xg.shape[0] - cut)
            shap[head] = b.predict(Xg[cut:cut + ns], num_iteration=b.best_iteration or None,
                                   pred_contrib=True)
        if len(shap) == 2:
            ns = min(s.shape[0] for s in shap.values())
            Xs = Xg[cut:cut + ns]
            keep = set()
            for h in ("self", "other"):
                keep |= {names.index(d["feature"]) for d in rec["heads"][h]["top_gain"][:12]}
            fd = {}
            for j in sorted(keep):
                x = Xs[:, j].astype(np.float64)
                row = {"categorical": j not in numeric_set}
                for h in ("self", "other"):
                    s = shap[h][:ns, j].astype(np.float64)
                    row[f"mean_abs_shap_{h}"] = round(float(np.abs(s).mean()), 6)
                    ok = np.isfinite(x) & np.isfinite(s)
                    r = (float(np.corrcoef(x[ok], s[ok])[0, 1])
                         if ok.sum() > 100 and np.std(x[ok]) > 0 and np.std(s[ok]) > 0 else np.nan)
                    row[f"dir_{h}"] = None if not np.isfinite(r) else round(r, 4)
                if row["dir_self"] is not None and row["dir_other"] is not None:
                    row["self_minus_other"] = round(row["dir_self"] - row["dir_other"], 4)
                fd[names[j]] = row
            rec["direction"] = fd
            dirs[g] = fd
        out["subgroups"][g] = rec
        del Xg, cg
        gc.collect()

    a, b_ = SUBGROUPS
    if a in dirs and b_ in dirs:
        for f in sorted(set(dirs[a]) & set(dirs[b_])):
            x, y = dirs[a][f].get("self_minus_other"), dirs[b_][f].get("self_minus_other")
            if x is None or y is None:
                continue
            out["flip_table"].append({
                "feature": f, f"self_minus_other_{a}": x, f"self_minus_other_{b_}": y,
                "符号翻转": bool(x * y < 0), "翻转幅度": round(abs(x - y), 4),
                "categorical": dirs[a][f]["categorical"]})
        out["flip_table"].sort(key=lambda d: -d["翻转幅度"])

    # 不依赖模型的 pp 级佐证：数值特征四分位内「自撤占已知撤销的比例」
    feats = [d["feature"] for d in out["flip_table"][:12] if not d["categorical"]] or \
            [names[j] for j in sorted(numeric_set)][:8]
    for g in SUBGROUPS:
        if g not in name2code:
            continue
        m = (ot == float(name2code[g])) & np.isin(cause_tr, (1, 2))
        if m.sum() < 1000:
            continue
        cg, tab = cause_tr[m], {}
        for f in feats:
            j = names.index(f)
            x = Xtr[m, j].astype(np.float64)
            ok = np.isfinite(x)
            if ok.sum() < 1000:
                continue
            q = np.quantile(x[ok], [0.25, 0.75])
            if q[0] >= q[1]:
                continue
            lo, hi = ok & (x <= q[0]), ok & (x >= q[1])
            sl, nl = int((cg[lo] == 1).sum()), int(lo.sum())
            sh, nh = int((cg[hi] == 1).sum()), int(hi.sum())
            if min(nl, nh) < 10:
                continue
            tab[f] = {"自撤占比_低四分位pct": C.pct(sl, nl), "自撤占比_高四分位pct": C.pct(sh, nh),
                      "高−低_pp": round(C.pct(sh, nh) - C.pct(sl, nl), 2),
                      "n_低": C.sup(nl), "n_高": C.sup(nh)}
        out["model_free_pp"][g] = tab
    if len(out["model_free_pp"]) == 2:
        a_t, b_t = out["model_free_pp"][SUBGROUPS[0]], out["model_free_pp"][SUBGROUPS[1]]
        out["model_free_pp"]["符号相反的特征"] = sorted(
            f for f in set(a_t) & set(b_t) if a_t[f]["高−低_pp"] * b_t[f]["高−低_pp"] < 0)
    return out


# --------------------------------------------------------------------------- 主流程

def main() -> None:
    ap = argparse.ArgumentParser(description="s05 基线模型 + 预注册主实验")
    ap.add_argument("--smoke", action="store_true", help="抽 1%% subject 跑通")
    ap.add_argument("--synth", type=int, default=0, help="合成数据 dry run：患者数")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--max-train-rows", type=int, default=10_000_000)
    ap.add_argument("--max-val-rows", type=int, default=3_000_000, help="早停用的 val 子样本上限")
    ap.add_argument("--logreg-max-rows", type=int, default=2_000_000)
    ap.add_argument("--time-budget-s", type=float, default=1200.0, help="单个 LightGBM 的墙钟上限")
    ap.add_argument("--with13", action="store_true", help="额外训 13 类 multiclass（慢）")
    ap.add_argument("--rounds13", type=int, default=300)
    ap.add_argument("--no-subgroup", action="store_true")
    ap.add_argument("--models", default="all", help="逗号分隔：baserate,logreg,lgbm_single,lgbm_dual")
    ap.add_argument("--tag", default="", help="给 step/指标文件名加后缀，避免补跑时覆盖主结果")
    a = ap.parse_args()

    synth = a.synth > 0
    step = "s05_baseline_synth" if synth else ("s05_baseline_smoke" if a.smoke else "s05_baseline")
    step += a.tag
    data_dir = Path(a.data_dir) if a.data_dir else C.DATA
    out_root = Path(a.out_root) if a.out_root else C.ROOT
    t0 = time.time()
    if not synth:
        t0, _ = C.start_log(step)
    print(f"[{step}] data={data_dir}  out={out_root}")

    cfg = C.cfg()
    seed = int(cfg["seed"])
    want = None if a.models == "all" else set(a.models.split(","))
    preds_dir = out_root / ("results/preds_smoke" if a.smoke else "results/preds")
    ckpt_dir = out_root / "checkpoints" / ("smoke" if a.smoke else "")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    M = {"run": {"step": step, "smoke": a.smoke, "synth": synth,
                 "max_train_rows": a.max_train_rows, "time_budget_s": a.time_budget_s,
                 "seed": seed, "lgbm_params_from_config": cfg["lgbm"],
                 "preds_dir": str(preds_dir.relative_to(out_root)),
                 "备注": "本文件里所有 AUPRC 都是 sklearn 粗略自查值，正式指标见 s07_eval"},
         "models": {}, "preds": {}, "notes": []}

    if synth:
        make_synth(data_dir, a.synth, seed)

    # ---------- 1. 索引 + 列 ----------
    idx = load_index(data_dir, a.smoke)
    fpath = data_dir / "features.parquet"
    pf = pq.ParquetFile(fpath)
    fs, n_feat_rows = pf.schema_arrow, pf.metadata.num_rows
    if n_feat_rows != idx["n_rows"]:
        raise RuntimeError(f"features({n_feat_rows:,}) 与 labels({idx['n_rows']:,}) 行数不一致")
    fkey = read_col(fpath, "key").astype(np.int64)
    if not np.array_equal(fkey, idx["key"]):
        raise RuntimeError("features.parquet 与 labels.parquet 的 key 不是逐行对齐的；"
                           "契约 §3 要求同序对齐，请先修 s03。")
    del fkey
    names, num_cols, cat_cols, num_int = classify_cols(fs)
    M["leak_check"] = leak_check(list(fs.names))
    M["features"] = {"n_total": len(names), "n_numeric": len(num_cols),
                     "n_categorical": len(cat_cols), "numeric": num_cols,
                     "categorical": cat_cols, "整数型数值列_哨兵-1还原为NaN": sorted(num_int)}
    print(f"特征 {len(names)} 列：数值 {len(num_cols)}，类别 {len(cat_cols)}")
    if M["leak_check"]["allowlisted_disc_cols"]:
        print(f"ℹ 契约白名单内的滚动前缀统计（名字带 disc 但合法）："
              f"{M['leak_check']['allowlisted_disc_cols']}")
    cat_idx = [names.index(c) for c in cat_cols]

    ot_i = names.index("f_order_type") if "f_order_type" in names else None
    vpath = data_dir / "vocab.json"
    ot_vocab = []
    if vpath.exists():
        try:
            v = json.loads(vpath.read_text(encoding="utf-8"))
            for k in ("f_order_type", "order_type"):
                if isinstance(v.get(k), list):
                    ot_vocab = [str(x) for x in v[k]]
                elif isinstance(v.get(k), dict):          # {name: code} 或 {code: name}
                    d = v[k]
                    if all(str(x).lstrip("-").isdigit() for x in d.values()):
                        ot_vocab = [n for n, _ in sorted(d.items(), key=lambda kv: int(kv[1]))]
                    else:
                        ot_vocab = [n for _, n in sorted(d.items(), key=lambda kv: int(kv[0]))]
                if ot_vocab:
                    break
        except Exception as e:                            # noqa: BLE001
            print(f"⚠ vocab.json 解析失败：{e}")
    if not ot_vocab:
        M["notes"].append("未找到 f_order_type 词表，子群方向分析可能被跳过")

    # ---------- 2. 训练集 ----------
    tr_mask, tr_frac = pick_train_rows(idx, a.max_train_rows)
    va_mask = idx["valid"] & (idx["split"] == 1)
    te_mask = idx["valid"] & (idx["split"] == 2)
    lp = data_dir / "labels.parquet"
    y_disc_all = read_col(lp, "y_disc").astype(bool)
    cause_all = read_col(lp, "cause").astype(np.int8)
    tbin_all = read_col(lp, "t_bin").astype(np.int8)
    y13_all = read_col(lp, "y13").astype(np.int8)

    ytr, ctr, ttr = y_disc_all[tr_mask], cause_all[tr_mask], tbin_all[tr_mask]
    n_tr = ytr.size
    d = ytr.sum()
    r24_all = float(np.mean(ttr[ytr] <= 2)) if d else 0.0
    r24_self = float(np.mean(ttr[ctr == 1] <= 2)) if (ctr == 1).any() else 0.0
    r24_other = float(np.mean(ttr[ctr == 2] <= 2)) if (ctr == 2).any() else 0.0
    base = {"disc": float(ytr.mean()), "self": float((ctr == 1).mean()),
            "other": float((ctr == 2).mean()), "cause3": float((ctr == 3).mean()),
            "disc24h": float(np.mean(ytr & (ttr <= 2) & (ttr >= 0)))}
    M["cohort"] = {
        "n_train_rows_available": int((idx["valid"] & (idx["split"] == 0)).sum()),
        "n_train_rows_used": n_tr, "train_sample_frac": round(tr_frac, 6),
        "n_val_rows": int(va_mask.sum()), "n_test_rows": int(te_mask.sum()),
        "n_train_subjects": int(np.unique(idx["subject_id"][tr_mask]).size),
        "train_rates": {k: round(v, 6) for k, v in base.items()},
        "r24_given": {"all": round(r24_all, 6), "self": round(r24_self, 6),
                      "other": round(r24_other, 6)},
        "抽样说明": "内存约束下按 subject 哈希抽样，同一患者整体进出；val/test 未抽样（smoke 除外）",
    }
    M["cause3_penalty"] = {
        "训练集_cause3占全部撤销pct": round(100 * base["cause3"] / max(base["disc"], 1e-12), 4),
        "机制": "cause==3（撤销方或本单 provider 缺失，不知谁撤）在 lgbm_single 的标签里是正类，"
                "但在双头里两个头都算负类 ⇒ p_disc=p_self+p_other 会系统性偏低约这个量级。"
                "这是双头的真实劣势，主结果保留全部样本让它吃这个亏；"
                "另出 *_nc3 preds（模型不重训，仅评测行集合排除 cause==3）做敏感性分析。"}
    print(f"训练标签率：撤销 {base['disc'] * 100:.2f}%，自撤 {base['self'] * 100:.2f}%，"
          f"他撤 {base['other'] * 100:.2f}%，cause3 {base['cause3'] * 100:.3f}%")

    Xtr = build_matrix(fpath, names, tr_mask, "train", num_int)
    M["feature_invariants"] = check_feature_invariants(Xtr, names)
    # 类别特征的基数（只记**个数/最大编码**，绝不记取值——f_prov_code 是 provider 账号）
    M["features"]["categorical_cardinality"] = {
        c: int(np.nanmax(Xtr[:, names.index(c)])) + 1 for c in cat_cols}
    hi = {c: v for c, v in M["features"]["categorical_cardinality"].items() if v > 1000}
    if hi:
        M["notes"].append(f"高基数类别特征 {hi}：交给 LightGBM 原生 categorical 处理；"
                          f"logreg 侧按频次截到 max_categories=64（target-free）。"
                          f"f_prov_code 是 order_provider_id 的编码，**取值全程不输出**。")

    hv = idx["hash"][va_mask]
    if va_mask.sum() > a.max_val_rows:
        b = (hv % np.uint64(1024)).astype(np.int64)
        cum = np.cumsum(np.bincount(b, minlength=1024))
        kk = max(int(np.searchsorted(cum, a.max_val_rows, side="right")), 1)
        es_mask = va_mask & ((idx["hash"] % np.uint64(1024)) < np.uint64(kk))
    else:
        es_mask = va_mask.copy()
    print(f"早停用 val 子样本 {int(es_mask.sum()):,} 行")
    Xes = build_matrix(fpath, names, es_mask, "val_es", num_int)
    yes_d, ces = y_disc_all[es_mask], cause_all[es_mask]
    M["cohort"]["n_val_rows_earlystop"] = int(es_mask.sum())

    # ---------- 3. 训练 ----------
    g = cfg["lgbm"]
    P = lgb_params(cfg, "binary")
    rounds = int(g["n_estimators"]) if not (a.smoke or synth) else 120
    es_rounds = int(g["early_stopping_rounds"])
    boosters, logreg = {}, None

    if want is None or "lgbm_single" in want:
        boosters["single"], M["models"]["lgbm_single"] = train_lgb(
            "lgbm_single", Xtr, ytr.astype(np.int8), Xes, yes_d.astype(np.int8), cat_idx, names,
            P, rounds, es_rounds, a.time_budget_s, ckpt_dir / "lgbm_single.txt")
        M["models"]["lgbm_single"]["top30_gain"] = top_gain(boosters["single"], names)

    if want is None or "lgbm_dual" in want:
        for h, pos in (("self", 1), ("other", 2)):
            boosters[h], M["models"][f"lgbm_{h}_head"] = train_lgb(
                f"lgbm_{h}", Xtr, (ctr == pos).astype(np.int8), Xes, (ces == pos).astype(np.int8),
                cat_idx, names, P, rounds, es_rounds, a.time_budget_s,
                ckpt_dir / f"lgbm_{h}.txt")
            M["models"][f"lgbm_{h}_head"]["top30_gain"] = top_gain(boosters[h], names)

    if a.with13 and (want is None or "lgbm_13" in want):
        ytr13, yes13 = y13_all[tr_mask], y13_all[es_mask]
        k13 = ytr13 >= 0
        boosters["m13"], M["models"]["lgbm_13"] = train_lgb(
            "lgbm_13", Xtr[k13], ytr13[k13], Xes[yes13 >= 0], yes13[yes13 >= 0], cat_idx, names,
            lgb_params(cfg, "multiclass", 13), a.rounds13, 50, a.time_budget_s * 2,
            ckpt_dir / "lgbm_13.txt")
        M["models"]["lgbm_13"]["top30_gain"] = top_gain(boosters["m13"], names)
        M["models"]["lgbm_13"]["y13_excluded_rows"] = int((~k13).sum())

    if want is None or "logreg" in want:
        try:
            t = time.time()
            lr_n = min(a.logreg_max_rows, n_tr)
            sub = slice(None) if lr_n == n_tr else np.sort(
                np.random.default_rng(seed).choice(n_tr, lr_n, replace=False))
            pipe = LogregPipe(cat_idx, [names.index(c) for c in num_cols], seed)
            Z = pipe.fit_transform(Xtr[sub])
            print(f"[logreg] 设计矩阵 {Z.shape}，nnz={Z.nnz:,}（{Z.data.nbytes / 2 ** 20:.0f} MiB）")
            heads = {"disc": ytr[sub], "self": (ctr[sub] == 1), "other": (ctr[sub] == 2),
                     "d24": ytr[sub] & (ttr[sub] <= 2) & (ttr[sub] >= 0)}
            logreg = {k: fit_sgd(Z, v, seed) for k, v in heads.items()}
            M["models"]["logreg"] = {
                "train_s": round(time.time() - t, 1), "n_rows": int(lr_n),
                "n_design_cols": int(Z.shape[1]), "nnz_per_row": round(Z.nnz / Z.shape[0], 2),
                "heads": list(heads), "solver": "SGDClassifier(log_loss, l2, averaged)",
                "encoding": "类别 one-hot（max_categories=64，稀有类归并，target-free）+ 数值标准化",
                "备注": "p_disc 来自 disc 头；p_self/p_other/p_disc_24h 各自独立训一个头"}
            del Z
            gc.collect()
        except Exception as e:                            # noqa: BLE001
            print(f"⚠ logreg 失败：{type(e).__name__}: {e}")
            M["models"]["logreg"] = {"failed": f"{type(e).__name__}: {e}"}
            logreg = None

    # ---------- 4. 子群方向分析 ----------
    if not a.no_subgroup:
        try:
            M["subgroup_direction"] = subgroup_direction(
                Xtr, ctr, ot_i, ot_vocab, cat_idx, names, P, cfg,
                min(a.time_budget_s, 420), ckpt_dir, seed,
                max_rows=300_000 if (a.smoke or synth) else 1_500_000,
                rounds=80 if (a.smoke or synth) else 300)
        except Exception as e:                            # noqa: BLE001
            print(f"⚠ 子群方向分析失败：{type(e).__name__}: {e}")
            M["subgroup_direction"] = {"failed": f"{type(e).__name__}: {e}"}

    del Xtr, Xes
    gc.collect()

    # ---------- 5. 预测 ----------
    for sname, smask in (("val", va_mask), ("test", te_mask)):
        if smask.sum() == 0:
            continue
        X = build_matrix(fpath, names, smask, sname, num_int)
        k, s = idx["key"][smask], idx["subject_id"][smask]
        yd, cz, tb = y_disc_all[smask], cause_all[smask], tbin_all[smask]
        y24 = yd & (tb <= 2) & (tb >= 0)
        nc3 = cz != 3
        store = {}

        def emit(model, pd_, ps, po, p24, p13=None):
            M["preds"].setdefault(model, {})[sname] = write_preds(
                preds_dir, model, sname, k, s, pd_, ps, po, p24, p13)
            M["preds"][model][sname]["auprc_selfcheck_仅自查"] = {
                "p_disc_vs_y_disc": ap_selfcheck(yd, pd_),
                "p_self_vs_self": ap_selfcheck(cz == 1, ps),
                "p_other_vs_other": ap_selfcheck(cz == 2, po),
                "p_disc_24h_vs_y24": ap_selfcheck(y24, p24)}
            store[model] = pd_

        if want is None or "baserate" in want:
            o = np.ones(len(k), np.float32)
            emit("baserate", o * base["disc"], o * base["self"], o * base["other"],
                 o * base["disc24h"])
        if logreg:
            out = {h: np.empty(len(k), np.float32) for h in logreg}
            for aa in range(0, len(k), 500_000):       # 分块过 one-hot，避免稀疏矩阵吃爆内存
                bb = min(aa + 500_000, len(k))
                Zc = pipe.transform(X[aa:bb])
                for h, m in logreg.items():
                    out[h][aa:bb] = m.predict_proba(Zc)[:, 1]
                del Zc
            emit("logreg", out["disc"], out["self"], out["other"], out["d24"])
            del out
        if "single" in boosters:
            p = predict_chunked(boosters["single"], X)
            emit("lgbm_single", p, p * base["self"] / max(base["disc"], 1e-9),
                 p * base["other"] / max(base["disc"], 1e-9), p * r24_all)
            M["preds"]["lgbm_single"][sname]["说明"] = (
                "单头只训了 y_disc；p_self/p_other/p_disc_24h 由训练集的固定比例拆分而来"
                "（非独立训练的头），AUPRC 排序与 p_disc 完全一致，仅为满足契约 §7 的列。")
        if "self" in boosters and "other" in boosters:
            ps = predict_chunked(boosters["self"], X)
            po = predict_chunked(boosters["other"], X)
            pd_sum = ps + po
            pd_ind = 1 - (1 - ps) * (1 - po)
            p24 = ps * r24_self + po * r24_other
            emit("lgbm_dual", pd_sum, ps, po, p24)
            M["preds"]["lgbm_dual"][sname]["composition"] = {
                "primary": "p_disc = p_self + p_other（互斥竞争事件）",
                "alt_independent": "1-(1-p_self)(1-p_other)（假设独立共存，**不是**主结果）",
                "auprc_primary_仅自查": ap_selfcheck(yd, pd_sum),
                "auprc_alt_independent_仅自查": ap_selfcheck(yd, pd_ind),
                "p_disc_24h": "p_self*r24_self + p_other*r24_other（两个 cause 的时序分布不同，"
                              f"r24_self={r24_self:.3f} vs r24_other={r24_other:.3f}）"}
            if "single" in boosters:
                pnc = {"lgbm_single_nc3": store["lgbm_single"][nc3],
                       "lgbm_dual_nc3": pd_sum[nc3]}
                write_preds(preds_dir, "lgbm_single_nc3", sname, k[nc3], s[nc3],
                            pnc["lgbm_single_nc3"],
                            pnc["lgbm_single_nc3"] * base["self"] / max(base["disc"], 1e-9),
                            pnc["lgbm_single_nc3"] * base["other"] / max(base["disc"], 1e-9),
                            pnc["lgbm_single_nc3"] * r24_all)
                write_preds(preds_dir, "lgbm_dual_nc3", sname, k[nc3], s[nc3],
                            pd_sum[nc3], ps[nc3], po[nc3], p24[nc3])
                M["preds"].setdefault("_nc3_敏感性分析", {"做法":
                    "模型**不重训**，只把评测行集合限制到 cause!=3；"
                    "两个模型用同一批行，双头不再吃『撤了但不知谁撤』的亏"}).update(
                    {"n_rows_" + sname: int(nc3.sum()),
                     "excluded_" + sname: int((~nc3).sum())})
                M.setdefault("prereg_selfcheck_仅自查", {})[sname] = {
                    "auprc_lgbm_single": ap_selfcheck(yd, store["lgbm_single"]),
                    "auprc_lgbm_dual": ap_selfcheck(yd, pd_sum),
                    "delta_auprc": None, "判据": "ΔAUPRC>=0.01 且 bootstrap 95% CI 不含 0",
                    "nc3_auprc_single": ap_selfcheck(yd[nc3], store["lgbm_single"][nc3]),
                    "nc3_auprc_dual": ap_selfcheck(yd[nc3], pd_sum[nc3]),
                    "⚠": "无 CI、非正式；正式判定见 s07_eval 的 bootstrap"}
                q = M["prereg_selfcheck_仅自查"][sname]
                if q["auprc_lgbm_single"] and q["auprc_lgbm_dual"]:
                    q["delta_auprc"] = round(q["auprc_lgbm_dual"] - q["auprc_lgbm_single"], 6)
                    q["nc3_delta_auprc"] = round(q["nc3_auprc_dual"] - q["nc3_auprc_single"], 6)
            del ps, po, pd_sum, pd_ind, p24
        if "m13" in boosters:
            p13 = predict_chunked(boosters["m13"], X, n_class=13)
            p13 = (p13 / p13.sum(1, keepdims=True)).astype(np.float32)
            emit("lgbm_13", 1 - p13[:, 0], p13[:, 1:7].sum(1), p13[:, 7:13].sum(1),
                 p13[:, [1, 2, 3, 7, 8, 9]].sum(1), p13)
            del p13
        del X, store
        gc.collect()

    # ---------- 6. 落盘 ----------
    M["elapsed_total_s"] = round(time.time() - t0, 1)
    if synth:
        out = out_root / "s05_baseline_synth.json"
        M["_env"] = C.env_fingerprint()
        C.assert_no_pii(M, step)
        out.write_text(json.dumps(M, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"→ [synth] 指标写到 {out}")
    else:
        C.save_metrics(step, M, t0)
        pr = M.get("prereg_selfcheck_仅自查", {}).get("test", {})
        C.append_runlog(step, f"基线完成：{', '.join(M['models'])}；"
                              f"训练 {M['cohort']['n_train_rows_used']:,} 行"
                              f"（抽样率 {M['cohort']['train_sample_frac']:.3f}）；"
                              f"test 粗查 ΔAUPRC(dual−single)={pr.get('delta_auprc')}"
                              f"（仅自查，正式见 s07_eval）")
    print(f"[{step}] 全部完成，用时 {M['elapsed_total_s']:.0f}s")


if __name__ == "__main__":
    main()
