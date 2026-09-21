# 医嘱撤销预测（Order Retraction Prediction）

> **一句话**：别人预测病人会怎么样，我们预测**下这条医嘱的医生会不会把它撤回来、多久之后撤、
> 是自己撤还是别人撤**。
>
> NUS **BN5212** 小组课程项目 · 数据：MIMIC-IV v3.1 · 全部代码、日志、指标、报告都在本仓库里。

---

## 1. 这个项目在做什么

在医院的电子医嘱系统（CPOE）里，医生下达一条医嘱之后，这条医嘱有可能被撤销——
可能是医生自己反应过来剂量开错了，也可能是药师复核时拦下来、或者交接班后接手的同事换了方案。
**每一次撤销，都意味着药房、护理、检验环节的一次返工。**

我们做的事：**在医嘱刚被下达的那一刻（t₀），预测它会不会被撤销、多久之后被撤、以及会被谁撤。**

这件事之所以能做，是因为 MIMIC-IV 的 `hosp.poe` 表里有一个**系统自带的显式指针**
`discontinued_by_poe_id`，直接指出「这条医嘱是被哪条医嘱撤销的」。
标签是白送的——不需要人工病历回顾，不需要代理指标。实测双向匹配率 **99.98%**。

### 核心想法：撤销不是一件事，是两件方向相反的事

| | 自撤（self-cancel） | 他撤（other-cancel） |
|---|---|---|
| 是谁 | 开单医生**自己**把它撤了 | **别人**把它撤了 |
| 含义 | 自我纠错 | 复核拦截 / 方案被覆盖 / 交接班改方案 |
| **实测时延** | **38.6% 在 1 小时内**，之后单调递减 | 只有 4.8% 在 1 小时内，**峰值在 6–24 小时** |

两者的 hazard 形状完全不是一回事。把它们**合并成一个「会不会被撤」的二分类**，
等于把两股方向相反的力叠在一起训练，梯度会互相抵消。

⇒ **预注册主张**：自撤 / 他撤分头建模再合成，优于直接训单头。
⇒ **预注册判据**：test 集上 `ΔAUPRC ≥ 0.01` 且按患者重抽的 bootstrap 95% CI 不含 0。
⇒ **不达标就如实报告。** 这是课程项目，不是刷榜；负结果照样写进报告。

详细方案见 [`docs/00_项目方案.md`](docs/00_项目方案.md)，接口规范见 [`docs/01_数据契约.md`](docs/01_数据契约.md)。

---

## 2. 已经量出来的事实（全量，非抽样）

| 量 | 实测值 |
|---|---|
| 医嘱总数 | **52,212,109** 条 |
| 被撤销 | **10,653,951** 条（**20.41%**） |
| 指针可解析率 | **99.98%** |
| 撤销时延 p50 / p90 | **13.22 h** / 74.31 h |
| 自撤 : 他撤 | **42.56% : 57.44%**（provider 缺失导致无法判定的占撤销的 1.07%） |
| 出院/转科**批量清理**嫌疑 | 仅 **1.16%**（撤销发生在出院后或出院前 1 h 内）——**这个风险不成立**，85% 的撤销发生在出院前 24 h 以上 |
| 撤销率最高的医嘱类型 | Nutrition 40.35% > ADT 24.41% > General Care 23.98% > Medications 22.10% |
| 自撤占比最高的类型 | Consults 69.71% ≈ Radiology 68.23% ≈ Lab 68.14%，而 Medications 只有 39.11% |

全部数字来自 [`results/metrics/s02_labels.json`](results/metrics/s02_labels.json)，可复现。

---

## 3. 怎么复现

### 3.1 拿数据（这一步我们没法替你做）

MIMIC-IV 是 **PhysioNet 凭证数据**，DUA 禁止再分发。你需要自己：

1. 在 [PhysioNet](https://physionet.org/content/mimiciv/3.1/) 完成 CITI 培训并申请凭证；
2. 下载 MIMIC-IV v3.1；
3. 把压缩包路径填进 `configs/config.yaml` 的 `paths.mimic_zip`，或设环境变量 `MIMIC_IV_ZIP`。

**本仓库里没有任何患者数据**，`data/` 和 `checkpoints/` 都被 `.gitignore` 排除了。

### 3.2 装环境

```bash
pip install -r requirements.txt
# torch 请按自己的 CUDA 版本装，本机是 cu128：
# pip install torch --index-url https://download.pytorch.org/whl/cu128
```

### 3.3 按顺序跑

```bash
python src/s01_extract.py        # zip → parquet（不解压，流式读）      约 5 分钟
python src/s02_labels.py         # 指针 → 13 类竞争风险标签             约 1 分钟
python src/s03_features.py       # 场景 A 静态特征（含防泄漏滚动统计）
python src/s04_split.py          # 按 subject_id 确定性切分 80/10/10
python src/s05_baseline.py       # 基准率 / 逻辑回归 / LightGBM 单头与双头
python src/s06a_build_seq.py     # 构建医嘱序列
python src/s06_seq_model.py      # causal Transformer + 竞争风险头（GPU）
python src/s07_eval.py --compare lgbm_dual lgbm_single --split test   # 预注册主实验
python src/s08_figures.py        # 出图
```

每个脚本都会自动留下三份痕迹：
`logs/<步骤>.log`（运行日志）、`results/metrics/<步骤>.json`（聚合指标 + 环境指纹）、
`docs/02_实验记录.md`（一行摘要）。**脚本跑完不删。**

先跑小样本验证管线：`python src/s01_extract.py --smoke`。

---

## 4. 目录

```
Project/
├── README.md              本文件
├── AGENTS.md / CLAUDE.md  AI 协作协议（隐私红线、写码规矩、硬件约束）
├── requirements.txt
├── configs/config.yaml    路径、分桶、超参、随机种子——常数都在这里，不散在脚本里
├── docs/
│   ├── 00_项目方案.md       研究方案
│   ├── 01_数据契约.md       ★ 接口规范（列名/dtype/13 类编码/泄漏铁律/预测输出格式）
│   ├── 02_实验记录.md       每次运行自动追加
│   ├── 03_相关工作.md       文献综述
│   ├── 04_结果报告.md       最终结果
│   └── references/         .bib + 下载的 PDF
├── src/                   全部脚本，按编号即执行顺序
├── data/                  ⛔ 患者级中间产物，不进 git（但留在本地工程目录里）
├── checkpoints/           ⛔ 模型权重，不进 git
├── logs/                  ✅ 运行日志
└── results/
    ├── metrics/*.json     ✅ 聚合指标
    ├── preds/*.parquet    ⛔ 预测（含 key，属患者级派生数据）
    └── figures/*.png      ✅ 图
```

---

## 5. 🔴 隐私与合规（不可协商）

1. 患者级的行、标识符取值、绝对时间戳、记录级取值，**一律不得**进入
   git、报告、日志、图表、幻灯片、任何 LLM 的 prompt、任何网络检索。
2. 只允许输出 schema、计数、队列流、聚合统计；**n < 10 的格子一律抑制成 `"<10"`**。
3. `order_provider_id` / `field_value` 的**具体取值绝对不许打印**（本项目只用它们的整数编码）。
4. **不得把患者记录发给任何第三方 LLM API。**
5. 每次写指标文件时，`common.assert_no_pii()` 会做一次机械筛查——
   但它拦得住的东西远少于拦不住的，**真正的防线是「只写聚合量」这条纪律**。

---

## 6. 诚实的局限（会写进最终报告，不藏）

| # | 局限 |
|---|---|
| L1 | 「撤销」大量是**改剂量 / 改频次的修订**，不等于「医生反悔了」。技术措辞一律用 *discontinuation / revision* |
| L2 | `order_provider_id` 是**账号不是人**。「自撤」严格说是「同一账号撤的」 |
| L3 | 1.07% 的撤销因 provider 缺失无法判定自撤/他撤，被排除在 13 类损失之外（仍保留在二分类里） |
| L4 | `anchor_year_group` 只有 3 年一档，时间漂移分析的分辨率有限 |
| L5 | 本地 `icu.chartevents` 缺 28.3%、`hosp.emar_detail` 缺 34.4%，**全工程禁用这两张表** |
| L6 | MIMIC-IV 是**单中心**（BIDMC）数据，结论的外推性未经验证 |

---

## 7. 引用

如果这份代码对你有用，请同时引用 MIMIC-IV 数据集本身（见 `docs/references/references.bib`）。
