# 医嘱撤销预测 · Order Retraction Prediction

**在医生下达医嘱的那一刻，预测这条医嘱会不会被撤回、多久后撤、以及会被谁撤。**
*Predicting, at order-entry time, whether a clinical order will be discontinued — how soon, and by whom.*

NUS **BN5212** course project · MIMIC-IV v3.1 · 52,212,109 orders
[中文](#中文) · [English](#english)

---

<a name="中文"></a>

# 中文

## 1. 这个项目在做什么

医院的电子医嘱系统（CPOE）里，医生下达一条医嘱之后，它有可能被撤销——
可能是医生自己反应过来剂量开错了，也可能是药师复核时拦下来、或者交接班后接手的同事换了方案。
**每一次撤销，都意味着药房、护理、检验环节的一次返工。**

我们做的事：**在医嘱刚被下达的那一刻（t₀），预测它会不会被撤销、多久之后被撤、以及会被谁撤。**

标签是白送的：MIMIC-IV 的 `hosp.poe` 自带一个显式指针 `discontinued_by_poe_id`，
直接指出「这条医嘱是被哪条医嘱撤销的」。不需要人工病历回顾，不需要代理指标。
实测双向匹配率 **99.98%**。

### 核心想法：撤销不是一件事，是两件方向相反的事

| | 自撤（self-cancel） | 他撤（other-cancel） |
|---|---|---|
| 是谁 | 开单医生**自己**撤的 | **别人**撤的 |
| 含义 | 自我纠错 | 复核拦截 / 方案被覆盖 / 交接班改方案 |
| **实测时延** | **38.6% 在 1 小时内**，之后单调递减 | 只有 4.8% 在 1 小时内，**峰值在 6–24 小时** |

两者的 hazard 形状完全不是一回事。

## 2. 已经量出来的事实（全量，非抽样）

| 量 | 实测值 |
|---|---|
| 医嘱总数 | **52,212,109** 条 |
| 被撤销 | **10,653,951** 条（**20.41%**） |
| 指针可解析率 | **99.98%** |
| 撤销时延 p50 / p90 | **13.22 h** / 74.31 h |
| 自撤 : 他撤 | **42.56% : 57.44%**（1.07% 因 provider 缺失无法判定） |
| 出院/转科**批量清理**嫌疑 | 仅 **1.16%**——**这个风险不成立**，85% 的撤销发生在出院前 24 h 以上 |
| 撤销率最高的类型 | Nutrition 40.35% > ADT 24.41% > General Care 23.98% > Medications 22.10% |
| 自撤占比最高的类型 | Consults 69.71% ≈ Radiology 68.23% ≈ Lab 68.14%，而 Medications 仅 39.11% |

## 3. 模型结果（test，5,183,024 条医嘱，正类率 20.35%）

| 模型 | AUROC | **AUPRC** | Brier | ECE | **C-index(td)** | AUPRC@24h | 自撤 | 他撤 |
|---|---|---|---|---|---|---|---|---|
| 基准率 | 0.500 | 0.2035 | 0.1621 | 0.0007 | 0.500 | 0.1397 | 0.0860 | 0.1156 |
| 逻辑回归 | 0.7802 | 0.5105 | 0.1544 | 0.1329 | 0.7525 | 0.3370 | 0.2267 | 0.3416 |
| LightGBM 单头 | 0.8077 | 0.5636 | 0.1241 | 0.0048 | 0.7480 | 0.3820 | 0.2009 | 0.3644 |
| LightGBM 双头 | 0.8044 | 0.5570 | 0.1250 | 0.0065 | 0.7867 | 0.3664 | 0.2814 | 0.4248 |
| LightGBM 三分类 | 0.8050 | 0.5607 | 0.1247 | 0.0065 | 0.7872 | 0.3700 | 0.2827 | 0.4274 |
| **Transformer 单头** | **0.8266** | **0.6009** | **0.1187** | 0.0038 | 0.7665 | 0.4090 | 0.2075 | 0.4025 |
| **Transformer 竞争风险头** | 0.8208 | 0.5801 | 0.1242 | 0.0304 | **0.8066** | **0.4458** | **0.2998** | **0.4608** |

两个冠军各管一摊：**「会不会被撤」** 是 Transformer 单头最好；
**「谁会撤、什么时候撤」** 是 Transformer 竞争风险头最好。

### 说人话：模型有多准

| 挑出最可疑的 | 条数 | 其中真的被撤 | **命中率** | 相对瞎猜 | 抓住了全部撤销的 |
|---|---|---|---|---|---|
| 前 **1%** | 51,830 | 50,032 | **96.5%** | 4.74× | 4.7% |
| 前 **5%** | 259,151 | 218,100 | **84.2%** | 4.14× | 20.7% |
| 前 **10%** | 518,302 | 363,945 | **70.2%** | 3.45× | 34.5% |
| 前 20% | 1,036,604 | 562,733 | 54.3% | 2.67× | 53.4% |

⚠ **我们不报「准确率」**：20% 的医嘱会被撤，所以一个一律回答「不会被撤」的傻瓜模型
准确率就有 **79.65%**（我们的是 83.71%）。这个数会骗人，所以全程用 AUPRC 和上面这张命中率表。

## 4. 🔴 预注册主张：**四次独立检验全部不成立**

我们在**看数据之前**把判据写进了 `configs/config.yaml`：`ΔAUPRC ≥ 0.01` 且 bootstrap 95% CI 不含 0。

| 检验 | ΔAUPRC | 95% CI | 判据 |
|---|---|---|---|
| 表格 + LightGBM 双头 − 单头 | −0.0066 | [−0.0072, −0.0062] | ❌ |
| 表格 + LightGBM 三分类 − 单头 | −0.0029 | [−0.0034, −0.0025] | ❌ |
| **序列 + Transformer，同编码器只换最后一层** | **−0.0208** | [−0.0214, −0.0202] | ❌ |
| **场景 B：把携带反向效应的 eMAR 特征放进来** | −0.0021（无 eMAR 时 −0.0017） | [−0.0029, −0.0014] | ❌ |

第三行是最干净的对照（同一份序列、同一个 8 层编码器、同样 3 个 epoch，唯一差别是输出层），
它输得**最多**。第四行是对机制假说最直接的否证：
原推理链是「存在方向相反的协变量效应 → 单头梯度抵消 → 分头更准」，
**前件成立**（`f_prov_prior_self_rate` 对自撤 +0.856 / 对他撤 −0.889，近乎完美反对称），
**后件不成立**——断掉的是中间那一步。

尺度参照：同一套 bootstrap 下，「LightGBM − 逻辑回归」= **+0.0532**，
「Transformer 单头 − LightGBM 单头」= **+0.037**。
**换表示、换模型族的收益，是 cause 分解那点差异的一个数量级以上。**

### 但分解买到了单头**在结构上给不出**的东西

同一组干净对照（Transformer，只换头）：逐因 AUPRC 自撤 **+45%**、他撤 **+14%**，
C-index(td) **+0.040**。单头只能说「这条医嘱 70% 会被撤」，
说不出「大概率 6 小时内被**别人**撤」。（这三项不是预注册判据，按事后发现标注。）

## 5. 怎么复现

### 5.1 拿数据（这一步我们没法替你做）

MIMIC-IV 是 **PhysioNet 凭证数据**，DUA 禁止再分发。你需要自己：

1. 在 [PhysioNet](https://physionet.org/content/mimiciv/3.1/) 完成 CITI 培训并申请凭证；
2. 下载 MIMIC-IV v3.1；
3. 把压缩包路径填进 `configs/config.yaml` 的 `paths.mimic_zip`，或设环境变量 `MIMIC_IV_ZIP`。

**本仓库里没有任何患者数据**，`data/` 和 `checkpoints/` 都被 `.gitignore` 排除了。

### 5.2 装环境并按顺序跑

```bash
pip install -r requirements.txt
# torch 按自己的 CUDA 版本装，本机是 cu128：
# pip install torch --index-url https://download.pytorch.org/whl/cu128

python src/s01_extract.py        # zip → parquet（不解压，流式读）        ~5 min
python src/s02_labels.py         # 指针 → 13 类竞争风险标签               ~1 min
python src/s03_features.py       # 30 个场景 A 特征（含防泄漏滚动统计）    ~6 min
python src/s04_split.py          # 按 subject_id 确定性切分 80/10/10      ~30 s
python src/t01_leak_test.py      # ★ 泄漏审计，改完特征必跑               ~2 min
python src/s05_baseline.py       # 基准率 / 逻辑回归 / LightGBM 三种头     ~85 min
python src/s06a_build_seq.py     # 构建医嘱序列                           ~4 min
python src/s06_seq_model.py --variant cr      # Transformer 竞争风险头    ~105 min (GPU)
python src/s06_seq_model.py --variant single  # 单头对照                  ~66 min (GPU)
python src/s07_eval.py --compare transformer_cr transformer_single --split test
python src/s08_figures.py        # 出 9 张图
python src/s09_scenario_b.py     # 场景 B：eMAR landmark                  ~13 min
```

先跑小样本验证管线：`python src/s01_extract.py --smoke`。
每个脚本自动留三份痕迹：`logs/<步骤>.log`、`results/metrics/<步骤>.json`（含环境指纹 + git commit）、
`docs/02_实验记录.md` 追加一行。**脚本跑完不删。**

### 5.3 本机实测环境

Python 3.11 / Windows 11 / **RTX 3060 Laptop 6 GB** / 20 逻辑核 / 16 GB 内存。
Transformer 显存峰值实测仅 **0.791 GiB**（bf16 + flash attention）。

## 6. 目录

```
Project/
├── README.md              本文件
├── AGENTS.md / CLAUDE.md  AI 协作协议（隐私红线、写码规矩、硬件约束）
├── configs/config.yaml    路径、分桶、超参、随机种子——常数都在这里
├── docs/
│   ├── 00_项目方案.md       立项时的原始提案（原样保留，附「哪些前提被推翻了」的头注）
│   ├── 01_数据契约.md       ★ 接口规范（列名/dtype/13 类编码/泄漏铁律/预测输出格式）
│   ├── 02_实验记录.md       每次运行自动追加
│   ├── 03_相关工作.md       文献综述（五支 + 对照表 + 研究空白）
│   ├── 04_结果报告.md       ★ 完整结果与分析
│   └── references/         20 条 .bib + 8 篇 PDF
├── src/                   12 个脚本，按编号即执行顺序，跑完不删
├── data/                  ⛔ 患者级中间产物，不进 git
├── checkpoints/           ⛔ 模型权重，不进 git
├── logs/                  ✅ 39 份运行日志
└── results/
    ├── metrics/*.json     ✅ 32 份聚合指标
    ├── preds/*.parquet    ⛔ 预测（含 key，属患者级派生数据）
    └── figures/*.png      ✅ 9 张图
```

## 7. 🔴 隐私与合规（不可协商）

1. 患者级的行、标识符取值、绝对时间戳、记录级取值，**一律不得**进入
   git、报告、日志、图表、幻灯片、任何 LLM 的 prompt、任何网络检索。
2. 只允许输出 schema、计数、队列流、聚合统计；**n < 10 的格子一律抑制成 `"<10"`**。
3. `order_provider_id` / `field_value` 的**具体取值绝对不许打印**（只用整数编码）。
4. **不得把患者记录发给任何第三方 LLM API。**

## 8. 工程质量：四个只有独立审计才抓得出来的缺陷

这可能是本项目最值得看的部分。我们写了一个用**完全不同算法**（组内暴力双重循环）
独立复算的审计脚本 `src/t01_leak_test.py`，它逐层剥出四个缺陷：

| # | 缺陷 | 后果 | 怎么发现的 |
|---|---|---|---|
| 1 | float32 反推撤销时刻，在分钟粒度上翻车 | 计数特征 16.7% 的行差 1 | 暴力法独立复算 |
| 2 | 前缀口径不一致 ⇒ **自指泄漏**（191 条医嘱把「自己已被撤销」数进自己的特征，**专挑正类**） | 38 行逻辑上不可能的结果 | 同上 |
| 3 | 43 条时间倒挂的数据异常（最极端 10.3 年） | 破坏不变式 | **写成断言后当场炸的** |
| 4 | 评测库标签对齐用了排列的**逆置换** | AUROC 0.808 → **0.542**，全部数字作废 | **两条独立管线报出不同的数** |

前三个**不会让任何指标报警**；第四个恰恰相反，它让所有指标错得离谱，
却因为**自测数据碰巧有序**（有序时排列与其逆置换都等于恒等映射）而躲过了 40 项自测。

> **教训一**：凡是涉及排列 / 重排序的代码，测试输入必须**故意打乱**。
> **教训二**：模型脚本里那些「仅自查」的冗余指标不要删——正是它抓出了评测库的致命 bug。
> **教训三**：把不变式写成**断言**比写进文档有用。

另有一个插曲：场景 B 的 1% smoke 跑出「−0.0053 → −0.0004」，看上去正是机制在起作用；
**全量是「−0.0017 → −0.0021」，那个"改善"是噪声。** 小样本上看到符合预期的趋势最骗人。

## 9. 诚实的局限

| # | 局限 |
|---|---|
| L1 | 「撤销」大量是改剂量/改频次的**修订**，不等于「医生反悔了」。技术措辞一律用 *discontinuation / revision* |
| L2 | `order_provider_id` 是**账号不是人**；「自撤」严格说是「同一账号撤的」 |
| L3 | 1.07% 的撤销因 provider 缺失无法判定 cause，排除在 13 类损失外（保留在二分类里） |
| L4 | `anchor_year_group` 只有 3 年一档，时间漂移分析分辨率有限 |
| L5 | 本地 `icu.chartevents` 缺 28.3%、`hosp.emar_detail` 缺 34.4%，**全工程禁用** |
| L6 | MIMIC-IV 是**单中心**（BIDMC）数据，外推性未经验证 |
| L7 | 立项时引用的 −14.51pp/+12.02pp 反向效应**不可复现**（那一格 n 只有 716），详见结果报告 §6.8 |

---

<a name="english"></a>

# English

## 1. What this project does

In a hospital's computerized provider order entry (CPOE) system, an order a clinician has just
placed may later be discontinued — because the clinician catches their own dosing mistake,
because a pharmacist intercepts it on verification, or because the colleague taking over after
handoff changes the plan. **Every discontinuation means rework for pharmacy, nursing, or the lab.**

We predict, **at the moment the order is placed (t₀), whether it will be discontinued,
how soon, and by whom.**

Labels come for free: MIMIC-IV's `hosp.poe` carries an explicit pointer
`discontinued_by_poe_id` stating exactly which order discontinued which. No chart review,
no proxy outcome. Measured bidirectional match rate: **99.98%**.

### Core idea: discontinuation is not one process but two, pulling in opposite directions

| | Self-cancel | Other-cancel |
|---|---|---|
| Who | The **ordering clinician** | **Someone else** |
| Meaning | Self-correction | Pharmacist interception / plan override / handoff change |
| **Measured delay** | **38.6% within 1 hour**, monotonically decreasing after | Only 4.8% within 1 hour, **peaks at 6–24 hours** |

The two hazard shapes are simply not the same thing.

## 2. Measured facts (full data, not sampled)

| Quantity | Value |
|---|---|
| Total orders | **52,212,109** |
| Discontinued | **10,653,951** (**20.41%**) |
| Pointer resolution rate | **99.98%** |
| Delay p50 / p90 | **13.22 h** / 74.31 h |
| Self : other | **42.56% : 57.44%** (1.07% undeterminable, missing provider) |
| Discharge-time **bulk cleanup** suspicion | only **1.16%** — **this risk does not hold**; 85% of discontinuations occur >24 h before discharge |
| Highest discontinuation rate | Nutrition 40.35% > ADT 24.41% > General Care 23.98% > Medications 22.10% |
| Highest self-cancel share | Consults 69.71% ≈ Radiology 68.23% ≈ Lab 68.14%, vs Medications only 39.11% |

## 3. Model results (test, 5,183,024 orders, 20.35% positive)

| Model | AUROC | **AUPRC** | Brier | ECE | **C-index(td)** | AUPRC@24h | Self | Other |
|---|---|---|---|---|---|---|---|---|
| Base rate | 0.500 | 0.2035 | 0.1621 | 0.0007 | 0.500 | 0.1397 | 0.0860 | 0.1156 |
| Logistic regression | 0.7802 | 0.5105 | 0.1544 | 0.1329 | 0.7525 | 0.3370 | 0.2267 | 0.3416 |
| LightGBM single-head | 0.8077 | 0.5636 | 0.1241 | 0.0048 | 0.7480 | 0.3820 | 0.2009 | 0.3644 |
| LightGBM dual-head | 0.8044 | 0.5570 | 0.1250 | 0.0065 | 0.7867 | 0.3664 | 0.2814 | 0.4248 |
| LightGBM 3-class | 0.8050 | 0.5607 | 0.1247 | 0.0065 | 0.7872 | 0.3700 | 0.2827 | 0.4274 |
| **Transformer single-head** | **0.8266** | **0.6009** | **0.1187** | 0.0038 | 0.7665 | 0.4090 | 0.2075 | 0.4025 |
| **Transformer competing-risks head** | 0.8208 | 0.5801 | 0.1242 | 0.0304 | **0.8066** | **0.4458** | **0.2998** | **0.4608** |

Two winners, two jobs: **"will it be discontinued"** goes to the Transformer single-head;
**"who will cancel it, and when"** goes to the Transformer competing-risks head.

### In plain terms: how accurate is it?

| Flag the top | Orders | Actually discontinued | **Precision** | vs. random | Recall |
|---|---|---|---|---|---|
| **1%** | 51,830 | 50,032 | **96.5%** | 4.74× | 4.7% |
| **5%** | 259,151 | 218,100 | **84.2%** | 4.14× | 20.7% |
| **10%** | 518,302 | 363,945 | **70.2%** | 3.45× | 34.5% |
| 20% | 1,036,604 | 562,733 | 54.3% | 2.67× | 53.4% |

⚠ **We deliberately do not report "accuracy."** With a 20% positive rate, a trivial model that
always answers "will not be discontinued" already scores **79.65%** (ours is 83.71%).
That number misleads, so we use AUPRC and the precision table above throughout.

## 4. 🔴 The preregistered claim: **refuted by four independent tests**

The criterion was written into `configs/config.yaml` **before looking at any results**:
`ΔAUPRC ≥ 0.01` with a bootstrap 95% CI excluding 0.

| Test | ΔAUPRC | 95% CI | Criterion |
|---|---|---|---|
| Tabular + LightGBM, dual − single | −0.0066 | [−0.0072, −0.0062] | ❌ |
| Tabular + LightGBM, 3-class − single | −0.0029 | [−0.0034, −0.0025] | ❌ |
| **Sequence + Transformer, same encoder, only the final layer swapped** | **−0.0208** | [−0.0214, −0.0202] | ❌ |
| **Scenario B: with the very eMAR feature that carries the reversed effect** | −0.0021 (vs −0.0017 without eMAR) | [−0.0029, −0.0014] | ❌ |

Row 3 is the cleanest possible comparison — identical sequences, identical 8-layer encoder,
identical 3 epochs, the *only* difference being the output layer — and it loses by the
**largest** margin. Row 4 directly refutes the mechanism: the reasoning chain was
"opposite-signed covariate effects ⇒ gradient cancellation in the single head ⇒ decomposition wins."
The **antecedent holds** (`f_prov_prior_self_rate` has SHAP direction +0.856 on self-cancel and
−0.889 on other-cancel — almost perfectly antisymmetric), but the **consequent does not**.
What breaks is the step in between.

For scale, under the same bootstrap: LightGBM − logistic regression = **+0.0532**;
Transformer single-head − LightGBM single-head = **+0.037**.
**Changing the representation buys an order of magnitude more than changing the output head.**

### But the decomposition buys something a single head structurally cannot give

In that same clean comparison: per-cause AUPRC **+45%** (self) and **+14%** (other),
time-dependent C-index **+0.040**. A single head can say "this order has a 70% chance of being
discontinued"; it cannot say "most likely discontinued by **someone else** within 6 hours."
(These three are not the preregistered criterion and are labelled exploratory.)

## 5. Reproducing this

### 5.1 Get the data (we cannot do this for you)

MIMIC-IV is **PhysioNet credentialed data**; the DUA forbids redistribution. You must:

1. Complete CITI training and request credentials at
   [PhysioNet](https://physionet.org/content/mimiciv/3.1/);
2. Download MIMIC-IV v3.1;
3. Point `paths.mimic_zip` in `configs/config.yaml` at your archive, or set `MIMIC_IV_ZIP`.

**This repository contains no patient data.** `data/` and `checkpoints/` are gitignored.

### 5.2 Install and run in order

```bash
pip install -r requirements.txt
# install torch for your CUDA version; this machine used cu128:
# pip install torch --index-url https://download.pytorch.org/whl/cu128

python src/s01_extract.py        # zip → parquet (streamed, never unpacked)   ~5 min
python src/s02_labels.py         # pointers → 13-class competing-risks labels ~1 min
python src/s03_features.py       # 30 scenario-A features (leak-safe rolling) ~6 min
python src/s04_split.py          # deterministic 80/10/10 split by subject_id ~30 s
python src/t01_leak_test.py      # ★ leakage audit — rerun after any feature change
python src/s05_baseline.py       # base rate / logreg / three LightGBM heads   ~85 min
python src/s06a_build_seq.py     # build order sequences                       ~4 min
python src/s06_seq_model.py --variant cr      # competing-risks head     ~105 min (GPU)
python src/s06_seq_model.py --variant single  # single-head control       ~66 min (GPU)
python src/s07_eval.py --compare transformer_cr transformer_single --split test
python src/s08_figures.py        # 9 figures
python src/s09_scenario_b.py     # scenario B: eMAR landmark analysis         ~13 min
```

Smoke-test the pipeline first: `python src/s01_extract.py --smoke`.
Every script leaves three traces automatically: `logs/<step>.log`,
`results/metrics/<step>.json` (with an environment fingerprint and git commit), and one
appended line in `docs/02_实验记录.md`. **Scripts are never deleted after running.**

### 5.3 Hardware this was measured on

Python 3.11 / Windows 11 / **RTX 3060 Laptop, 6 GB** / 20 logical cores / 16 GB RAM.
Transformer peak GPU memory measured at just **0.791 GiB** (bf16 + flash attention).

## 6. Layout

See the directory tree in the Chinese section above. In short: `src/` holds 12 numbered scripts
(execution order = numeric order), `docs/` holds the proposal, the **data contract**
(the single authoritative interface spec), the experiment log, the literature review and the
**full results report**; `results/` holds aggregate metrics, figures and predictions;
`data/` and `checkpoints/` stay local and are gitignored.

## 7. 🔴 Privacy and compliance (non-negotiable)

1. Patient-level rows, identifier values, absolute timestamps and record-level values must
   **never** enter git, reports, logs, figures, slides, any LLM prompt, or any web search.
2. Only schemas, counts, cohort flow and aggregate statistics may be emitted;
   **any cell with n < 10 is suppressed to `"<10"`**.
3. Values of `order_provider_id` and `field_value` are **never printed** (integer codes only).
4. **Patient records are never sent to any third-party LLM API.**

## 8. Engineering: four defects only an independent audit could catch

Possibly the most instructive part of this project. `src/t01_leak_test.py` recomputes the
leak-sensitive features with a **completely different algorithm** (brute-force double loop
within each admission). It peeled off four defects, one layer at a time:

| # | Defect | Consequence | How it surfaced |
|---|---|---|---|
| 1 | float32 back-derivation of the discontinuation timestamp, wrong at minute granularity | a count feature off by 1 on 16.7% of rows | brute-force recomputation |
| 2 | inconsistent prefix semantics ⇒ **self-referential leakage** (191 orders counted "I have already been discontinued" into their own feature — **targeting the positive class**) | 38 logically impossible rows | same |
| 3 | 43 time-inverted data anomalies (worst: 10.3 years) | broke the invariant | **an assertion blew up** |
| 4 | evaluation library aligned labels with the **inverse** of a permutation | AUROC 0.808 → **0.542**; every number invalid | **two independent pipelines disagreed** |

The first three **set off no metric alarm at all**. The fourth is the opposite: it made every
metric wildly wrong, yet slipped past 40 self-tests because **the synthetic test keys happened to
be sorted** (when keys are sorted, a permutation and its inverse are both the identity).

> **Lesson 1**: any code involving permutations or re-ordering must be tested on
> **deliberately shuffled** input. Sorted test data cannot detect a direction error.
> **Lesson 2**: don't delete the "self-check only" redundant metrics in model scripts —
> that redundancy is exactly what caught the fatal evaluation bug.
> **Lesson 3**: an invariant written as an **assertion** is worth more than one written in a doc.

One more cautionary note: scenario B's 1% smoke run produced "−0.0053 → −0.0004", which looked
exactly like the hypothesized mechanism kicking in. **The full run gave "−0.0017 → −0.0021";
that "improvement" was noise.** Seeing the trend you expected on a small sample is the easiest
way to fool yourself.

## 9. Honest limitations

| # | Limitation |
|---|---|
| L1 | Many "discontinuations" are dose/frequency **revisions**, not the clinician changing their mind. We use *discontinuation / revision* throughout |
| L2 | `order_provider_id` is an **account, not a person**; "self-cancel" strictly means "same account" |
| L3 | 1.07% of discontinuations have an undeterminable cause (missing provider); excluded from the 13-class loss, retained in the binary task |
| L4 | `anchor_year_group` is only 3-year granular, limiting temporal-drift analysis |
| L5 | The local copy of `icu.chartevents` is 28.3% incomplete and `hosp.emar_detail` 34.4%; **both are banned project-wide** |
| L6 | MIMIC-IV is **single-centre** (BIDMC); external validity is unverified |
| L7 | The −14.51pp/+12.02pp reversed effect cited at proposal time **does not replicate** (that cell had n = 716); see report §6.8 |

## 10. Citing

If this code is useful to you, please also cite the MIMIC-IV dataset itself — see
`docs/references/references.bib`.
