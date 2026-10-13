**English** | [中文](README_zh.md)

# Order Retraction Prediction · 医嘱撤销预测

**Predicting, at order-entry time, whether a clinical order will be discontinued — how soon, and by whom.**
*在医生下达医嘱的那一刻，预测这条医嘱会不会被撤回、多久后撤、以及会被谁撤。*

NUS **BN5212** course project · MIMIC-IV v3.1 · 52,212,109 orders

---

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

The documents under `docs/` are written in Chinese.

```
Project/
├── README.md              this file (English); README_zh.md is the Chinese version
├── AGENTS.md / CLAUDE.md  AI-collaboration protocol (privacy red lines, coding rules, hardware limits)
├── configs/config.yaml    paths, buckets, hyperparameters, random seeds — every constant lives here
├── docs/
│   ├── 00_项目方案.md       the original proposal, kept as written, with a header noting which premises were overturned
│   ├── 01_数据契约.md       ★ the data contract: column names / dtypes / 13-class coding / leakage rules / prediction output format
│   ├── 02_实验记录.md       experiment log, one line appended automatically on every run
│   ├── 03_相关工作.md       literature review: five strands, a comparison table, the research gap
│   ├── 04_结果报告.md       ★ full results and analysis
│   └── references/         20 .bib entries + 8 PDFs
├── src/                   12 scripts, numbered in execution order, never deleted after running
├── data/                  ⛔ patient-level intermediates, not in git
├── checkpoints/           ⛔ model weights, not in git
├── logs/                  ✅ 39 run logs
└── results/
    ├── metrics/*.json     ✅ 32 aggregate metric files
    ├── preds/*.parquet    ⛔ predictions (keyed, hence patient-level derived data)
    └── figures/*.png      ✅ 9 figures
```

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
