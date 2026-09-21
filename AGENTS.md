# AGENTS.md —— 本仓库的 AI 协作协议

任何 AI 编码助手（Claude Code / Codex / Cursor / Copilot Workspace / 其它 agent）
在本仓库工作前**必须先读完本文件**。Claude Code 另见 `CLAUDE.md`（内容一致，外加本机陷阱）。

---

## 1. 这是什么项目

NUS **BN5212** 小组课程项目：**在医嘱下达时刻预测这条医嘱会不会被撤销。**

- 数据：MIMIC-IV v3.1 `hosp` 模块（PhysioNet 凭证数据，**不在本仓库里**）
- 标签白送：`hosp.poe.discontinued_by_poe_id` 是系统自带的显式指针，双向匹配率 99.98%，不需要人工病历回顾
- 三个预测目标：① 会不会被撤　② 多久后被撤　③ **自己撤还是别人撤**
- 主模型：把撤销切成 6 个时间桶 × 2 个 cause + 1 个「未撤」= **13 类离散时间竞争风险**
- 核心主张（预注册、可证伪）：**撤销是两个方向相反的过程的混合**，
  自撤/他撤分头建模后再合成，会优于直接训一个「会不会被撤」的单头。
  判据：held-out 患者上 `ΔAUPRC >= 0.01` 且 bootstrap 95% CI 不含 0。**不成立就如实报告。**

完整方案见 `docs/00_项目方案.md`，接口规范见 **`docs/01_数据契约.md`（唯一权威）**。

---

## 2. 🔴 隐私红线（不可协商，违反一次就是事故）

MIMIC-IV 是 **PhysioNet 凭证数据**，受 DUA 约束。

1. **患者级的行、标识符取值、绝对时间戳、记录级取值，一律不得**进入
   git、报告、日志、图表、幻灯片、任何 LLM 的 prompt、任何网络检索。
2. 只允许输出：schema、计数、队列流、聚合统计。
   **n < 10 的格子一律抑制成 `"<10"`**（用 `common.sup()`）。
3. `order_provider_id` / `caregiver_id` / `field_value` 的**具体取值绝对不许打印**。
4. **不得把患者记录发给任何第三方 LLM API**（包括但不限于 OpenAI / DeepSeek / Gemini）。
   合规的做法只有三种：① 只发 schema / 列名 / 词表 / 聚合统计级的提示词；
   ② 用 PhysioNet 批准的受控云环境；③ 用结构相同的合成数据。
5. `data/` 与 `checkpoints/` 已被 `.gitignore` 排除。**不要用 `git add -f` 绕过。**
   组员拿不到数据是正常的——他们应当用**自己的 PhysioNet 凭证**下载原始 zip，本地重跑 `s01`。

写任何输出前，`common.save_metrics()` 会自动跑一遍 `assert_no_pii()`。
**它拦不住的东西比它能拦的多**，真正的防线是「只写聚合量」这条纪律。

---

## 3. 目录与流水线

```
Project/
├── README.md              项目总览 + 复现步骤
├── AGENTS.md / CLAUDE.md  本协议
├── requirements.txt
├── configs/config.yaml    路径、分桶、超参、随机种子
├── docs/
│   ├── 00_项目方案.md       研究方案（为什么做、做什么）
│   ├── 01_数据契约.md       ★ 接口规范，写码前必读
│   ├── 02_实验记录.md       每次运行自动追加一行
│   ├── 03_相关工作.md       文献综述
│   ├── 04_结果报告.md       最终结果
│   └── references/         .bib + 下载的 PDF
├── src/                   全部脚本，**跑完不删**
│   ├── common.py          路径/流式读zip/时间转换/隐私抑制/日志/指标
│   ├── s01_extract.py     zip → parquet
│   ├── s02_labels.py      指针 → 13 类竞争风险标签
│   ├── s03_features.py    场景 A 静态特征（含防泄漏的滚动统计）
│   ├── s04_split.py       按 subject_id 切分
│   ├── s05_baseline.py    基准率 / 逻辑回归 / LightGBM
│   ├── s06_seq_model.py   causal Transformer + 13 类竞争风险头（GPU）
│   ├── s07_eval.py        统一评测（所有模型走同一套指标函数）
│   └── s08_figures.py     出图
├── data/                  ⛔ 患者级中间产物，不进 git
├── checkpoints/           ⛔ 模型权重（派生数据），不进 git
├── logs/                  ✅ 运行日志，进 git（留痕）
└── results/
    ├── metrics/*.json     ✅ 聚合指标，进 git
    └── figures/*.png      ✅ 图，进 git
```

执行顺序就是脚本编号顺序。每一步都写日志 + 指标 + 实验记录三份留痕。

---

## 4. 写码规矩

1. **先读 `docs/01_数据契约.md`**，按里面约定的列名/dtype/编码读写，不要猜上一步产出了什么。
   要改契约，先改那份文件，再改代码，并在 `docs/02_实验记录.md` 写一行为什么。
2. 每个脚本开头 `import common as C`，用 `C.start_log()` / `C.save_metrics()` / `C.append_runlog()`，
   **不要各写一套日志和指标**。
3. **脚本跑完不删，临时脚本不要放进 `src/`**（放系统临时目录）。
   一次性的探查脚本如果结论有用，就整理成 `src/` 下的正式脚本，否则丢掉。
4. **内存**：本机 16 GB、常态可用约 6 GB，而 poe 有 5200 万行。
   大表一律 pyarrow + numpy 向量化，**不要 pandas 自连接、不要 `groupby.apply`**；
   指针解析用排序 + `np.searchsorted`。目标峰值 ≤ 4 GB。
5. **GPU**：只有一张 RTX 3060 Laptop **6 GB**。
   同一时刻**只跑一个** GPU 作业（显存独占，并发必 OOM）；
   batch/seq 改大之前先算显存。实测 44.7 M 参数 / batch 32 / seq 256 / fp32 峰值 4.13 GiB。
   输出层**不要**做 `batch × seq × 大词表` 的 softmax（单个 logits 张量能吃 1.46 GiB）。
6. **CPU**：20 逻辑核。LightGBM / 特征工程可以吃满；
   但**读同一张大 parquet 的进程不要并行**（I/O 争抢反而更慢）。
7. 随机种子写进 `configs/config.yaml`，切分用**确定性哈希**而非随机洗牌，保证任何人重跑结果一致。

---

## 5. 科学诚实

- **预注册的判据不许事后改**。`ΔAUPRC >= 0.01` 没达到就写没达到，负结果照样是结果。
- 报告里不写「首次」「第一个」。检索不到只能写
  「据我们检索（Crossref + PubMed，检索词 X/Y/Z，检索日期 YYYY-MM-DD）未见」。
- 数字必须来自真跑出来的 `results/metrics/*.json`，**不许凭印象写进报告**。
- 已知的局限（见 `docs/00_项目方案.md` §9）必须写进最终报告，不许藏：
  「撤销」大量是改剂量的**修订**而非「反悔」；`order_provider_id` 是账号不是人；
  出院/转科时有批量清理；本地 `chartevents`/`emar_detail` 不完整。

---

## 6. 多 agent 并行时

- **同时跑的 agent 不得写同一个文件。** 派活前先确认文件归属。
- 派活前先问三句：这活能不能用一个确定性脚本做掉？能不能并进已有的交付物？
  它的产出会不会和别人的重复？**三句都否，才值得单开一个 agent。**
- agent 之间通过**文件**交接，不要靠转述（转述会漂）。
