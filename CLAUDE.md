# CLAUDE.md —— 给 Claude Code 的项目协议

**先完整读 [`AGENTS.md`](AGENTS.md)**：项目是什么、隐私红线、目录结构、写码规矩、科学诚实，
全在那里，本文件不重复。下面只写 Claude Code 在**这台机器上**要额外注意的东西。

---

## 写码前必读的两份文件

1. `AGENTS.md` —— 协议（尤其是 §2 隐私红线）
2. `docs/01_数据契约.md` —— **接口规范，唯一权威**。列名、dtype、13 类编码、泄漏铁律都在里面。

---

## 🔴 本机环境陷阱（都是真踩过的坑）

1. **Git Bash 下禁用 heredoc。** `<<'EOF'` 会被单引号截断，报
   `unexpected EOF while looking for matching`。
2. **禁止在 Bash 的双引号字符串里写反引号。** `python -c "... \`x\` ..."` 里只要出现一对反引号，
   **bash 会当命令替换先执行掉**，内容被悄悄换成空字符串，而且不报错。
   ⇒ 统一规则：**凡是要往文件里写带格式的中文 / Markdown / 代码，
   一律用 Write 工具落成 `.py` 或 `.md`，再 `python <路径>` 执行。不要用 `python -c`。**
3. 脚本开头写 `sys.stdout.reconfigure(encoding="utf-8")`，否则中文在 Windows 控制台乱码。
   （`src/common.py` 已经做了，`import common` 即可。）
4. **PowerShell 没有 `&&` / `||`**（Windows PowerShell 5.1）。要串命令用 `;` 或 `if ($?) { }`。
5. 补丁脚本的**幂等检查串必须是该处独有的一句话**，不要查文件名、不要查新串前缀——
   查文件名会被正文里恰好提到的同名字符串骗过去，补丁静默跳过还打印 ✓。
   **改完 grep 一次确认真写进去了，不要只信脚本的 ✓。**

---

## 硬件与并发

| 资源 | 实况 | 纪律 |
|---|---|---|
| GPU | RTX 3060 Laptop **6 GB**，CUDA 12.8，torch 2.11 | **同一时刻只跑一个 GPU 作业**，并发必 OOM |
| CPU | 20 逻辑核 | LightGBM / 特征工程可吃满；**读同一张大 parquet 的进程不要并行** |
| 内存 | 16 GB 总量，**常态可用只有 ~6 GB** | 52 M 行的表一律 arrow + numpy 向量化，峰值目标 ≤ 4 GB；**不许 pandas 自连接、不许 `groupby.apply`** |
| 磁盘 | H: 盘约 600 GB 空闲 | 中间产物放 `Project/data/`，不要写 C 盘 |

**多 agent 并发上限：最多 3 个 Opus + 6 个 Sonnet。**
派活前先问三句：这活能不能用一个确定性脚本做掉？能不能并进别人已有的交付物？产出会不会和别人重复？
**三句都否，才值得单开一个 agent。空着的槽位不必填满——填一个没想清楚的 agent，
成本是一整份上下文，而它多半在重做别人做过的事。**
同时跑的 agent **不得写同一个文件**，交接一律通过文件，不靠转述。

---

## 数据在哪

- 原始：`H:\BN5212\MIMIC_IV.zip`（9.4 GB，**不解压**，`common.stream()` 流式读）
  - 可用环境变量 `MIMIC_IV_ZIP` 覆盖路径
- 中间产物：`Project/data/*.parquet`（`.gitignore` 排除，但**留在工程目录里**）
- 可选文本：`H:\BN5212\physionet.org.zip`（MIMIC-IV-Note v2.2，主线跑通前不要碰）

**禁用表**：`icu.chartevents`（本地缺 28.3%）、`hosp.emar_detail`（缺 34.4%）——
`common.stream()` 会直接抛异常拦住。

---

## 留痕要求（用户明确要求，不许省）

每个脚本三件套，一件都不能少：

```python
import common as C
t0, _ = C.start_log("s0X_名字")          # → logs/s0X_名字.log
...
C.save_metrics("s0X_名字", 指标字典, t0)   # → results/metrics/s0X_名字.json（自动附环境指纹 + 隐私自检）
C.append_runlog("s0X_名字", "一句话摘要")  # → docs/02_实验记录.md 追加一行
```

**脚本跑完不删。** 临时探查脚本放系统临时目录，不要留在 `src/`；
如果它的结论有用，整理成正式脚本再放进来。

---

## 当前状态

见 `docs/02_实验记录.md`（每次运行自动追加）与 `results/metrics/` 下的 JSON。
