# -*- coding: utf-8 -*-
"""全工程公用工具：路径、配置、流式读 zip、时间转换、隐私抑制、日志与实验记录。

设计原则
--------
1. **自包含**：本目录可以整个拷走 / 发 GitHub，不依赖 H:\\AIAUTORESEARCH 下的其它工程。
   唯一的外部依赖是 MIMIC-IV v3.1 的 zip 本身（路径见 config.yaml，可被环境变量覆盖）。
2. **不解压**：9.4 GB 的 zip 一律流式读，不落地 CSV。
3. **隐私红线**：本文件提供 sup()（小格子抑制）与 assert_no_pii()（输出前自检）。
   患者级的行、标识符取值、时间戳取值，一律不得进入任何 JSON / Markdown / 日志。
   允许输出的只有：schema、计数、队列流、聚合统计；n < 10 的格子一律写成 "<10"。
"""
from __future__ import annotations

import io
import json
import os
import platform
import re
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# --------------------------------------------------------------------------- 路径
ROOT = Path(__file__).resolve().parent.parent          # H:\AIAUTORESEARCH\Project
DATA = Path(os.environ.get("BN5212_DATA", ROOT / "data"))       # 患者级中间产物，永不进 git
RESULTS = ROOT / "results"
METRICS = RESULTS / "metrics"
FIGURES = RESULTS / "figures"
LOGS = ROOT / "logs"
DOCS = ROOT / "docs"

for _p in (DATA, RESULTS, METRICS, FIGURES, LOGS):
    _p.mkdir(parents=True, exist_ok=True)

_README = DATA / "README.txt"
if not _README.exists():
    _README.write_text(
        "MIMIC-IV v3.1 派生的患者级中间产物。\n"
        "只允许留在本机：不得进 git、不得上传云端、不得进任何 LLM prompt、不得进幻灯片。\n"
        "本目录已被 .gitignore 排除。\n", encoding="utf-8")

# --------------------------------------------------------------------------- 数据源
MIMIC_ZIP = Path(os.environ.get("MIMIC_IV_ZIP", r"H:\BN5212\MIMIC_IV.zip"))
_MEMBER_ROOT = "MIMIC_IV/physionet.org/files/mimiciv/3.1"

# 本地副本不完整，全工程禁用（实测缺失率见 docs/01_数据契约.md）
FORBIDDEN_TABLES = {"icu.chartevents", "hosp.emar_detail"}

# 表名 -> zip 内成员路径。列名以 MIMIC-IV v3.1 官方 schema 为准，逐字核过。
TABLES = {
    "hosp.poe":        f"{_MEMBER_ROOT}/hosp/poe.csv",
    "hosp.poe_detail": f"{_MEMBER_ROOT}/hosp/poe_detail.csv",
    "hosp.emar":       f"{_MEMBER_ROOT}/hosp/emar.csv",
    "hosp.admissions": f"{_MEMBER_ROOT}/hosp/admissions.csv",
    "hosp.transfers":  f"{_MEMBER_ROOT}/hosp/transfers.csv",
    "hosp.patients":   f"{_MEMBER_ROOT}/hosp/patients.csv",
    "hosp.pharmacy":   f"{_MEMBER_ROOT}/hosp/pharmacy.csv",
    "hosp.services":   f"{_MEMBER_ROOT}/hosp/services.csv",
}


_CFG = None


def cfg() -> dict:
    """读 configs/config.yaml（只读一次）。常数一律放那里，不要散在脚本里。"""
    global _CFG
    if _CFG is None:
        import yaml
        _CFG = yaml.safe_load((ROOT / "configs" / "config.yaml").read_text(encoding="utf-8"))
    return _CFG


def col_np(path: Path, name: str, fill=-1, dictionary: bool = False):
    """从 parquet 里**单独**读一列成 numpy 数组（省内存：不把整表读进来）。

    dictionary=True 时返回 `(codes, names)`：codes 是字典编码的 int32（缺失为 fill=-1），
    names 是词表（list[str]）。高基数字符串列（如 order_provider_id）因此不必把
    字符串本体读进内存；names 只用于给**聚合统计**贴标签，绝不逐行输出。

    ⚠ pyarrow 25 里 ChunkedArray.combine_chunks() 直接返回 Array（不是单块 ChunkedArray），
      所以这里要兼容两种返回类型，不能无脑 .chunk(0)。
    """
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    t = pq.read_table(path, columns=[name],
                      read_dictionary=[name] if dictionary else None)
    a = t[name]
    if not dictionary:
        return pc.fill_null(a, fill).to_numpy(zero_copy_only=False)
    a = a.unify_dictionaries().combine_chunks()
    if hasattr(a, "num_chunks"):                     # 仍是 ChunkedArray
        a = a.chunk(0) if a.num_chunks else a
    names = [str(x) for x in a.dictionary]
    codes = pc.fill_null(a.indices, fill).to_numpy(zero_copy_only=False).astype(np.int32)
    return codes, names


def stream(table: str, usecols=None, chunksize: int = 500_000, dtype=None,
           max_chunks: int | None = None):
    """流式读 zip 内 CSV，逐块 yield DataFrame。不解压、不落地。"""
    if table in FORBIDDEN_TABLES:
        raise RuntimeError(f"{table} 本地副本不完整，全工程禁用（见 docs/01_数据契约.md）")
    if table not in TABLES:
        raise KeyError(f"未登记的表名 {table}；已登记={sorted(TABLES)}")
    if not MIMIC_ZIP.exists():
        raise FileNotFoundError(f"找不到 MIMIC-IV zip：{MIMIC_ZIP}（可用环境变量 MIMIC_IV_ZIP 覆盖）")
    with zipfile.ZipFile(MIMIC_ZIP) as z:
        with z.open(TABLES[table]) as raw:
            wrapper = io.TextIOWrapper(raw, encoding="utf-8", newline="")
            reader = pd.read_csv(wrapper, usecols=usecols, chunksize=chunksize, dtype=dtype)
            for i, chunk in enumerate(reader):
                yield chunk
                if max_chunks and i + 1 >= max_chunks:
                    return


def peek_columns(table: str) -> list[str]:
    """只读表头，用于防幻觉校验列名。"""
    for c in stream(table, chunksize=5, max_chunks=1):
        return list(c.columns)
    return []


# --------------------------------------------------------------------------- 时间
def to_epoch_s(series) -> pd.arrays.IntegerArray:
    """时间字符串 -> epoch 秒（Int64，缺失为 pd.NA）。

    ⚠ 本机踩过的坑：datetime64[us] 直接 astype('int64') 会把 13 小时算成 47 秒。
    这里显式走 datetime64[ns] 再 //10**9，并用 isnat 掩掉 NaT。
    """
    dt = pd.to_datetime(series, errors="coerce")
    arr = np.asarray(dt.to_numpy(dtype="datetime64[ns]"))
    na = np.isnat(arr)
    iv = arr.view("int64") // 10 ** 9
    out = pd.array(iv, dtype="Int64")
    out[na] = pd.NA
    return out


# --------------------------------------------------------------------------- 隐私
def sup(n, k: int = 10):
    """小格子抑制：1 <= n < k 一律写成 '<k'。0 与 >=k 原样返回。"""
    n = int(n)
    return f"<{k}" if 0 < n < k else n


def supdict(d: dict, k: int = 10) -> dict:
    return {kk: sup(vv, k) for kk, vv in d.items()}


def pct(a, b, nd: int = 2):
    return round(100.0 * float(a) / float(b), nd) if b else None


# 只匹配「一定是取值、不可能是聚合量」的东西：
#   - 绝对时间戳（MIMIC 的 charttime/ordertime 形如 2180-07-23 14:00:00）
#   - provider_id / caregiver_id（形如 P77FS2：字母 P + 5 位字母数字）
# ⚠ 故意**不**匹配纯数字：合法的聚合计数（如 10655618 条医嘱）也是 8 位数，
#   用位数判 subject_id 会永久误报，规则就会被人关掉——那比没有规则更糟。
#   真正的防线是「只往 metrics 里写聚合量」这条纪律，本函数只是最后一道机械筛。
_PII_PAT = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}[ T][0-9]{2}:[0-9]{2}|\bP[0-9A-Z]{5}\b")


def assert_no_pii(obj, where: str = ""):
    """输出前的自检：拦截疑似绝对时间戳 / provider_id 取值。

    以 "_" 开头的键（如 _env 环境指纹）是本工程自己的元数据，不参与扫描。
    """
    scan = {k: v for k, v in obj.items() if not str(k).startswith("_")} \
        if isinstance(obj, dict) else obj
    s = json.dumps(scan, ensure_ascii=False, default=str)
    hit = _PII_PAT.findall(s)
    if hit:
        raise RuntimeError(f"隐私自检失败 @{where}：疑似标识符/时间戳取值 {hit[:3]} ...")
    return True


# --------------------------------------------------------------------------- 日志与实验记录
class Tee:
    """同时写屏幕与日志文件。"""

    def __init__(self, path: Path):
        self.f = open(path, "a", encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, s):
        self.stdout.write(s)
        self.f.write(s)
        self.f.flush()      # 立即落盘：长任务跑到一半想看进度时，日志必须是实时的

    def flush(self):
        self.stdout.flush()
        self.f.flush()


def start_log(step: str):
    """把本步骤的 stdout 同时落到 logs/<step>.log，返回 (t0, 日志路径)。"""
    p = LOGS / f"{step}.log"
    t = Tee(p)
    sys.stdout = t
    sys.stderr = t          # 报错也要留痕，否则崩了日志里什么都没有
    print("\n" + "=" * 78)
    print(f"[{step}] 开始 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  "
          f"python={platform.python_version()}  host-cores={os.cpu_count()}")
    return time.time(), p


def env_fingerprint() -> dict:
    """环境指纹，写进每份 metrics，保证可复现。"""
    fp = {"python": platform.python_version(), "platform": platform.platform(),
          "cpu_count": os.cpu_count(),
          "utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    for m in ("numpy", "pandas", "pyarrow", "sklearn", "lightgbm", "torch"):
        try:
            fp[m] = __import__(m).__version__
        except Exception:
            fp[m] = None
    try:
        import torch
        if torch.cuda.is_available():
            fp["gpu"] = torch.cuda.get_device_name(0)
            fp["gpu_mem_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, 2)
    except Exception:
        pass
    try:
        fp["git"] = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                   capture_output=True, text=True, timeout=10).stdout.strip() or None
    except Exception:
        fp["git"] = None
    return fp


def save_metrics(name: str, obj: dict, t0: float | None = None):
    """写 results/metrics/<name>.json。写之前过一遍隐私自检。"""
    payload = dict(obj)
    payload["_env"] = env_fingerprint()
    if t0 is not None:
        payload["_elapsed_s"] = round(time.time() - t0, 1)
    assert_no_pii(payload, name)
    p = METRICS / f"{name}.json"
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"→ 指标已写 {p.relative_to(ROOT)}")
    return p


def append_runlog(step: str, summary: str):
    """把一行运行记录追加到 docs/02_实验记录.md（给人看的时间线）。"""
    p = DOCS / "02_实验记录.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        p.write_text("# 实验记录（自动追加，勿手改正文表格）\n\n"
                     "| 时间 | 步骤 | 结果摘要 |\n|---|---|---|\n", encoding="utf-8")
    line = f"| {datetime.now().strftime('%m-%d %H:%M')} | `{step}` | {summary} |\n"
    with open(p, "a", encoding="utf-8") as f:
        f.write(line)
