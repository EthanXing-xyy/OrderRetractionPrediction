# -*- coding: utf-8 -*-
"""S01 数据抽取：MIMIC-IV v3.1 zip → 本工程 data/ 下的 parquet。

做什么
------
不解压 9.4 GB 的 zip，流式读所需的 7 张表的**所需列**，做三件事后落 parquet：
  1. 时间列统一成 epoch 秒（Int64），避开 datetime64[us] 的换算陷阱；
  2. `poe_id` 形如 "<subject_id>-<poe_seq>"，本步把两个撤销指针**解析成 poe_seq 整数**
     （并统计前缀 != subject_id 的异常数），下游一律用 (subject_id, poe_seq) 做键，
     52 M 行省掉三列长字符串；
  3. 低基数字符串转 category，parquet 用 zstd + 字典编码。

只打印行数、列名、缺失率等聚合量，不打印任何一条记录。

用法
----
    python src/s01_extract.py                      # 全部表，全量
    python src/s01_extract.py --tables poe,emar    # 指定表
    python src/s01_extract.py --smoke              # 每张表只读 2 块，验证管线通不通
    python src/s01_extract.py --force              # 覆盖已有 parquet
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

# 表 -> (读哪些列, 哪些是时间列, 哪些转 category)
SPEC = {
    "poe": dict(
        table="hosp.poe",
        usecols=["poe_id", "poe_seq", "subject_id", "hadm_id", "ordertime", "order_type",
                 "order_subtype", "transaction_type", "order_status",
                 "discontinue_of_poe_id", "discontinued_by_poe_id", "order_provider_id"],
        times=["ordertime"],
        cats=["order_type", "order_subtype", "transaction_type", "order_status"],
        ptrs=["discontinue_of_poe_id", "discontinued_by_poe_id"],
        ints=["poe_seq", "subject_id", "hadm_id"]),
    "admissions": dict(
        table="hosp.admissions",
        usecols=["subject_id", "hadm_id", "admittime", "dischtime", "deathtime",
                 "admission_type", "admission_location", "discharge_location",
                 "insurance", "race", "hospital_expire_flag", "edregtime", "edouttime"],
        times=["admittime", "dischtime", "deathtime", "edregtime", "edouttime"],
        cats=["admission_type", "admission_location", "discharge_location", "insurance", "race"],
        ptrs=[], ints=["subject_id", "hadm_id", "hospital_expire_flag"]),
    "patients": dict(
        table="hosp.patients",
        usecols=["subject_id", "gender", "anchor_age", "anchor_year", "anchor_year_group"],
        times=[], cats=["gender", "anchor_year_group"], ptrs=[],
        ints=["subject_id", "anchor_age", "anchor_year"]),
    "transfers": dict(
        table="hosp.transfers",
        usecols=["subject_id", "hadm_id", "transfer_id", "eventtype", "careunit",
                 "intime", "outtime"],
        times=["intime", "outtime"], cats=["eventtype", "careunit"], ptrs=[],
        ints=["subject_id", "hadm_id", "transfer_id"]),
    "services": dict(
        table="hosp.services",
        usecols=["subject_id", "hadm_id", "transfertime", "prev_service", "curr_service"],
        times=["transfertime"], cats=["prev_service", "curr_service"], ptrs=[],
        ints=["subject_id", "hadm_id"]),
    "poe_detail": dict(
        table="hosp.poe_detail",
        usecols=["poe_id", "poe_seq", "subject_id", "field_name", "field_value"],
        times=[], cats=["field_name"], ptrs=[], ints=["poe_seq", "subject_id"]),
    "emar": dict(
        table="hosp.emar",
        usecols=["subject_id", "hadm_id", "emar_seq", "poe_id", "charttime",
                 "event_txt", "scheduletime"],
        times=["charttime", "scheduletime"], cats=["event_txt"], ptrs=["poe_id"],
        ints=["subject_id", "hadm_id", "emar_seq"]),
}

STR_READ = {"poe_id", "discontinue_of_poe_id", "discontinued_by_poe_id",
            "order_provider_id", "field_value"}


def parse_ptr(s: pd.Series, subject: pd.Series):
    """'<subject_id>-<seq>' -> (seq:Int32, 前缀与 subject_id 不符的条数)。"""
    sp = s.str.split("-", n=1, expand=True)
    if sp.shape[1] < 2:
        return pd.array(np.full(len(s), pd.NA, dtype=object), dtype="Int32"), 0
    pre = pd.to_numeric(sp[0], errors="coerce")
    seq = pd.to_numeric(sp[1], errors="coerce")
    bad = int(((pre.notna()) & (pre != subject)).sum())
    return seq.astype("Int32"), bad


def run_one(key: str, args) -> dict:
    spec = SPEC[key]
    out = C.DATA / f"{key}.parquet"
    if out.exists() and not args.force and not args.smoke:
        print(f"~ {out.name} 已存在，跳过（--force 可覆盖）")
        return {"skipped": True}
    if args.smoke:
        out = C.DATA / "smoke" / f"{key}.parquet"
        out.parent.mkdir(exist_ok=True)

    t0 = time.time()
    # 防幻觉：列名先对一次表头
    head = C.peek_columns(spec["table"])
    miss = [c for c in spec["usecols"] if c not in head]
    if miss:
        raise KeyError(f"{spec['table']} 实际表头没有列 {miss}；实际={head}")

    dtype = {c: "string" for c in spec["usecols"] if c in STR_READ}
    writer, n, bad_ptr, nulls = None, 0, {p: 0 for p in spec["ptrs"]}, {}
    seq_max = 0
    for chunk in C.stream(spec["table"], usecols=spec["usecols"], chunksize=500_000,
                          dtype=dtype, max_chunks=2 if args.smoke else None):
        for c in spec["times"]:
            chunk[c] = C.to_epoch_s(chunk[c])
        for p in spec["ptrs"]:
            seq, bad = parse_ptr(chunk[p], chunk["subject_id"])
            newname = "poe_seq" if p == "poe_id" else p.replace("_poe_id", "_seq")
            chunk[newname] = seq
            bad_ptr[p] += bad
            chunk.drop(columns=[p], inplace=True)
        for c in spec["ints"]:
            if c in chunk.columns:
                chunk[c] = pd.to_numeric(chunk[c], errors="coerce").astype("Int64")
        if "poe_seq" in chunk.columns:
            m = chunk["poe_seq"].max()
            seq_max = max(seq_max, int(m) if pd.notna(m) else 0)
        for c in spec["cats"]:
            chunk[c] = chunk[c].astype("category")
        for c in chunk.columns:
            nulls[c] = nulls.get(c, 0) + int(chunk[c].isna().sum())

        tbl = pa.Table.from_pandas(chunk, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(out, tbl.schema, compression="zstd", compression_level=3)
        writer.write_table(tbl.cast(writer.schema) if tbl.schema != writer.schema else tbl)
        n += len(chunk)
        if n % 5_000_000 == 0:
            print(f"    {key}: {n:,} 行  {time.time() - t0:.0f}s", flush=True)
    if writer:
        writer.close()

    dt = time.time() - t0
    mb = out.stat().st_size / 1024 ** 2
    rep = {"rows": n, "parquet_mb": round(mb, 1), "seconds": round(dt), "poe_seq_max": seq_max,
           "ptr_prefix_mismatch": bad_ptr,
           "null_pct": {k: C.pct(v, n) for k, v in sorted(nulls.items()) if v}}
    print(f"✓ {key}: {n:,} 行 → {out.name}  {mb:,.0f} MB  {dt:.0f}s  "
          f"指针前缀异常={bad_ptr}")
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", default=",".join(SPEC))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    keys = [k.strip() for k in a.tables.split(",") if k.strip()]
    bad = [k for k in keys if k not in SPEC]
    if bad:
        sys.exit(f"未知表 {bad}；可选={sorted(SPEC)}")

    t0, _ = C.start_log(f"s01_extract_{'_'.join(keys)}{'_smoke' if a.smoke else ''}")
    print(f"源 zip: {C.MIMIC_ZIP}    目标: {C.DATA}")
    rep = {"smoke": a.smoke, "tables": {}}
    for k in keys:
        rep["tables"][k] = run_one(k, a)
    tag = "s01_extract_smoke" if a.smoke else f"s01_extract_{'_'.join(keys)}"
    C.save_metrics(tag, rep, t0)
    done = {k: v.get("rows") for k, v in rep["tables"].items() if not v.get("skipped")}
    C.append_runlog("s01_extract", f"抽取 {done}，耗时 {time.time() - t0:.0f}s")
    print(f"S01 完成，总耗时 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
