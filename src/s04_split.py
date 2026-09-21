# -*- coding: utf-8 -*-
"""S04 数据切分：按 subject_id 切 80/10/10，外加一组时间外推切分。

为什么不用随机洗牌
------------------
随机洗牌要靠种子 + 库版本 + 平台才能复现，组员换台机器就可能对不上。
这里用 **splitmix64 确定性哈希**：给定同一个 `subject_id` 和同一个盐，
任何机器、任何时间、任何 numpy 版本算出来的都是同一个桶。

为什么按 subject_id 切
----------------------
同一个患者的医嘱高度相关（同一个主治、同一套医嘱模板、同一次住院的上下文）。
按行切会让同一次住院的医嘱同时出现在训练集和测试集里，**指标会虚高**。
按 `subject_id` 切保证同一患者的所有住院、所有医嘱都在同一侧。

输出：data/splits.parquet + results/metrics/s04_split.json
用法： python src/s04_split.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

M64 = np.uint64(0xFFFFFFFFFFFFFFFF)


def splitmix64(x: np.ndarray) -> np.ndarray:
    """确定性 64 位混淆函数（Vigna 的 splitmix64）。纯整数运算，跨平台一致。"""
    z = (x.astype(np.uint64) + np.uint64(0x9E3779B97F4A7C15)) & M64
    z = ((z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)) & M64
    z = ((z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)) & M64
    return (z ^ (z >> np.uint64(31))) & M64


def main():
    t0, _ = C.start_log("s04_split")
    cf = C.cfg()
    sp = cf["split"]
    salt = np.uint64(abs(hash(sp["hash_salt"])) % (1 << 32)) if False else np.uint64(0x424E3532)
    # ↑ 盐写死成常量而不是 hash(字符串)：python 的 str hash 每个进程都不一样，不可复现。

    LAB = C.DATA / "labels.parquet"
    subj = C.col_np(LAB, "subject_id").astype(np.int64)
    in_co = C.col_np(LAB, "in_cohort", fill=False).astype(bool)
    uniq = np.unique(subj)
    print(f"患者数 {len(uniq):,}，医嘱数 {len(subj):,}")

    bucket = (splitmix64(uniq.astype(np.uint64) ^ salt) % np.uint64(100)).astype(np.int64)
    split = np.where(bucket < sp["train_pct"], 0,
                     np.where(bucket < sp["train_pct"] + sp["val_pct"], 1, 2)).astype(np.int8)

    # 时间外推：按 anchor_year_group 训早测晚（MIMIC-IV 的年份是 3 年一档的粗分组）
    Pt = pq.read_table(C.DATA / "patients.parquet", columns=["subject_id", "anchor_year_group"])
    p_s = Pt["subject_id"].to_numpy(zero_copy_only=False).astype(np.int64)
    yg, ygn = C.col_np(C.DATA / "patients.parquet", "anchor_year_group", dictionary=True)
    o = np.argsort(p_s); p_s, yg = p_s[o], yg[o]
    pos = np.clip(np.searchsorted(p_s, uniq), 0, len(p_s) - 1)
    ygu = np.where(p_s[pos] == uniq, yg[pos], -1)
    late = len(ygn) - 1                       # 词表按年份升序，最后一档 = 最晚
    split_time = np.where(ygu == late, 1, np.where(ygu >= 0, 0, -1)).astype(np.int8)
    del Pt, p_s, yg, o, pos

    out = C.DATA / "splits.parquet"
    pq.write_table(pa.table({"subject_id": uniq, "split": split, "split_time": split_time},
                            schema=pa.schema([("subject_id", pa.int64()), ("split", pa.int8()),
                                              ("split_time", pa.int8())])),
                   out, compression="zstd")

    # ---- 统计：患者数与医嘱数都要报，两者比例不同说明有大户患者 ----
    sp_of_row = split[np.searchsorted(uniq, subj)]
    names = {0: "train", 1: "val", 2: "test"}
    S = {"n_subjects": len(uniq), "n_orders": len(subj),
         "salt_hex": hex(int(salt)), "hash": "splitmix64(subject_id ^ salt) % 100",
         "患者数": {names[k]: int((split == k).sum()) for k in (0, 1, 2)},
         "患者占比pct": {names[k]: C.pct((split == k).sum(), len(uniq)) for k in (0, 1, 2)},
         "医嘱数": {names[k]: int((sp_of_row == k).sum()) for k in (0, 1, 2)},
         "医嘱占比pct": {names[k]: C.pct((sp_of_row == k).sum(), len(subj)) for k in (0, 1, 2)},
         "队列内医嘱数": {names[k]: int(((sp_of_row == k) & in_co).sum()) for k in (0, 1, 2)},
         "时间外推": {"年份档数": len(ygn),
                      "早期训练患者": int((split_time == 0).sum()),
                      "晚期测试患者": int((split_time == 1).sum()),
                      "无年份": C.sup(int((split_time == -1).sum()))}}
    C.save_metrics("s04_split", S, t0)
    C.append_runlog("s04_split",
                    f"按患者切 {S['患者占比pct']}，对应医嘱 {S['医嘱占比pct']}")
    print(f"患者占比 {S['患者占比pct']}\n医嘱占比 {S['医嘱占比pct']}\n"
          f"队列内医嘱 {S['队列内医嘱数']}\n→ {out}   耗时 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
