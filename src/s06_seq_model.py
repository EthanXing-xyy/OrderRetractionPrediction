# -*- coding: utf-8 -*-
"""s06_seq_model.py —— 医嘱序列上的 causal Transformer + 离散时间竞争风险头

两个变体，**共用同一个编码器、同样的训练预算**，只换输出头：

    transformer_cr      13 类 softmax 头（竞争风险，主模型）
    transformer_single   1 个 logit 的二分类头（预测 y_disc），预注册对照

数据来自 `s06a_build_seq.py` 写的 `data/seq/`（契约 §5），预测写成契约 §7 的
`results/preds/<模型名>_<split>.parquet`。**本脚本不算最终指标**，评测统一由 `s07_eval.py` 做；
这里只在训练中打印粗略的 val loss / AUPRC 自查。

13 类离散时间竞争风险的似然（本文件的核心正确性）
-------------------------------------------------
契约 §2.2 的类定义：

    class 0          到删失时刻仍未被撤销
    class 1..6       自撤，发生在第 0..5 个时间桶
    class 7..12      他撤，发生在第 0..5 个时间桶

设 softmax 输出 `p[0..12]`，样本的 `censor_bin = c`。

* **事件样本**（`y13 = m >= 1`）：似然 = `p[m]`，损失 = `-log p[m]`（就是普通交叉熵）。
* **删失样本**（`y13 = 0`）：我们只知道「到第 c 个桶结束时还没被撤」，
  但它**完全可能在第 c 个桶之后被撤，只是观察不到**。所以似然是所有与观测相容的类之和：

      L = p[0] + Σ_{b > c} ( p[1+b] + p[7+b] )      损失 = -log L

  把删失样本写成 `-log p[0]` 是**错的**，那等于假装随访无限长，会系统性高估
  「不会被撤」的概率、压低所有撤销概率。`--selftest` 里有专门的单元测试盯这一条。
* `y13 = -1`（padding / 不参与损失）的位置用 mask 剔除，损失分母只数有效位置。

显存（RTX 3060 Laptop 6 GB，同一时刻只跑一个 GPU 作业）
-------------------------------------------------------
* 注意力走 `F.scaled_dot_product_attention(..., is_causal=True)`，
  用官方的 flash / mem-efficient kernel，不显式构造 (L,L) 分数矩阵。
* 默认 bf16 autocast（sm_86 支持）。
* padding 一律在块尾部，加上因果 mask，真实位置**永远看不到 padding**，
  所以不需要 key_padding_mask（也就避开了「整行被 mask 光 → NaN」的坑）。
* 正式训练前先 `--bench`：20 步，打印 `torch.cuda.max_memory_allocated()`。

用法
----
    python src/s06_seq_model.py --selftest                     # 单元测试（不需要数据、不需要 GPU）
    python src/s06_seq_model.py --variant cr --smoke           # 1% 块跑通全流程
    python src/s06_seq_model.py --variant cr --bench           # 量显存
    python src/s06_seq_model.py --variant cr                   # 全量训练 + 出 val/test 预测
    python src/s06_seq_model.py --variant cr --predict-only    # 只用已有 checkpoint 出预测
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import common as C

N_BIN, N_CAUSE, K_CR = 6, 2, 13
VARIANT_NAME = {"cr": "transformer_cr", "single": "transformer_single"}


# =========================================================================== 数据
def load_meta(seqdir: Path) -> dict:
    p = seqdir / "meta.json"
    if not p.exists():
        raise FileNotFoundError(f"缺少 {p}，先跑 python src/s06a_build_seq.py")
    return json.loads(p.read_text(encoding="utf-8"))


class SeqBlocks(Dataset):
    """按块读 data/seq/ 的 .npy（mmap，不把几 GB 读进内存）。

    memmap 在**每个 worker 里惰性打开**（Windows 是 spawn，不能把 mmap 对象 pickle 过去）。
    """

    def __init__(self, seqdir: Path, split: str, idx: np.ndarray | None = None):
        self.seqdir, self.split = Path(seqdir), split
        self.idx = None if idx is None else np.asarray(idx, np.int64)
        self._a = None
        self._n = int(np.load(self.seqdir / f"{split}_y13.npy", mmap_mode="r").shape[0])

    def _open(self):
        self._a = {k: np.load(self.seqdir / f"{self.split}_{k}.npy", mmap_mode="r")
                   for k in ("tokens", "num", "y13", "censorbin")}

    def __len__(self):
        return self._n if self.idx is None else int(self.idx.size)

    def __getitem__(self, i):
        if self._a is None:
            self._open()
        b = int(i if self.idx is None else self.idx[i])
        a = self._a
        # np.array(..., copy) 而不是 ascontiguousarray：后者对只读 mmap 返回视图，
        # torch.from_numpy 会警告「非可写数组」。
        return (torch.from_numpy(np.array(a["tokens"][b], dtype=np.int32)),
                torch.from_numpy(np.array(a["num"][b], dtype=np.float32)),
                torch.from_numpy(np.array(a["y13"][b], dtype=np.int64)),
                torch.from_numpy(np.array(a["censorbin"][b], dtype=np.int64)),
                b)


def make_loader(ds, batch, shuffle, workers, seed=0):
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=workers,
                      pin_memory=torch.cuda.is_available(), drop_last=False, generator=g,
                      persistent_workers=workers > 0, prefetch_factor=4 if workers > 0 else None)


# =========================================================================== 模型
class OrderEmbed(nn.Module):
    """一条医嘱 = 若干类别 embedding 求和 + 数值特征过小 MLP 相加 + 位置 embedding。"""

    def __init__(self, vocabs: list[int], n_num: int, d: int, seq_len: int, dropout: float):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(v, d, padding_idx=0) for v in vocabs])
        for e in self.embs:
            nn.init.normal_(e.weight, 0.0, 0.02)
            with torch.no_grad():
                e.weight[0].zero_()
        self.num_mlp = nn.Sequential(nn.Linear(n_num, d), nn.GELU(), nn.Linear(d, d)) if n_num else None
        self.pos = nn.Embedding(seq_len, d)
        nn.init.normal_(self.pos.weight, 0.0, 0.02)
        self.ln = nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)

    def forward(self, tok, num):
        x = None
        for f, e in enumerate(self.embs):
            v = e(tok[..., f])
            x = v if x is None else x + v
        if self.num_mlp is not None:
            v = self.num_mlp(num)
            x = v if x is None else x + v
        x = x + self.pos.weight[: x.shape[1]].unsqueeze(0)
        return self.drop(self.ln(x))


class CausalBlock(nn.Module):
    """pre-LN Transformer encoder layer，注意力强制 causal（位置 i 只能看到 <= i）。"""

    def __init__(self, d: int, h: int, ff: int, dropout: float):
        super().__init__()
        assert d % h == 0, "d_model 必须能被 n_heads 整除"
        self.h, self.dh, self.p = h, d // h, dropout
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Linear(ff, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        B, L, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(D, dim=-1)
        q = q.view(B, L, self.h, self.dh).transpose(1, 2)
        k = k.view(B, L, self.h, self.dh).transpose(1, 2)
        v = v.view(B, L, self.h, self.dh).transpose(1, 2)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                           dropout_p=self.p if self.training else 0.0)
        x = x + self.drop(self.proj(a.transpose(1, 2).reshape(B, L, D)))
        x = x + self.drop(self.mlp(self.ln2(x)))
        return x


class SeqModel(nn.Module):
    """causal Transformer；每个位置输出该位置那条医嘱的预测（一次前向拿整段住院的预测）。"""

    def __init__(self, vocabs, n_num, seq_len, d=256, n_layers=8, n_heads=8,
                 ff=None, dropout=0.1, n_out=K_CR):
        super().__init__()
        ff = ff or 4 * d
        self.embed = OrderEmbed(vocabs, n_num, d, seq_len, dropout)
        self.blocks = nn.ModuleList([CausalBlock(d, n_heads, ff, dropout) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, n_out)
        nn.init.zeros_(self.head.bias)
        nn.init.normal_(self.head.weight, 0.0, 0.02)

    def forward(self, tok, num):
        x = self.embed(tok.long(), num)
        for b in self.blocks:
            x = b(x)
        return self.head(self.ln_f(x))


# =========================================================================== 损失
def build_allow(device=None) -> torch.Tensor:
    """allow[c] = 「删失在第 c 个桶」时与观测相容的类集合（bool，(6, 13)）。

    相容 = 未撤（class 0）∪ 发生在第 c 个桶**之后**的所有 (cause, bin) 类。
    """
    allow = torch.zeros(N_BIN, K_CR, dtype=torch.bool)
    allow[:, 0] = True
    for c in range(N_BIN):
        for b in range(N_BIN):
            if b > c:
                allow[c, 1 + b] = True
                allow[c, 1 + N_BIN + b] = True
    return allow if device is None else allow.to(device)


def cr_loss(logits: torch.Tensor, y13: torch.Tensor, censor_bin: torch.Tensor,
            allow: torch.Tensor) -> torch.Tensor:
    """离散时间竞争风险的逐样本负对数似然。logits/y13/censor_bin 都已展平成 (M, ...)。

    事件样本 -> -log p[m]；删失样本 -> -log ( p[0] + Σ_{b>c} (p[1+b] + p[7+b]) )。
    """
    logp = F.log_softmax(logits.float(), dim=-1)
    ev = y13 >= 1
    l_event = -logp.gather(1, y13.clamp(min=0).unsqueeze(1)).squeeze(1)
    m = allow[censor_bin.clamp(0, N_BIN - 1)]                      # (M, 13) bool
    l_cens = -torch.logsumexp(logp.masked_fill(~m, float("-inf")), dim=-1)
    return torch.where(ev, l_event, l_cens)


def batch_loss(logits, y13, cb, allow, variant):
    """只在 y13 >= 0 的位置计损失；分母 = 有效位置数。返回 (loss, n_valid)。"""
    m = (y13 >= 0).reshape(-1)
    n = int(m.sum())
    if n == 0:
        return logits.sum() * 0.0, 0
    if variant == "cr":
        lg = logits.reshape(-1, K_CR)[m]
        return cr_loss(lg, y13.reshape(-1)[m], cb.reshape(-1)[m], allow).mean(), n
    lg = logits.reshape(-1)[m].float()
    tgt = (y13.reshape(-1)[m] >= 1).float()
    return F.binary_cross_entropy_with_logits(lg, tgt), n


# =========================================================================== 单元测试
def selftest() -> bool:
    """不需要数据、不需要 GPU。重点盯删失似然与因果性。"""
    torch.manual_seed(0)
    allow = build_allow()
    ok = True

    def chk(name, cond, extra=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'通过' if cond else '失败'}] {name} {extra}")

    # --- allow 矩阵本身
    chk("allow[c=5] 只剩 class 0", allow[5].sum().item() == 1 and bool(allow[5, 0]))
    chk("allow[c=0] = class0 + bin1..5 的自撤他撤 = 11 类", int(allow[0].sum()) == 1 + 2 * 5)

    # --- ① 事件样本 = 普通交叉熵
    lg = torch.randn(7, K_CR)
    y = torch.tensor([1, 5, 6, 7, 12, 3, 9])
    cb = torch.randint(0, 6, (7,))
    chk("事件样本损失 == cross_entropy",
        torch.allclose(cr_loss(lg, y, cb, allow), F.cross_entropy(lg, y, reduction="none"), atol=1e-6))

    # --- ② 删失且 c=5（随访到最后一个桶）→ 退化成 -log p0
    lg = torch.randn(5, K_CR)
    y0 = torch.zeros(5, dtype=torch.long)
    c5 = torch.full((5,), 5, dtype=torch.long)
    chk("删失 c=5 时损失 == -log p0",
        torch.allclose(cr_loss(lg, y0, c5, allow), -F.log_softmax(lg, -1)[:, 0], atol=1e-6))

    # --- ③ 删失且 c<5 时**严格小于** -log p0（证明没写成 -log p0）
    c2 = torch.full((5,), 2, dtype=torch.long)
    l2 = cr_loss(lg, y0, c2, allow)
    chk("删失 c=2 时损失 < -log p0（未来桶的质量确实被算进似然）",
        bool((l2 < -F.log_softmax(lg, -1)[:, 0] - 1e-6).all()))

    # --- ④ 题面要求的方向性测试：censor_bin=2，抬高 bin 3/4/5 的 logit → 损失必须下降
    base_lg = torch.zeros(1, K_CR)
    y_c = torch.zeros(1, dtype=torch.long)
    c_c = torch.tensor([2])
    base = cr_loss(base_lg, y_c, c_c, allow).item()
    fut = [1 + 3, 1 + 4, 1 + 5, 7 + 3, 7 + 4, 7 + 5]          # bin 3/4/5 的自撤 + 他撤
    past = [1 + 0, 1 + 1, 1 + 2, 7 + 0, 7 + 1, 7 + 2]         # bin 0/1/2（与「到 c 还没撤」矛盾）
    up = base_lg.clone()
    up[0, fut] += 1.0
    dn = base_lg.clone()
    dn[0, past] += 1.0
    chk("censor_bin=2：抬高 bin 3/4/5 的 logit → 损失下降",
        cr_loss(up, y_c, c_c, allow).item() < base - 1e-6,
        f"{base:.4f} -> {cr_loss(up, y_c, c_c, allow).item():.4f}")
    chk("censor_bin=2：抬高 bin 0/1/2 的 logit → 损失上升",
        cr_loss(dn, y_c, c_c, allow).item() > base + 1e-6,
        f"{base:.4f} -> {cr_loss(dn, y_c, c_c, allow).item():.4f}")

    # --- ⑤ 同一条的梯度版本：未来桶 logit 的梯度必须为负，过去桶为正
    g_lg = torch.zeros(1, K_CR, requires_grad=True)
    cr_loss(g_lg, y_c, c_c, allow).sum().backward()
    g = g_lg.grad[0]
    chk("未来桶 logit 的梯度 < 0（上调它会降损失）", bool((g[fut] < 0).all()))
    chk("过去桶 logit 的梯度 > 0", bool((g[past] > 0).all()))

    # --- ⑥ mask：y13=-1 的位置不进损失
    lg = torch.randn(2, 4, K_CR)
    y = torch.tensor([[0, -1, 3, -1], [-1, -1, 7, 0]])
    cb = torch.tensor([[2, 0, 1, 0], [0, 0, 4, 5]])
    l_all, n_all = batch_loss(lg, y, cb, allow, "cr")
    lg2 = lg.clone()
    lg2[y < 0] = torch.randn_like(lg2[y < 0]) * 10          # 只改无效位置
    l_2, n_2 = batch_loss(lg2, y, cb, allow, "cr")
    chk("padding / y13=-1 不进损失", n_all == 4 and n_2 == 4 and abs(l_all.item() - l_2.item()) < 1e-6)

    # --- ⑦ 因果性：改动位置 i 之后的输入，位置 <= i 的输出必须不变
    vocabs = [7, 5]
    mdl = SeqModel(vocabs, 3, seq_len=16, d=32, n_layers=2, n_heads=4, ff=64, dropout=0.0).eval()
    tok = torch.randint(1, 5, (2, 16, 2))
    num = torch.randn(2, 16, 3)
    with torch.no_grad():
        o1 = mdl(tok, num)
        tok2, num2 = tok.clone(), num.clone()
        tok2[:, 9:] = torch.randint(1, 5, (2, 7, 2))
        num2[:, 9:] = torch.randn(2, 7, 3)
        o2 = mdl(tok2, num2)
    chk("causal mask：位置 <= 8 的输出不受后面 token 影响",
        torch.allclose(o1[:, :9], o2[:, :9], atol=1e-5),
        f"max|Δ|={float((o1[:, :9] - o2[:, :9]).abs().max()):.2e}")
    chk("对照：位置 >= 9 的输出确实变了", float((o1[:, 9:] - o2[:, 9:]).abs().max()) > 1e-4)

    # --- ⑧ padding token 走 padding_idx=0，embedding 为零向量
    chk("Embedding padding_idx=0 是零向量",
        all(float(e.weight[0].detach().abs().sum()) == 0.0 for e in mdl.embed.embs))

    # --- ⑨ 契约 §7 的换算关系
    p = torch.softmax(torch.randn(11, K_CR), -1).numpy()
    cols = p13_to_contract(p)
    chk("p13 和为 1", np.allclose(p.sum(1), 1.0, atol=1e-3))
    chk("p_disc == 1 - p13_0", np.allclose(cols["p_disc"], 1 - p[:, 0], atol=1e-6))
    chk("p_self + p_other == p_disc",
        np.allclose(cols["p_self"] + cols["p_other"], cols["p_disc"], atol=1e-5))
    chk("p_disc_24h <= p_disc", bool((cols["p_disc_24h"] <= cols["p_disc"] + 1e-6).all()))

    print(f"\n单元测试：{'全部通过' if ok else '有失败项'}")
    return ok


# =========================================================================== 预测换算
def train_base_rates(seqdir: Path) -> dict:
    """从 **train** 的 y13 统计固定拆分比例，给单头基线补契约 §7 的另外三列。

    口径与 `s05_baseline.py` 的 `lgbm_single` 完全一致：单头只训了 y_disc，
    `p_self / p_other / p_disc_24h` 由**训练集**的固定比例拆分而来（不是独立训练的头），
    排序与 `p_disc` 完全一致，只为满足契约 §7 的列齐全。
    """
    y = np.load(seqdir / "train_y13.npy", mmap_mode="r").reshape(-1)
    cnt = np.zeros(K_CR, np.int64)
    for i in range(0, y.size, 20_000_000):
        b = np.asarray(y[i:i + 20_000_000])
        cnt += np.bincount(b[b >= 0], minlength=K_CR)
    n_self, n_other = int(cnt[1:1 + N_BIN].sum()), int(cnt[1 + N_BIN:].sum())
    n_disc = max(1, n_self + n_other)
    n24 = int(cnt[[1, 2, 3]].sum() + cnt[[7, 8, 9]].sum())
    r = {"self": n_self / n_disc, "other": n_other / n_disc, "r24": n24 / n_disc,
         "train_disc_rate": n_disc / max(1, int(cnt.sum()))}
    print(f"  训练集固定拆分比例（单头基线用）：自撤 {r['self']:.4f} / 他撤 {r['other']:.4f} / "
          f"24h 内 {r['r24']:.4f}；训练集撤销率 {r['train_disc_rate']:.4f}")
    return r


def p13_to_contract(p: np.ndarray) -> dict:
    """13 类概率 → 契约 §7 的派生列。p: (M, 13) float32。"""
    return {
        "p_disc": (1.0 - p[:, 0]).astype(np.float32),
        "p_self": p[:, 1:1 + N_BIN].sum(1).astype(np.float32),
        "p_other": p[:, 1 + N_BIN:1 + 2 * N_BIN].sum(1).astype(np.float32),
        # 桶 0/1/2 = <=1h、1-6h、6-24h
        "p_disc_24h": (p[:, [1, 2, 3]].sum(1) + p[:, [7, 8, 9]].sum(1)).astype(np.float32),
    }


# =========================================================================== 构建 / 训练 / 推理
def build_model(meta: dict, variant: str, tcfg: dict) -> SeqModel:
    return SeqModel(
        vocabs=[int(v) for v in meta["cat_vocab_list"]],
        n_num=int(meta["n_num_field"]),
        seq_len=int(meta["seq_len"]),
        d=int(tcfg.get("d_model", 256)),
        n_layers=int(tcfg.get("n_layers", 8)),
        n_heads=int(tcfg.get("n_heads", 8)),
        ff=int(tcfg["d_ff"]) if tcfg.get("d_ff") else None,
        dropout=float(tcfg.get("dropout", 0.1)),
        n_out=K_CR if variant == "cr" else 1,
    )


def lr_lambda(step, total, warmup, floor=0.1):
    if step < warmup:
        return (step + 1) / max(1, warmup)
    t = (step - warmup) / max(1, total - warmup)
    return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(1.0, t)))


@torch.no_grad()
def evaluate(model, loader, allow, variant, device, amp_dt, max_batch=None):
    """粗略自查：val 平均损失 + AUPRC（**不是**最终指标，最终指标由 s07_eval.py 统一算）。"""
    model.eval()
    tot, cnt = 0.0, 0
    ps, ys = [], []
    for i, (tok, num, y, cb, _) in enumerate(loader):
        if max_batch and i >= max_batch:
            break
        tok, num = tok.to(device, non_blocking=True), num.to(device, non_blocking=True)
        y, cb = y.to(device, non_blocking=True), cb.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=amp_dt, enabled=(device.type == "cuda")):
            logits = model(tok, num)
        loss, n = batch_loss(logits.float(), y, cb, allow, variant)
        if n:
            tot += float(loss) * n
            cnt += n
            m = (y >= 0).reshape(-1)
            if variant == "cr":
                pd = 1.0 - torch.softmax(logits.float().reshape(-1, K_CR)[m], -1)[:, 0]
            else:
                pd = torch.sigmoid(logits.float().reshape(-1)[m])
            ps.append(pd.cpu().numpy())
            ys.append((y.reshape(-1)[m] >= 1).cpu().numpy())
    model.train()
    out = {"loss": tot / cnt if cnt else float("nan"), "n": cnt}
    if ps:
        p, yy = np.concatenate(ps), np.concatenate(ys)
        try:
            from sklearn.metrics import average_precision_score, roc_auc_score
            if 0 < yy.sum() < yy.size:
                out["auprc_rough"] = float(average_precision_score(yy, p))
                out["auroc_rough"] = float(roc_auc_score(yy, p))
        except Exception as e:                                   # 自查失败不该拖垮训练
            print(f"  （AUPRC 自查跳过：{e}）")
        out["pos_rate"] = float(yy.mean())
    return out


def run_bench(model, loader, allow, variant, device, amp_dt, opt, steps=20):
    """只跑 steps 步，量显存峰值与吞吐。确认 < 5.0 GiB 再开正式训练。"""
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    model.train()
    t0, done = time.time(), 0
    for tok, num, y, cb, _ in loader:
        tok, num = tok.to(device, non_blocking=True), num.to(device, non_blocking=True)
        y, cb = y.to(device, non_blocking=True), cb.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=amp_dt, enabled=(device.type == "cuda")):
            logits = model(tok, num)
        loss, n = batch_loss(logits.float(), y, cb, allow, variant)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        done += 1
        if done == 3 and device.type == "cuda":                 # 前几步含 cudnn 预热，从第 3 步开始计时
            torch.cuda.synchronize()
            t0 = time.time()
        if done >= steps:
            break
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = max(1e-9, time.time() - t0)
    sps = (done - 2) / dt if done > 2 else done / dt
    out = {"steps": done, "step_per_s": round(sps, 3)}
    if device.type == "cuda":
        out["max_mem_alloc_gib"] = round(torch.cuda.max_memory_allocated() / 1024 ** 3, 3)
        out["max_mem_reserved_gib"] = round(torch.cuda.max_memory_reserved() / 1024 ** 3, 3)
    print(f"  [bench] {out}")
    return out


@torch.no_grad()
def predict(model, meta, seqdir: Path, split: str, variant: str, device, amp_dt,
            batch: int, workers: int, in_cohort_of_key, outdir: Path, smoke_idx=None,
            base_ratio=None):
    """出契约 §7 的预测文件。只对 in_cohort==True 且非 padding 的位置写。"""
    model.eval()
    ds = SeqBlocks(seqdir, split, smoke_idx)
    dl = make_loader(ds, batch, False, workers)
    keys_mm = np.load(seqdir / f"{split}_key.npy", mmap_mode="r")
    cap = int(meta["splits"][split]["n_row"]) if smoke_idx is None else int(len(ds)) * int(meta["seq_len"])
    kk = np.empty(cap, np.int64)
    K = K_CR if variant == "cr" else 1
    pp = np.empty((cap, K), np.float32)
    cur = 0
    for tok, num, _, _, bidx in dl:
        tok, num = tok.to(device, non_blocking=True), num.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=amp_dt, enabled=(device.type == "cuda")):
            logits = model(tok, num)
        logits = logits.float()
        p = (torch.softmax(logits, -1) if variant == "cr" else torch.sigmoid(logits)).cpu().numpy()
        kb = np.asarray(keys_mm[bidx.numpy()])                  # (B, L)
        v = kb >= 0
        nv = int(v.sum())
        kk[cur:cur + nv] = kb[v]
        pp[cur:cur + nv] = p[v]
        cur += nv
    kk, pp = kk[:cur], pp[:cur]
    print(f"  {split}: 非 padding 位置 {cur}")

    keep = in_cohort_of_key(kk)
    kk, pp = kk[keep], pp[keep]
    print(f"  {split}: in_cohort 位置 {kk.size}（契约 §7 只在 in_cohort 上出预测）")

    cols = {"key": kk, "subject_id": (kk >> 24).astype(np.int64)}
    if variant == "cr":
        cols.update(p13_to_contract(pp))
        for j in range(K_CR):
            cols[f"p13_{j}"] = pp[:, j].astype(np.float32)
    else:
        # 单头只有 p_disc。契约 §7 说基线可以留空，但**「留空」= 整列不写**，不是填 NaN：
        #   ① s05_baseline.write_preds 对 p13 就是 `if p13 is not None` 整组省略；
        #   ② s07_eval.check_pred_contract 把必需列里的 NaN 直接判为违约（n_p_self_nan）。
        # 所以这里照 lgbm_single 的既定做法：p13_* 整组不写，另外三列用训练集固定比例拆分。
        pd_ = pp[:, 0].astype(np.float32)
        cols["p_disc"] = pd_
        cols["p_self"] = np.clip(pd_ * base_ratio["self"], 0, 1).astype(np.float32)
        cols["p_other"] = np.clip(pd_ * base_ratio["other"], 0, 1).astype(np.float32)
        cols["p_disc_24h"] = np.clip(pd_ * base_ratio["r24"], 0, 1).astype(np.float32)

    import pyarrow as pa
    import pyarrow.parquet as pq
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f"{VARIANT_NAME[variant]}_{split}.parquet"
    pq.write_table(pa.table(cols), out, compression="zstd")
    print(f"→ 预测已写 {out}（{kk.size} 行）")
    try:
        rel = str(out.relative_to(C.ROOT))
    except ValueError:
        rel = out.name
    return {"n_pred": int(kk.size), "path": rel}


def make_in_cohort_lookup(smoke: bool):
    """返回一个函数：给一批 key，返回它们在 labels.parquet 里的 in_cohort 布尔。"""
    import pyarrow.compute as pcc
    import pyarrow.parquet as pq
    p = C.DATA / "labels.parquet"
    lk = pq.read_table(p, columns=["key"])["key"].to_numpy(zero_copy_only=False).astype(np.int64)
    ic = pcc.fill_null(pq.read_table(p, columns=["in_cohort"])["in_cohort"], False) \
        .to_numpy(zero_copy_only=False).astype(bool)
    o = np.argsort(lk, kind="stable")
    lk, ic = lk[o], ic[o]
    del o

    def f(keys: np.ndarray) -> np.ndarray:
        if keys.size == 0:
            return np.zeros(0, bool)
        pos = np.clip(np.searchsorted(lk, keys), 0, lk.size - 1)
        return (lk[pos] == keys) & ic[pos]

    return f


def main():
    ap = argparse.ArgumentParser(description="causal Transformer + 13 类离散时间竞争风险头")
    ap.add_argument("--variant", choices=["cr", "single"], default="cr")
    ap.add_argument("--selftest", action="store_true", help="只跑单元测试")
    ap.add_argument("--smoke", action="store_true", help="抽 1% 的块、1 个 epoch，把整条路跑通")
    ap.add_argument("--bench", action="store_true", help="只跑 20 步，量显存峰值")
    ap.add_argument("--predict-only", action="store_true", help="载入 checkpoint 只出预测")
    ap.add_argument("--resume", action="store_true", help="从 checkpoints/<name>_last.pt 续训")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seq-dir", default=None)
    ap.add_argument("--pred-dir", default=None, help="预测输出目录，默认 results/preds（合成数据自测时改掉）")
    ap.add_argument("--predict-splits", default="val,test")
    ap.add_argument("--val-batches", type=int, default=400, help="每个 epoch 末 val 自查用多少个 batch")
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--tag", default="")
    ap.add_argument("--no-metrics", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        print("=" * 60)
        print("s06_seq_model 单元测试（删失似然 / 因果性 / 契约换算）")
        print("=" * 60)
        raise SystemExit(0 if selftest() else 1)

    name = VARIANT_NAME[args.variant]
    step = f"s06_seq_model_{args.variant}" + (("_" + args.tag) if args.tag else "")
    t0, _ = C.start_log(step)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    seed = int(C.cfg().get("seed", 0))
    torch.manual_seed(seed)
    np.random.seed(seed % (2 ** 31))

    tcfg = dict(C.cfg()["transformer"])
    seqdir = Path(args.seq_dir) if args.seq_dir else (C.DATA / "seq")
    meta = load_meta(seqdir)
    batch = args.batch or int(tcfg.get("batch_size", 32))
    epochs = args.epochs or int(tcfg.get("epochs", 3))
    lr = args.lr or float(tcfg.get("lr", 3e-4))
    clip = float(tcfg.get("grad_clip", 1.0))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dt = torch.bfloat16 if str(tcfg.get("amp", "bf16")).lower() == "bf16" else torch.float16
    if args.smoke:
        epochs = 1

    print(f"变体={name}  device={device}  batch={batch}  epochs={epochs}  lr={lr}  amp={amp_dt}")
    print(f"seq_len={meta['seq_len']}  n_cat={meta['n_cat_field']}  n_num={meta['n_num_field']}  "
          f"词表合计={sum(meta['cat_vocab_list'])}")

    model = build_model(meta, args.variant, tcfg).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"参数量 {n_par / 1e6:.2f} M（其中 embedding "
          f"{sum(p.numel() for p in model.embed.parameters()) / 1e6:.2f} M）")
    allow = build_allow(device)

    ckdir = C.ROOT / "checkpoints"
    ckdir.mkdir(exist_ok=True)
    cklast = ckdir / f"{name}{('_' + args.tag) if args.tag else ''}_last.pt"

    # ---------------------------------------------------------------- 数据
    nb_train = int(meta["splits"]["train"]["n_block"])
    tr_idx = None
    if args.smoke:
        rng = np.random.default_rng(seed)
        tr_idx = rng.choice(nb_train, max(8, int(0.01 * nb_train)), replace=False)
        tr_idx.sort()
    ds_tr = SeqBlocks(seqdir, "train", tr_idx)
    dl_tr = make_loader(ds_tr, batch, True, args.workers, seed)
    has_val = "val" in meta["splits"]
    dl_va = None
    if has_val:
        va_idx = None
        if args.smoke:
            nv = int(meta["splits"]["val"]["n_block"])
            va_idx = np.sort(np.random.default_rng(seed + 1).choice(nv, max(8, int(0.02 * nv)), False))
        dl_va = make_loader(SeqBlocks(seqdir, "val", va_idx), batch, False, args.workers)
    print(f"train 块 {len(ds_tr)}（全量 {nb_train}）  每 epoch {math.ceil(len(ds_tr) / batch)} 步")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.95))

    # ---------------------------------------------------------------- bench
    rec = {"variant": name, "n_params_m": round(n_par / 1e6, 2), "batch": batch,
           "seq_len": meta["seq_len"], "epochs": epochs, "lr": lr,
           "n_cat_field": meta["n_cat_field"], "n_num_field": meta["n_num_field"],
           "n_block_train": len(ds_tr), "amp": str(amp_dt).replace("torch.", "")}
    if args.bench:
        rec["bench"] = run_bench(model, dl_tr, allow, args.variant, device, amp_dt, opt)
        g = rec["bench"].get("max_mem_alloc_gib")
        print(f"[bench] 显存峰值 {g} GiB —— {'低于 5.0 GiB，可以开全量' if (g or 9) < 5.0 else '⚠ 超过 5.0 GiB，降 batch 或开梯度检查点'}")
        if not args.no_metrics:
            C.save_metrics(step + "_bench", rec, t0)
        return

    # ---------------------------------------------------------------- 训练
    total_steps = max(1, math.ceil(len(ds_tr) / batch) * epochs)
    warmup = min(500, max(10, int(0.03 * total_steps)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_lambda(s, total_steps, warmup))
    start_ep, gstep = 0, 0
    if (args.resume or args.predict_only) and cklast.exists():
        ck = torch.load(cklast, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        if args.resume:
            opt.load_state_dict(ck["opt"])
            sched.load_state_dict(ck["sched"])
            start_ep, gstep = int(ck["epoch"]) + 1, int(ck["step"])
            print(f"从 {cklast.name} 续训：下一个 epoch = {start_ep}")
        else:
            print(f"载入 {cklast.name}（epoch {ck['epoch']}），只出预测")
    elif args.predict_only:
        raise FileNotFoundError(f"找不到 {cklast}，没有可用的 checkpoint")

    hist = []
    if not args.predict_only:
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        model.train()
        for ep in range(start_ep, epochs):
            te, run, runn, seen = time.time(), 0.0, 0, 0
            for it, (tok, num, y, cb, _) in enumerate(dl_tr):
                tok, num = tok.to(device, non_blocking=True), num.to(device, non_blocking=True)
                y, cb = y.to(device, non_blocking=True), cb.to(device, non_blocking=True)
                with torch.autocast("cuda", dtype=amp_dt, enabled=(device.type == "cuda")):
                    logits = model(tok, num)
                loss, n = batch_loss(logits.float(), y, cb, allow, args.variant)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                opt.step()
                opt.zero_grad(set_to_none=True)
                sched.step()
                gstep += 1
                seen += 1
                run += float(loss.detach()) * n
                runn += n
                if seen % args.log_every == 0:
                    el = time.time() - te
                    print(f"  ep{ep} step {seen}/{math.ceil(len(ds_tr) / batch)}  "
                          f"loss={run / max(1, runn):.5f}  lr={sched.get_last_lr()[0]:.2e}  "
                          f"{seen / el:.2f} step/s  已用 {el / 60:.1f} min")
                    run, runn = 0.0, 0
            ck = {"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                  "epoch": ep, "step": gstep, "variant": args.variant,
                  "meta_fields": {"cat": meta["cat_fields"], "num": meta["num_fields"]}}
            torch.save(ck, cklast)
            torch.save(ck, ckdir / f"{name}{('_' + args.tag) if args.tag else ''}_ep{ep}.pt")
            ev = evaluate(model, dl_va, allow, args.variant, device, amp_dt,
                          args.val_batches) if dl_va else {}
            hist.append({"epoch": ep, "train_loss": None, "elapsed_min": round((time.time() - te) / 60, 1),
                         **{f"val_{k}": v for k, v in ev.items()}})
            print(f"[epoch {ep}] 用时 {(time.time() - te) / 60:.1f} min  val={ev}")
        if device.type == "cuda":
            rec["train_max_mem_alloc_gib"] = round(torch.cuda.max_memory_allocated() / 1024 ** 3, 3)
    rec["history"] = hist

    # ---------------------------------------------------------------- 推理
    print("[预测] 读 labels.in_cohort 建查表 …")
    lookup = make_in_cohort_lookup(args.smoke)
    base_ratio = train_base_rates(seqdir) if args.variant == "single" else None
    rec["base_ratio_single"] = base_ratio
    preds = {}
    for sp in [s.strip() for s in args.predict_splits.split(",") if s.strip()]:
        if sp not in meta["splits"]:
            print(f"  跳过 {sp}（seq 里没有）")
            continue
        idx = None
        if args.smoke:
            nb = int(meta["splits"][sp]["n_block"])
            idx = np.sort(np.random.default_rng(seed + 2).choice(nb, max(8, int(0.02 * nb)), False))
        preds[sp] = predict(model, meta, seqdir, sp, args.variant, device, amp_dt, batch,
                            args.workers, lookup,
                            Path(args.pred_dir) if args.pred_dir else (C.RESULTS / "preds"),
                            idx, base_ratio)
    rec["preds"] = preds
    rec["smoke"] = bool(args.smoke)

    if not args.no_metrics:
        C.save_metrics(step, rec, t0)
        C.append_runlog(step, f"{name}：参数 {rec['n_params_m']}M，"
                              f"{'smoke' if args.smoke else f'{epochs} epoch'}，"
                              + "；".join(f"{k} 预测 {v['n_pred']} 行" for k, v in preds.items()))
    print(f"[完成] 用时 {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
