"""WEAVE-v4 数据层：只读真实作品 latent，产出 train/val 划分与诊断索引。

三条原则：
1. 每张真实作品自身就是监督样本，不做跨风格内容近似配对；
2. 划分按作者分组（同作者不同作品尽量不跨 train/val），近重复组强制并入同一侧；
3. 跨风格近邻索引只用于诊断与后续条件训练，绝不作为 GT。

style id 顺序与评测端严格一致：``sorted(data/test)`` = ``sorted(packed/*.pt)``。
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

STYLE_NAMES = (
    "Early_Renaissance",
    "Impressionism",
    "Minimalism",
    "Rococo",
    "Ukiyo_e",
)

def parse_artist(filename: str) -> str:
    """``Style__artist_slug_title.pt`` -> ``artist_slug``；解析失败返回 ``unknown``。

    风格名本身可能带下划线（``Ukiyo_e`` / ``Early_Renaissance``），所以必须按 ``__`` 切分。
    """
    name = Path(filename).name
    if "__" not in name:
        return "unknown"
    rest = name.split("__", 1)[1]
    if "_" not in rest:
        return "unknown"
    return rest.split("_")[0] or "unknown"


@dataclass
class LatentBank:
    """全部真实作品 latent 与元数据。``latents`` 为 [N,4,64,64] float32。"""

    latents: torch.Tensor
    style_ids: torch.Tensor
    files: list[str]
    artists: list[str]

    def __len__(self) -> int:
        return int(self.latents.shape[0])

    @property
    def num_styles(self) -> int:
        return int(self.style_ids.max().item()) + 1

    def subset(self, idx: torch.Tensor | np.ndarray) -> dict[str, torch.Tensor]:
        idx = torch.as_tensor(np.asarray(idx), dtype=torch.long)
        return {"latents": self.latents.index_select(0, idx),
                "style_ids": self.style_ids.index_select(0, idx)}


def load_bank(packed_dir: str | Path) -> LatentBank:
    root = Path(packed_dir)
    files = sorted(root.glob("*.pt"))
    if not files:
        raise FileNotFoundError(f"no packed latents under {root}")
    names: list[str] = []
    latents: list[torch.Tensor] = []
    style_ids: list[torch.Tensor] = []
    for sid, path in enumerate(files):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or "latents" not in payload:
            raise ValueError(f"{path.name}: packed payload must be a dict with 'latents'")
        z = payload["latents"].float().contiguous()
        if z.dim() != 4 or tuple(z.shape[1:]) != (4, 64, 64):
            raise ValueError(f"{path.name}: expect [N,4,64,64], got {tuple(z.shape)}")
        latents.append(z)
        style_ids.append(torch.full((z.shape[0],), sid, dtype=torch.long))
        names.extend(_basenames(payload, z.shape[0]))
    bank = LatentBank(
        latents=torch.cat(latents, 0),
        style_ids=torch.cat(style_ids, 0),
        files=names,
        artists=[parse_artist(n) for n in names],
    )
    if len(files) != len(STYLE_NAMES):
        raise ValueError(f"expect 5 style packs, got {len(files)}")
    return bank


def _basenames(payload: dict, count: int) -> list[str]:
    raw = payload.get("files")
    if isinstance(raw, (list, tuple)) and len(raw) == count:
        return [Path(str(x)).name for x in raw]
    subdir = payload.get("subdir")
    if isinstance(subdir, (list, tuple)) and len(subdir) == count:
        return [Path(str(x)).name for x in subdir]
    raise ValueError("packed payload has no per-sample file list; cannot align metadata")


# ---------------------------------------------------------------------------
# 描述子 / 诊断索引
# ---------------------------------------------------------------------------
def latent_descriptor(latents: torch.Tensor, pool: int = 8) -> torch.Tensor:
    """粗内容描述子：8x8 平均池化 + 去掉每图通道均值，再做 L2 归一化。

    局限（必须写进报告）：这不是 DINO 语义特征，只覆盖构图/明暗布局，
    对风格仍有一定敏感性。
    """
    x = latents.float()
    x = torch.nn.functional.avg_pool2d(x, kernel_size=x.shape[-1] // pool)
    x = x - x.mean(dim=(2, 3), keepdim=True)
    flat = x.flatten(1)
    return torch.nn.functional.normalize(flat, dim=1)


def duplicate_groups(desc: torch.Tensor, *, threshold: float = 0.995) -> tuple[np.ndarray, int]:
    """并查集式近重复分组（余弦相似度阈值）。返回 (group_id, 重复对数)。"""
    n = desc.shape[0]
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    pairs = 0
    for start in range(0, n, 512):
        chunk = desc[start:start + 512]
        sim = chunk @ desc.T
        sim[:, :start] = -1.0
        for i in range(chunk.shape[0]):
            row = sim[i]
            row[: start + i + 1] = -1.0
            js = torch.nonzero(row > threshold, as_tuple=False).flatten().tolist()
            if js:
                a = find(start + i)
                for j in js:
                    b = find(j)
                    if a != b:
                        parent[b] = a
                    pairs += 1
    roots = [find(i) for i in range(n)]
    remap: dict[int, int] = {}
    group = np.empty(n, dtype=np.int64)
    for i, r in enumerate(roots):
        group[i] = remap.setdefault(r, len(remap))
    return group, pairs


def content_clusters(desc: torch.Tensor, *, k: int = 20, seed: int = 0, iters: int = 25) -> np.ndarray:
    """CPU k-means（球面相似度），标签仅用于均衡采样与泛化诊断。"""
    rng = np.random.default_rng(seed)
    n = desc.shape[0]
    k = min(k, n)
    centroids = desc[torch.from_numpy(rng.choice(n, size=k, replace=False))].clone()
    labels = np.zeros(n, dtype=np.int64)
    for _ in range(iters):
        labels = (desc @ centroids.T).argmax(dim=1).numpy()
        new = torch.zeros_like(centroids)
        counts = torch.zeros(k)
        for c in range(k):
            sel = torch.from_numpy(np.nonzero(labels == c)[0])
            if sel.numel() == 0:
                new[c] = desc[rng.integers(0, n)]
                continue
            new[c] = desc.index_select(0, sel).mean(0)
            counts[c] = sel.numel()
        new = torch.nn.functional.normalize(new, dim=1)
        if torch.allclose(new, centroids, atol=1e-3):
            centroids = new
            break
        centroids = new
    return labels


def cross_style_topk(
    desc: torch.Tensor, style_ids: torch.Tensor, *, k: int = 5, chunk: int = 512,
) -> dict[int, np.ndarray]:
    """每张图在其他风格里的 Top-K 近邻（仅诊断）。返回 {idx: [k] indices}。"""
    n = desc.shape[0]
    out: dict[int, np.ndarray] = {}
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        sim = desc[start:end] @ desc.T
        sid = style_ids[start:end].view(-1, 1)
        sim = sim.masked_fill(style_ids.view(1, -1) == sid, float("-inf"))
        vals, idx = sim.topk(k, dim=1)
        for row in range(end - start):
            out[start + row] = idx[row].numpy()
    return out


# ---------------------------------------------------------------------------
# 划分
# ---------------------------------------------------------------------------
def build_split(
    bank: LatentBank,
    *,
    val_per_style: int,
    seed: int = 0,
    dup_group: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """按作者分组做 holdout；作者过大的风格退化为作品级切分（报告里标明）。

    近重复约束：若 val 样本与 train 样本属于同一 dup 组，则该样本改回 train。
    """
    n = len(bank)
    rng = np.random.default_rng(seed)
    artists = np.asarray(bank.artists, dtype=object)
    style_ids = bank.style_ids.numpy()
    val_mask = np.zeros(n, dtype=bool)
    report: dict = {"styles": {}, "artist_holdout_feasible": {}}

    for sid, name in enumerate(STYLE_NAMES[: bank.num_styles]):
        idx = np.nonzero(style_ids == sid)[0]
        by_artist: dict[str, list[int]] = {}
        for i in idx:
            by_artist.setdefault(artists[i], []).append(int(i))
        order = sorted(by_artist.items(), key=lambda kv: (-len(kv[1]), kv[0]))
        chosen: list[str] = []
        taken = 0
        for artist, works in order:
            if taken >= val_per_style:
                break
            if taken + len(works) > val_per_style * 1.5:
                continue
            chosen.append(artist)
            taken += len(works)
        val_local: list[int] = [i for a in chosen for i in by_artist[a]]
        # 不足预算时按作品补齐 —— 该风格无法做纯作者留出
        deficit = val_per_style - len(val_local)
        partial_artist = False
        if deficit > 0:
            leftover = sorted(set(idx.tolist()) - set(val_local))
            pick = rng.choice(len(leftover), size=min(deficit, len(leftover)), replace=False)
            val_local.extend(int(leftover[j]) for j in pick)
            partial_artist = True
        val_mask[val_local] = True
        report["styles"][name] = {
            "total": int(len(idx)),
            "val": int(len(val_local)),
            "train": int(len(idx) - len(val_local)),
            "artists_total": int(len(by_artist)),
            "val_artists": chosen,
            "artist_holdout": not partial_artist,
            "largest_artist_works": int(len(order[0][1])) if order else 0,
        }
        report["artist_holdout_feasible"][name] = bool(not partial_artist)

    train_mask = ~val_mask

    if dup_group is not None:
        moved = 0
        g_train = set(dup_group[train_mask].tolist())
        for i in np.nonzero(val_mask)[0]:
            if dup_group[i] in g_train:
                val_mask[i] = False
                train_mask[i] = True
                moved += 1
        report["dup_moved_from_val"] = int(moved)
    report["train_total"] = int(train_mask.sum())
    report["val_total"] = int(val_mask.sum())
    return np.nonzero(train_mask)[0], np.nonzero(val_mask)[0], report


def save_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


__all__ = [
    "STYLE_NAMES", "LatentBank", "load_bank", "parse_artist", "latent_descriptor",
    "duplicate_groups", "content_clusters", "cross_style_topk", "build_split", "save_json",
]
