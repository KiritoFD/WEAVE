"""WEAVE-VAE V2: Style Reference Bank and Supervised Margin Ranking Loss.

Implements:
1. StyleReferenceBank:
   - Holds 8-16 detached reference descriptors (q1, q2, moments) per style (136 styles, ~1632 entries).
   - Tracks style_id, artist_id, genre_id for every reference.
   - Fast GPU distance calculation against anchor microbatch.
   - Supports progressive or periodic re-encoding refresh.
2. SupervisedStyleRankingLoss:
   - Margin-based softplus ranking loss:
     L_style = softplus((d(a, p) - d(a, n) + m) / tau)
   - Anchor: each image in current microbatch.
   - Positives: closest 2 from 8 positive candidates (same style, prioritized cross-artist).
   - Negatives: 8 semi-hard negatives from 32 same-genre + 32 random negatives.
"""
from __future__ import annotations

import collections
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tools.vae_suite.swd_metric import LatentSWDDescriptor


class StyleReferenceBank(nn.Module):
    """GPU-resident, detached descriptor bank covering all 136 styles."""

    def __init__(
        self,
        num_styles: int = 136,
        refs_per_style: int = 12,
        descriptor: Optional[LatentSWDDescriptor] = None,
    ):
        super().__init__()
        self.num_styles = num_styles
        self.refs_per_style = refs_per_style
        self.total_refs = num_styles * refs_per_style

        # Buffers for descriptors:
        # q1: (total_refs, 32, 16), q2: (total_refs, 32, 16), moments: (total_refs, 14)
        self.register_buffer("bank_q1", torch.zeros(self.total_refs, 32, 16))
        self.register_buffer("bank_q2", torch.zeros(self.total_refs, 32, 16))
        self.register_buffer("bank_mom", torch.zeros(self.total_refs, 14))

        self.register_buffer("bank_styles", torch.full((self.total_refs,), -1, dtype=torch.long))
        self.register_buffer("bank_artists", torch.full((self.total_refs,), -1, dtype=torch.long))
        self.register_buffer("bank_genres", torch.full((self.total_refs,), -1, dtype=torch.long))
        self.register_buffer("bank_image_indices", torch.full((self.total_refs,), -1, dtype=torch.long))

        self.descriptor = descriptor if descriptor is not None else LatentSWDDescriptor()

        # Cache mapping: style_id -> list of indices in bank
        self.style_to_bank_indices: Dict[int, List[int]] = collections.defaultdict(list)

    def populate(
        self,
        vae: nn.Module,
        dataset,
        device: str = "cuda",
        batch_size: int = 16,
    ):
        """Populates the reference bank using VAE encoder across all styles."""
        print(f"[Reference Bank] Populating {self.refs_per_style} refs for {self.num_styles} styles ({self.total_refs} total)...", flush=True)
        vae.eval()

        # Group dataset records by style
        style_to_records = collections.defaultdict(list)
        for i, idx in enumerate(dataset.indices):
            rec = dataset.records_by_idx[idx]
            style_to_records[rec["style_id"]].append((i, rec))

        chosen_records = []
        bank_idx = 0

        rng = random.Random(42)
        for s_id in range(self.num_styles):
            candidates = style_to_records.get(s_id, [])
            if not candidates:
                continue

            # Prioritize distinct artists
            artist_dict = collections.defaultdict(list)
            for c in candidates:
                artist_dict[c[1]["artist_id"]].append(c)

            distinct_picks = []
            artists_shuffled = list(artist_dict.keys())
            rng.shuffle(artists_shuffled)

            # Round-robin pick across artists
            while len(distinct_picks) < self.refs_per_style:
                progress = False
                for a in artists_shuffled:
                    if artist_dict[a]:
                        distinct_picks.append(artist_dict[a].pop(0))
                        progress = True
                        if len(distinct_picks) == self.refs_per_style:
                            break
                if not progress:
                    # Refill from candidates if needed
                    candidates_shuffled = candidates[:]
                    rng.shuffle(candidates_shuffled)
                    while len(distinct_picks) < self.refs_per_style and candidates_shuffled:
                        distinct_picks.append(candidates_shuffled.pop(0))
                    break

            for c_idx, rec in distinct_picks[:self.refs_per_style]:
                chosen_records.append((bank_idx, c_idx, rec))
                self.bank_styles[bank_idx] = rec["style_id"]
                self.bank_artists[bank_idx] = rec.get("artist_id", -1)
                self.bank_genres[bank_idx] = rec.get("genre_id", -1)
                self.bank_image_indices[bank_idx] = c_idx
                self.style_to_bank_indices[rec["style_id"]].append(bank_idx)
                bank_idx += 1

        actual_refs = bank_idx
        print(f"[Reference Bank] Selected {actual_refs} reference images. Extracting descriptors on GPU...", flush=True)

        # Batch encode
        with torch.inference_mode():
            for i in range(0, actual_refs, batch_size):
                chunk = chosen_records[i : i + batch_size]
                tensors = torch.stack([dataset[c[1]]["image"] for c in chunk]).to(device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    mu = vae.encode(tensors).latent_dist.mean.float()
                q1, q2, mom = self.descriptor(mu)

                for j, (b_i, _, _) in enumerate(chunk):
                    self.bank_q1[b_i].copy_(q1[j].detach())
                    self.bank_q2[b_i].copy_(q2[j].detach())
                    self.bank_mom[b_i].copy_(mom[j].detach())

        print(f"[Reference Bank] Successfully initialized {actual_refs} references.", flush=True)

    def refresh_descriptors(self, vae: nn.Module, dataset, device: str = "cuda", batch_size: int = 16):
        """Periodically re-encodes the bank samples using current VAE without changing images."""
        vae.eval()
        actual_refs = (self.bank_image_indices >= 0).sum().item()
        with torch.inference_mode():
            for i in range(0, actual_refs, batch_size):
                b_indices = list(range(i, min(i + batch_size, actual_refs)))
                img_indices = self.bank_image_indices[b_indices].cpu().tolist()
                tensors = torch.stack([dataset[idx]["image"] for idx in img_indices]).to(device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    mu = vae.encode(tensors).latent_dist.mean.float()
                q1, q2, mom = self.descriptor(mu)
                for j, b_i in enumerate(b_indices):
                    self.bank_q1[b_i].copy_(q1[j].detach())
                    self.bank_q2[b_i].copy_(q2[j].detach())
                    self.bank_mom[b_i].copy_(mom[j].detach())


class SupervisedStyleRankingLoss(nn.Module):
    """Computes Margin Ranking Loss over Bank: softplus((d(a, p) - d(a, n) + m) / tau)."""

    def __init__(
        self,
        margin: float = 0.15,
        temperature: float = 0.20,
        num_positives: int = 2,
        num_negatives: int = 8,
    ):
        super().__init__()
        self.margin = margin
        self.temperature = temperature
        self.num_positives = num_positives
        self.num_negatives = num_negatives

    def forward(
        self,
        q1_batch: torch.Tensor,
        q2_batch: torch.Tensor,
        mom_batch: torch.Tensor,
        style_ids: torch.Tensor,
        artist_ids: torch.Tensor,
        genre_ids: torch.Tensor,
        bank: StyleReferenceBank,
    ) -> Tuple[torch.Tensor, dict]:
        """Calculates supervised style ranking loss for current microbatch against Bank."""
        B = q1_batch.shape[0]
        device = q1_batch.device

        # Compute pairwise distance from batch (B) to entire bank (M)
        # Note: bank descriptors are detached, gradient flows purely into q1_batch, q2_batch, mom_batch
        d_mat = bank.descriptor.compute_pairwise_distance(
            q1_batch, q2_batch, mom_batch,
            bank.bank_q1, bank.bank_q2, bank.bank_mom,
        ) # (B, M)

        loss_list = []
        valid_anchors = 0
        mean_d_pos = []
        mean_d_neg = []

        bank_styles = bank.bank_styles # (M,)
        bank_artists = bank.bank_artists
        bank_genres = bank.bank_genres

        for i in range(B):
            s_a = style_ids[i].item()
            a_a = artist_ids[i].item()
            g_a = genre_ids[i].item()

            # 1. Positive Candidates: same style
            pos_mask = (bank_styles == s_a)
            pos_indices = torch.where(pos_mask)[0]
            if len(pos_indices) == 0:
                continue

            # Prioritize different artist among positives
            if a_a >= 0:
                cross_artist_pos = pos_indices[bank_artists[pos_indices] != a_a]
                if len(cross_artist_pos) >= 2:
                    pos_indices = cross_artist_pos

            # Sub-sample up to 8 positive candidates
            if len(pos_indices) > 8:
                perm = torch.randperm(len(pos_indices), device=device)[:8]
                pos_indices = pos_indices[perm]

            pos_dists = d_mat[i, pos_indices] # (K_p,)
            # Pick closest 2 positives
            topk_pos_idx = torch.topk(pos_dists, k=min(self.num_positives, len(pos_dists)), largest=False).indices
            d_p_selected = pos_dists[topk_pos_idx] # (2,)

            # 2. Negative Candidates: different style
            diff_style_mask = (bank_styles != s_a) & (bank_styles >= 0)
            
            # Hard negatives: same genre, different style
            hard_neg_mask = diff_style_mask & (bank_genres == g_a) if g_a >= 0 else diff_style_mask
            hard_indices = torch.where(hard_neg_mask)[0]
            if len(hard_indices) > 32:
                hard_indices = hard_indices[torch.randperm(len(hard_indices), device=device)[:32]]

            # Random negatives: different style
            rand_indices = torch.where(diff_style_mask)[0]
            if len(rand_indices) > 32:
                rand_indices = rand_indices[torch.randperm(len(rand_indices), device=device)[:32]]

            neg_pool = torch.unique(torch.cat([hard_indices, rand_indices]))
            if len(neg_pool) == 0:
                continue

            neg_dists = d_mat[i, neg_pool]

            # Select semi-hard negatives closest to positive distance
            # Softplus loss is highest when d_n is small
            topk_neg_idx = torch.topk(neg_dists, k=min(self.num_negatives, len(neg_dists)), largest=False).indices
            d_n_selected = neg_dists[topk_neg_idx] # (8,)

            # 3. Softplus Margin Ranking Loss for all pairs (p in d_p, n in d_n)
            # diff = (d_p - d_n + m) / tau
            pair_diff = (d_p_selected.unsqueeze(1) - d_n_selected.unsqueeze(0) + self.margin) / self.temperature
            loss_i = F.softplus(pair_diff).mean()

            loss_list.append(loss_i)
            valid_anchors += 1
            mean_d_pos.append(d_p_selected.detach().mean().item())
            mean_d_neg.append(d_n_selected.detach().mean().item())

        if valid_anchors == 0:
            loss = torch.tensor(0.0, device=device, requires_grad=True)
            stats = {"d_pos": 0.0, "d_neg": 0.0, "valid_rate": 0.0}
        else:
            loss = torch.stack(loss_list).mean()
            stats = {
                "d_pos": float(np.mean(mean_d_pos)),
                "d_neg": float(np.mean(mean_d_neg)),
                "valid_rate": float(valid_anchors / B),
            }

        return loss, stats
