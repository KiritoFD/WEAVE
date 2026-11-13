"""配对 latent 直通与 PairedLatentDataset 的契约测试。"""
import json
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.legacy_root.flow import FlowMatchingObjective  # noqa: E402
from tools.pseudo_gt.cache import SampleRecord, ShardWriter  # noqa: E402
from tools.legacy_utils.dataset import PairedLatentDataset  # noqa: E402


def test_passthrough_returns_target_untouched():
    obj = object.__new__(FlowMatchingObjective)
    obj.paired_latent_passthrough = True
    z0 = torch.zeros(2, 4, 64, 64)
    z1 = torch.ones(2, 4, 64, 64)
    out = FlowMatchingObjective._build_training_target(obj, z0, z1, torch.zeros(2, 4, 64, 64))
    assert torch.equal(out, z1)


def test_paired_dataset_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        writer = ShardWriter(root, shard_size=2)
        for i in range(3):
            z0 = torch.rand(4, 64, 64)
            z1 = torch.rand(4, 64, 64)
            writer.add(
                z0,
                z1,
                SampleRecord(
                    sample_id=f"id{i}",
                    source_id=f"s{i}",
                    donor_id=f"d{i}",
                    source_style="Impressionism",
                    target_style="Rococo",
                    method="A",
                    patch_spec="P2",
                    seed=i,
                    match_score=0.9,
                    nominal_coverage=0.25,
                    effective_coverage=0.25,
                    z0_is_original=True,
                    z1_is_original=False,
                ),
            )
        writer.close()
        ds = PairedLatentDataset(str(root))
        assert len(ds) == 3
        item = ds[0]
        assert item["content"].shape == (4, 64, 64)
        assert item["target_style"].shape == (4, 64, 64)
        assert int(item["target_style_id"]) == 3  # Rococo
        assert int(item["source_style_id"]) == 1  # Impressionism
        assert item["pair_sample_id"] == "id0"
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["count"] == 3
