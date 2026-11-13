"""§9.3: report dinov2 attention implementation used by the feature extractor."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.legacy_root.features import build_feature_extractor

e = build_feature_extractor("dinov2", device="cpu", local_dir="/mnt/i/Latent_Style/weights/dinov2-base")
print("attn_impl:", e.model.config._attn_implementation)
print("attn_layer:", type(e.model.encoder.layer[0].attention.attention).__name__)

