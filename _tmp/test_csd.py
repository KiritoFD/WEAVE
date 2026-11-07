import sys
import importlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

csd_module = importlib.import_module("tools.12h.csd_model")
load_csd = csd_module.load_csd

def main():
    print("Testing CSD load...")
    model, ck, msg = load_csd("cuda")
    print("CSD loaded successfully! msg:", msg)

if __name__ == "__main__":
    main()
