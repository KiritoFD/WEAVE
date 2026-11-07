from pathlib import Path

def main():
    test_dir = Path("/mnt/i/Latent_Style/WEAVE/data/test")
    styles = sorted([d.name for d in test_dir.iterdir() if d.is_dir()])
    print("Styles:", styles)
    for s in styles:
        imgs = sorted(list((test_dir / s).glob("*.jpg")))
        print(f"  {s}: {len(imgs)} images, first 3: {[im.name for im in imgs[:3]]}")

if __name__ == "__main__":
    main()
