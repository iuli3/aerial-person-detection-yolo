from pathlib import Path

BASE = Path("data/eval_scale_density_balanced")

subsets = [
    "scale_small_high_altitude",
    "scale_medium_scale",
    "scale_large_low_altitude",
    "density_sparse",
    "density_medium_density",
    "density_crowded",
]

prefixes = {
    "visdrone": "VisDrone",
    "hdtop": "Stanford/HDTopView",
    "suas": "SUAS",
}

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

for subset in subsets:
    img_dir = BASE / subset / "images" / "test"
    lbl_dir = BASE / subset / "labels" / "test"

    imgs = [p for p in img_dir.iterdir() if p.suffix.lower() in EXTS]

    print("\n" + "=" * 80)
    print(subset)
    print("=" * 80)
    print(f"Total images: {len(imgs)}")

    for prefix, name in prefixes.items():
        selected = [p for p in imgs if p.name.startswith(prefix)]
        n_imgs = len(selected)

        n_inst = 0
        for img in selected:
            lbl = lbl_dir / f"{img.stem}.txt"
            if lbl.exists():
                with open(lbl) as f:
                    n_inst += sum(1 for line in f if line.strip())

        pct = 100 * n_imgs / len(imgs) if imgs else 0
        avg = n_inst / n_imgs if n_imgs else 0

        print(f"{name:<18}: {n_imgs:>5} images ({pct:>5.1f}%), {n_inst:>6} instances, {avg:>6.2f} inst/img")