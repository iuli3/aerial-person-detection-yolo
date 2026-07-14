"""
create_test_subsets.py - Creates source-specific test subsets from merged_final
"""
from pathlib import Path
import shutil

BASE = Path('data/merged_final')
OUT  = Path('data/test_subsets')

subsets = {
    'visdrone': ['visdrone_det'],
    'hdtopview': ['hdtop', 'hdtop_orig'],
    'suas': ['suas', 'suas_orig'],
}

for subset_name, prefixes in subsets.items():
    for split in ['images', 'labels']:
        (OUT / subset_name / split / 'test').mkdir(parents=True, exist_ok=True)

    n_imgs = 0
    for prefix in prefixes:
        for img in (BASE / 'images' / 'test').glob(f'{prefix}_*'):
            shutil.copy2(img, OUT / subset_name / 'images' / 'test' / img.name)
            lbl = BASE / 'labels' / 'test' / (img.stem + '.txt')
            if lbl.exists():
                shutil.copy2(lbl, OUT / subset_name / 'labels' / 'test' / lbl.name)
            else:
                (OUT / subset_name / 'labels' / 'test' / (img.stem + '.txt')).touch()
            n_imgs += 1

    # data.yaml
    yaml = f"""path: {OUT / subset_name}
train: images/test
val: images/test
test: images/test
nc: 1
names: ['Person']
"""
    (OUT / subset_name / 'data.yaml').write_text(yaml)
    print(f"{subset_name}: {n_imgs} images")

print("Done!")