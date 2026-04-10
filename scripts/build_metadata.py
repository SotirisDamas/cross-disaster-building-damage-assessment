from pathlib import Path
import json
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROOT = PROJECT_ROOT / "train_images_labels_targets" / "train"
IMAGES_DIR = ROOT / "images"
LABELS_DIR = ROOT / "labels"
TARGETS_DIR = ROOT / "targets"
OUT_DIR = PROJECT_ROOT / "output" / "metadata"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def get_base_id(filename: str) -> str:
    name = Path(filename).stem

    for suffix in [
        "_pre_disaster_target",
        "_post_disaster_target",
        "_pre_disaster",
        "_post_disaster",
    ]:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break

    return name

def main():
    pre_imgs = {get_base_id(p.name): p for p in IMAGES_DIR.glob("*_pre_disaster.png")}
    post_imgs = {get_base_id(p.name): p for p in IMAGES_DIR.glob("*_post_disaster.png")}
    pre_targets = {get_base_id(p.name): p for p in TARGETS_DIR.glob("*_pre_disaster_target.png")}
    post_targets = {get_base_id(p.name): p for p in TARGETS_DIR.glob("*_post_disaster_target.png")}

    common_ids = sorted(set(pre_imgs) & set(post_imgs))

    rows = []
    skipped = 0

    for sample_id in common_ids:
        pre_path = pre_imgs[sample_id]
        post_path = post_imgs[sample_id]

        pre_label_path = LABELS_DIR / f"{pre_path.stem}.json"
        post_label_path = LABELS_DIR / f"{post_path.stem}.json"

        pre_target_path = pre_targets.get(sample_id, "")
        post_target_path = post_targets.get(sample_id, "")

        if not pre_label_path.exists() or not post_label_path.exists():
            skipped += 1
            continue

        with open(post_label_path, "r") as f:
            post_json = json.load(f)

        meta = post_json.get("metadata", {})

        rows.append({
            "sample_id": sample_id,
            "pre_path": str(pre_path),
            "post_path": str(post_path),
            "pre_label_path": str(pre_label_path),
            "post_label_path": str(post_label_path),
            "pre_target_path": str(pre_target_path),
            "post_target_path": str(post_target_path),
            "disaster": meta.get("disaster"),
            "disaster_type": meta.get("disaster_type"),
            "width": meta.get("width", 1024),
            "height": meta.get("height", 1024),
        })

    df = pd.DataFrame(rows)
    out_path = OUT_DIR / "xbd_pairs_metadata.csv"
    df.to_csv(out_path, index=False)

    print(f"Saved metadata to: {out_path}")
    print(f"Total paired samples: {len(df)}")
    print(f"Skipped: {skipped}")

    print("\nDisaster type counts:")
    print(df["disaster_type"].value_counts())

    print("\nMissing pre targets:", (df["pre_target_path"] == "").sum())
    print("Missing post targets:", (df["post_target_path"] == "").sum())

if __name__ == "__main__":
    main()