from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
META_CSV = PROJECT_ROOT / "output" / "metadata" / "xbd_pairs_metadata.csv"
OUT_DIR = PROJECT_ROOT / "output" / "metadata"

def make_in_domain_split(df: pd.DataFrame):
    # Keep only disaster types with enough samples
    counts = df["disaster_type"].value_counts()
    valid_types = counts[counts >= 50].index.tolist()
    df = df[df["disaster_type"].isin(valid_types)].copy()

    train_df, temp_df = train_test_split(
        df, test_size=0.30, random_state=42, stratify=df["disaster_type"]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=42, stratify=temp_df["disaster_type"]
    )

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    split_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    split_df.to_csv(OUT_DIR / "in_domain_split.csv", index=False)

    print("\nSaved in-domain split.")
    print(split_df["split"].value_counts())
    print(split_df.groupby(["split", "disaster_type"]).size())

def make_cross_disaster_split(df: pd.DataFrame):
    # Exlude volcano because too few samples
    df = df[df["disaster_type"] != "volcano"].copy()
    
    # choose one held-out disaster type
    held_out = "wind"

    train_val_df = df[df["disaster_type"] != held_out].copy()
    test_df = df[df["disaster_type"] == held_out].copy()

    train_df, val_df = train_test_split(
        train_val_df, test_size=0.15, random_state=42, stratify=train_val_df["disaster_type"]
    )

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    split_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    split_df.to_csv(OUT_DIR / "cross_disaster_split.csv", index=False)

    print("\nSaved cross-disaster split.")
    print(f"Held-out disaster type: {held_out}")
    print(split_df["split"].value_counts())
    print(split_df.groupby(["split", "disaster_type"]).size())

def main():
    df = pd.read_csv(META_CSV)
    make_in_domain_split(df)
    make_cross_disaster_split(df)

if __name__ == "__main__":
    main()