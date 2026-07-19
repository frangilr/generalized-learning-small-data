# %%
import os

# import gist
import numpy as np
import pandas as pd
from PIL import Image


cats = ["bottle", "bowl", "chair", "cup", "door", "spoon", "table", "window"]
compatibility_columns = ["subj", "img1", "img2", "category", "valid", "hist_corr"]
audit_required_columns = compatibility_columns + ["shuffle_img1", "shuffle_img2"]

rgb_dir = "../../data/PanelA/RGB_hist"
gist_out_dir = "../../data/PanelA/deep_features/GIST_cosSim"
audit_path = os.path.join(rgb_dir, "NULL_all2all_corr_shuffled_chunkedK1_audit.csv")


def return_image_set(df, img1_col="img1", img2_col="img2"):
    return set(df[img1_col]).union(set(df[img2_col]))


def require_columns(df, required_columns, name):
    missing = set(required_columns).difference(df.columns)
    if missing:
        raise ValueError(f"{name} is missing required columns: {sorted(missing)}")


def get_gist_similarity(img1, img2):
    denom = np.linalg.norm(img1) * np.linalg.norm(img2)
    if denom == 0:
        return 0
    return np.dot(img1, img2) / denom


# def extract_gist_features(target_imgs):
    # folder_path = "/path/to/egocentric/images"
    # all_features = {}

    # for img_name in sorted(target_imgs):
    #     img_path = os.path.join(folder_path, img_name)
    #     image = Image.open(img_path).convert("RGB")
    #     all_features[img_name] = gist.extract(np.array(image))

    # print(f"Extracted features for {len(all_features)} images.")

    # return all_features


def load_shuffled_null_inputs():
    if not os.path.exists(audit_path):
        raise FileNotFoundError(f"Missing shuffled-null audit CSV: {audit_path}")

    audit_df = pd.read_csv(audit_path)
    require_columns(audit_df, audit_required_columns, "audit_df")

    split_dfs = {}
    for cat in cats:
        split_path = os.path.join(rgb_dir, f"NULL_forRplots_shuffled_chunkedK1_{cat}.csv")
        split_dfs[cat] = pd.read_csv(
            split_path,
            header=None,
            names=compatibility_columns,
        )

    split_df = pd.concat(split_dfs.values(), ignore_index=True)
    original_img_pool = return_image_set(split_df)
    audit_original_img_pool = return_image_set(audit_df)
    if original_img_pool != audit_original_img_pool:
        raise AssertionError(
            "Split null CSV graph-facing image pool does not match audit original image pool. "
            f"Only in split: {len(original_img_pool - audit_original_img_pool)}; "
            f"only in audit: {len(audit_original_img_pool - original_img_pool)}"
        )

    all_shuffled_imgs = return_image_set(audit_df, "shuffle_img1", "shuffle_img2")
    if len(all_shuffled_imgs) != len(original_img_pool):
        raise AssertionError(
            f"Shuffled image pool size ({len(all_shuffled_imgs)}) does not match "
            f"original/null image pool size ({len(original_img_pool)})."
        )

    print(split_df.shape)
    print()
    print(split_df.head())
    print(f"Original/null image pool: {len(original_img_pool)}")
    print(f"Shuffled feature image pool: {len(all_shuffled_imgs)}")

    return audit_df, split_dfs, all_shuffled_imgs


def write_shuffled_null_gist_features(audit_df, split_dfs, gist_feats):
    os.makedirs(gist_out_dir, exist_ok=True)

    for cat in cats:
        temp_df = split_dfs[cat].copy()
        original_columns = temp_df[compatibility_columns].copy()

        valid_mask = audit_df["valid"].astype(str).str.lower().eq("true")
        audit_cat = audit_df[
            (audit_df["category"] == f"{cat}_{cat}") & valid_mask
        ].copy()

        if temp_df.shape[0] != audit_cat.shape[0]:
            raise AssertionError(
                f"Row count mismatch for {cat}: split has {temp_df.shape[0]}, "
                f"audit has {audit_cat.shape[0]}."
            )

        for col in compatibility_columns:
            split_col = temp_df[col].reset_index(drop=True)
            audit_col = audit_cat[col].reset_index(drop=True)
            if not split_col.equals(audit_col):
                raise AssertionError(
                    f"Split/audit row-order mismatch for {cat} column {col}."
                )

        temp_df["gist_sim"] = [
            get_gist_similarity(gist_feats[shuffle_img1], gist_feats[shuffle_img2])
            for shuffle_img1, shuffle_img2 in zip(
                audit_cat["shuffle_img1"],
                audit_cat["shuffle_img2"],
            )
        ]

        if temp_df.shape[0] != original_columns.shape[0]:
            raise AssertionError(f"Output row count changed for {cat}.")
        for col in compatibility_columns:
            if not temp_df[col].equals(original_columns[col]):
                raise AssertionError(f"Output compatibility column changed for {cat}: {col}.")

        out_path = os.path.join(
            gist_out_dir,
            f"NULL_forRplots_shuffled_chunkedK1_{cat}_withGIST_cosSim.csv",
        )
        temp_df.to_csv(out_path, header=False, index=False)
        del temp_df


def main():
    audit_df, split_dfs, all_shuffled_imgs = load_shuffled_null_inputs()
    # gist_feats = extract_gist_features(all_shuffled_imgs)
    # np.save("gist_feats_shuffled_null_K1.npy", gist_feats, allow_pickle=True)
    # raise
    gist_feats = np.load("gist_feats_shuffled_null_K1.npy", allow_pickle=True).item()
    missing_features = all_shuffled_imgs.difference(gist_feats)
    if missing_features:
        raise AssertionError(
            f"Missing extracted GIST features for {len(missing_features)} shuffled images. "
            f"First missing image: {sorted(missing_features)[0]}"
        )

    write_shuffled_null_gist_features(audit_df, split_dfs, gist_feats)


if __name__ == "__main__":
    main()
