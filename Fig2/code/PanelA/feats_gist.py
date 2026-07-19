# %%
import pandas as pd 
import numpy as np 
import os 
import gist
from PIL import Image

def return_image_set(df):
    return set(
        set(df['img1']).union(set(df['img2']))
    )

# %%
cats = ["bottle", "bowl", "chair", "cup", "door", "spoon", "table", "window"]
dfs = [
        pd.read_csv(f"../../data/PanelA/all2all_corr_bysubj_forRplots_{cat}.csv", header=None) for cat in cats
    ]

df = pd.concat(dfs, ignore_index=True)
df.columns = ["subj", "img1", "img2", "obj", "valid", "hist_corr"]

# %%
print(df.shape)
print()
print(df.head())

# %%
all_imgs = return_image_set(df)
print(len(all_imgs))

def get_gist_similarity(img1, img2):
    denom = np.linalg.norm(img1) * np.linalg.norm(img2)
    if denom == 0: return 0
    return np.dot(img1, img2) / denom

# %%
def extract_gist_features(target_imgs):

    folder_path = "/path/to/egocentric_images/"
    all_features = {}
    
    for img_name in target_imgs:
        img_path = os.path.join(folder_path, img_name)
        image = Image.open(img_path).convert("RGB")
        feat = gist.extract(np.array(image))
        all_features[img_name] = feat

    print(f"Extracted features for {len(all_features)} images.")

    return all_features


gist_feats = extract_gist_features(all_imgs)

# %%
pfx = "all2all_corr_bysubj"
for cat in cats:
    temp_df = pd.read_csv(f"../../data/PanelA/{pfx}_forRplots_{cat}.csv", header=None)
    temp_df.columns = ["subj", "img1", "img2", "obj", "valid", "hist_corr"]
    temp_df["gist_sim"] = temp_df.apply(
        lambda row: get_gist_similarity(gist_feats[row["img1"]], gist_feats[row["img2"]]), 
        axis=1
    )
    temp_df.to_csv(f"../../data/PanelA/{pfx}_forRplots_{cat}_withGIST_cosSim.csv", header=False, index=False)
    del temp_df

pfx = "NULL"
for cat in cats:
    temp_df = pd.read_csv(f"../../data/PanelA/{pfx}_forRplots_{cat}.csv", header=None)
    temp_df.columns = ["subj", "img1", "img2", "obj", "valid", "hist_corr"]
    temp_df["gist_sim"] = temp_df.apply(
        lambda row: get_gist_similarity(gist_feats[row["img1"]], gist_feats[row["img2"]]), 
        axis=1
    )
    temp_df.to_csv(f"../../data/PanelA/{pfx}_forRplots_{cat}_withGIST_cosSim.csv", header=False, index=False)
    del temp_df