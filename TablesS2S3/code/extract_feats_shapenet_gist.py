# %%
import pandas as pd
import os 
import numpy as np 
from PIL import Image

def get_gist_similarity(img1, img2):
    denom = np.linalg.norm(img1) * np.linalg.norm(img2)
    if denom == 0: return 0
    return np.dot(img1, img2) / denom

def return_image_set(df):
    return set(
        set(df['instance_i']).union(set(df['instance_j']))
    )

# %%
cats = ["airplane", "bed", "bottle", "camera", "car", "chair"]
rootdir = "../data/shapenet"

# NOTE can read images from a folder or here we use the dataframe itself to load the image set (and add new features/columns sequentially)
df = pd.read_csv(os.path.join(rootdir, "shapenet_sim_data.csv"))

# %%
print(df.shape)
print()
print(df.columns)

# %%
print(df['dataset'].unique())

# %%
all_imgs = return_image_set(df)
print(len(all_imgs))

# %%
clean_set = set()

for img in all_imgs:
    if len(img.split('_')) > 3:
        clean_set.add(img + '_double')
    else: 
        clean_set.add(img)

assert len(all_imgs) == len(clean_set)

# %%
def extract_GIST(target_imgs):
    import gist

    base_folder = "/your/path/datasets"

    all_features = {}
    
    for img_name in target_imgs:
        splitted = img_name.split('_')
        if "top" in splitted[1]:
            suffix = "top"
        elif "other" in splitted[1]:
            suffix = "other"
        else:
            raise

        cat = splitted[0]
        assert cat in cats

        if len(splitted) <= 3:
            img_path = os.path.join(base_folder, "shapeNet_custom_random", cat, "frames", suffix, img_name)
            image = Image.open(img_path).convert("RGB").resize((224, 224))
            all_features[img_name] = gist.extract(np.array(image))
        else: 
            raise

    print(f"Extracted features for {len(all_features)} images.")

    return all_features

PE_feats = extract_GIST(clean_set)

def RGB_router(r, feats, i):
    dataset = r['dataset']
    idx = "instance_i" if i == True else "instance_j"
    filename = r[idx]
    
    if (dataset.split('_')[-1] == "Random") or (len(filename.split('_')) <= 3):
        t = feats[filename]
    else:
        raise

    return t

# %%
rootdir = "../data/shapenet"


df["gist_cosSim"] = df.apply(
    lambda row: get_gist_similarity(RGB_router(row, PE_feats, i=True), RGB_router(row, PE_feats, i=False)), 
    axis=1
)
df.to_csv(os.path.join(rootdir, "shapenet_sim_data.csv"), index=False)