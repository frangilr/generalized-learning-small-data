# %%
import pandas as pd 
import numpy as np 
import os 
from PIL import Image
import itertools

import torch
import torch.nn.functional as F
from torchvision.transforms import v2

os.environ['CUDA_VISIBLE_DEVICES'] = "1"

def get_cosine_similarity(t1, t2):
    assert len(t1.shape) == 2
    assert len(t2.shape) == 2
    assert t1.shape[0] == 1
    assert t2.shape[0] == 1

    return F.cosine_similarity(t1, t2).item()

def get_gist_similarity(img1, img2):
    denom = np.linalg.norm(img1) * np.linalg.norm(img2)
    if denom == 0: return 0
    return np.dot(img1, img2) / denom

def return_image_set(df):
    return set(df['filename'].values)

def sum_to_skip(df):
    mask = df['filename'].apply(lambda x: x.startswith("repeated"))
    return mask.sum()

def filter_cat(df, cat):
    return df [ df['category'] == cat ].copy()

def pairwise(df, feat_dict, cats, out_dir, prefix, suffix, col_name):
    
    dfs_full = []
    for cat in cats:
        df_partial = {"instance_i": [], "instance_j": [], col_name: []}

        df_cat = filter_cat(df=df.copy(), cat=cat)
        all_img = list(return_image_set(df=df_cat))
        
        print(
            f"\nBegan category [{cat}] with [{len(all_img)}] unique images"
        )

        all_pairs = itertools.combinations(all_img, 2)
        for i,j in all_pairs:
            i_feat = feat_dict[i]
            j_feat = feat_dict[j]

            df_partial['instance_i'].append(i)
            df_partial['instance_j'].append(j)
            df_partial[col_name].append(
                # get_cosine_similarity(i_feat[0].unsqueeze(0), j_feat[0].unsqueeze(0)) # PE & CLIP
                # cv2.compareHist(i_feat, j_feat, cv2.HISTCMP_CORREL) # RGB histogram
                get_gist_similarity(i_feat, j_feat) # GIST
            )

        df_partial = pd.DataFrame(df_partial)
        df_partial['category'] = cat
        df_partial.to_csv(
            os.path.join(out_dir, "subs", f"petface_{prefix}_{suffix}_{cat}_perClass.csv"), index=False
        )
        dfs_full.append(df_partial)
            
        del df_partial
    
    df_full = pd.concat(dfs_full, ignore_index=True)
    df_full.to_csv(
        os.path.join(out_dir, f"petface_{prefix}_{suffix}_sim_data.csv"), index=False
    )    

@torch.inference_mode()
def extract_PE(df, target_imgs):
    import core.vision_encoder.pe as pe
    import core.vision_encoder.transforms as transforms

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = pe.VisionTransformer.from_config("PE-Spatial-L14-448", pretrained=True).to(device)

    preprocess = transforms.get_image_transform(model.image_size)
    
    base_folder = "/your/path/datasets/PetFace/images"

    all_features = {}

    def find_img(fname):
        row = df[ df['filename'] == fname ]
        assert row.shape[0] == 1

        img_path = os.path.join(base_folder, row.iloc[0, 0])

        return img_path
    
    @torch.inference_mode()
    def fwd(target_img):

        with torch.autocast(device_type="cuda", enabled=False):
            inputs = preprocess(target_img).unsqueeze(0).to(device)
            image_features, *_ = model(inputs)
        return image_features.cpu()
    
    for img_name in target_imgs:
        img_path = find_img(fname=img_name)
        image = Image.open(img_path).convert("RGB")#.resize((448, 448), resample=Image.Resampling.BICUBIC)
        all_features[img_name] = fwd(target_img=image)

    print(f"Extracted features for {len(all_features)} images.")

    return all_features

@torch.inference_mode()
def extract_CLIP(df, target_imgs):
    import clip

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    
    base_folder = "/your/path/datasets/PetFace/images"

    all_features = {}

    def find_img(fname):
        row = df[ df['filename'] == fname ]
        assert row.shape[0] == 1

        img_path = os.path.join(base_folder, row.iloc[0, 0])

        return img_path
    
    @torch.inference_mode()
    def fwd(target_img):

        with torch.autocast(device_type="cuda", enabled=False):
            inputs = preprocess(target_img).unsqueeze(0).to(device)
            outputs = model.encode_image(inputs)
        return outputs.cpu()
    
    for img_name in target_imgs:
        img_path = find_img(fname=img_name)
        image = Image.open(img_path).convert("RGB").resize((224, 224))
        all_features[img_name] = fwd(target_img=image)

    print(f"Extracted features for {len(all_features)} images.")

    return all_features

def extract_RGB_hist(df, target_imgs):
    import cv2

    base_folder = "/your/path/datasets/PetFace/images"

    all_features = {}

    def find_img(fname):
        row = df[ df['filename'] == fname ]
        assert row.shape[0] == 1

        img_path = os.path.join(base_folder, row.iloc[0, 0])

        return img_path
    
    for img_name in target_imgs:
        img_path = find_img(fname=img_name)
        image = cv2.imread(img_path)
        image = cv2.resize(image, (224, 224))
        # extract a 3D RGB color histogram from the image,
        # using 8 bins per channel, normalize, and update
        # the index
        hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8],
            [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()

        all_features[img_name] = hist

    print(f"Extracted features for {len(all_features)} images.")

    return all_features


def extract_GIST(df, target_imgs):
    import gist
    
    base_folder = "/your/path/datasets/PetFace/images"

    all_features = {}

    def find_img(fname):
        row = df[ df['filename'] == fname ]
        assert row.shape[0] == 1

        img_path = os.path.join(base_folder, row.iloc[0, 0])

        return img_path
    
    for img_name in target_imgs:
        img_path = find_img(fname=img_name)
        image = Image.open(img_path).convert("RGB").resize((224, 224))
        all_features[img_name] = gist.extract(np.array(image))

    print(f"Extracted features for {len(all_features)} images.")

    return all_features

def main():
    
    precomputed = False
    out_dir = "../data/petface/GIST"
    rootdir = "../data/petface"
    sfx = "lumpy"
    pfx = "GIST_cosSim"
    dataset = f"train_{sfx}_petface.csv"

    # ------------------------------------------------------------------------
    if not precomputed:

        df = pd.read_csv(os.path.join(rootdir, dataset))

        print(df.columns)
        print(df.shape)
        print()
        print(df.head())

        cats = list(df['category'].unique())
        assert len(cats) == 13

        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(os.path.join(out_dir, "subs"), exist_ok=True)

        # print("PE configs:", pe.VisionTransformer.available_configs())

        all_imgs = return_image_set(df=df)
        print(len(all_imgs))
        
        print()
        print("Began extracting features.")
        PE_feats = extract_GIST(df=df, target_imgs=all_imgs)

        print("Done extracting features.")
        np.save(
            os.path.join(out_dir, "subs", f"{pfx}_feats_{sfx}.npy"), PE_feats, allow_pickle=True
        )
        print("Saved features to disk.")

        del df
    
    # ------------------------------------------------------------------------
    else:
        PE_feats = np.load(os.path.join(out_dir, "subs", f"{pfx}_feats_{sfx}.npy"), allow_pickle=True).item()
        print("Loaded features from disk.")

    # ------------------------------------------------------------------------
    df = pd.read_csv(os.path.join(rootdir, dataset))
    cats = list(df['category'].unique())
    assert len(cats) == 13

    # ------------------------------------------------------------------------
    print()
    print("Began pairwise.")
    pairwise(df=df, feat_dict=PE_feats, cats=cats, out_dir=out_dir, prefix=pfx, suffix=sfx, col_name=f"{pfx}_cos_sim")

if __name__ == "__main__":
    main()
