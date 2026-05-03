import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import yaml

params = yaml.safe_load(open("params.yaml"))
D_VIS = params["normalise"]["pca_vis_dims"]
D_CLU = params["normalise"]["pca_cluster_dims"]

# --- Lobanov normalisation on acoustics ---
df = pd.read_csv("data/processed/features_acoustic.csv")

def lobanov(group):
    for col in ["F1","F2","F3"]:
        group[f"{col}_lob"] = (group[col] - group[col].mean()) / group[col].std()
    return group

df = df.groupby("speaker_id", group_keys=False).apply(lobanov)
df.to_csv("data/processed/features_acoustic_norm.csv", index=False)
print("Lobanov done:", df[["F1_lob","F2_lob"]].describe().round(3).to_string())

# --- PCA on neural embeddings ---
for model, path in [("whisper","data/processed/features_whisper.npz"),
                    ("xlsr",   "data/processed/features_xlsr.npz")]:
    data = np.load(path)
    out = {}
    for key in data.files:
        X = data[key]
        X = StandardScaler().fit_transform(X)
        out[f"{key}_vis"]  = PCA(D_VIS).fit_transform(X)
        out[f"{key}_clus"] = PCA(D_CLU).fit_transform(X)
        print(f"{model} {key} → vis{out[f'{key}_vis'].shape} clus{out[f'{key}_clus'].shape}")
    np.savez(f"data/processed/features_{model}_pca.npz", **out)

print("Normalisation complete.")
