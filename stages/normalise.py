import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import yaml

params = yaml.safe_load(open("params.yaml"))
D_VIS = params["normalise"]["pca_vis_dims"]
D_CLU = params["normalise"]["pca_cluster_dims"]

# --- Lobanov normalisation on acoustics ---
df = pd.read_csv("data/processed/features_acoustic.csv")
speaker_ids = df["speaker_id"].values

def lobanov(group):
    for col in ["F1","F2","F3"]:
        group[f"{col}_lob"] = (group[col] - group[col].mean()) / group[col].std()
    return group

df = df.groupby("speaker_id", group_keys=False).apply(lobanov).reset_index(drop=True)
df.insert(0, "speaker_id", speaker_ids)
df.to_csv("data/processed/features_acoustic_norm.csv", index=False)
print("Lobanov done. Columns:", df.columns.tolist())

# --- PCA on neural embeddings (no StandardScaler) ---
for model, path in [("whisper","data/processed/features_whisper.npz"),
                    ("xlsr",   "data/processed/features_xlsr.npz")]:
    data = np.load(path)
    out = {}
    for key in data.files:
        X = data[key].astype(np.float32)
        # remove zero-variance columns manually
        std = X.std(axis=0)
        X = X[:, std > 1e-6]
        # centre only
        X = X - X.mean(axis=0)
        vis  = PCA(D_VIS).fit_transform(X)
        clus = PCA(D_CLU).fit_transform(X)
        out[f"{key}_vis"]  = vis
        out[f"{key}_clus"] = clus
        print(f"{model} {key}: vis{vis.shape} min={vis.min():.3f} max={vis.max():.3f}")
    np.savez(f"data/processed/features_{model}_pca.npz", **out)

print("Normalisation complete.")
