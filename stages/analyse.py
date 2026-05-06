import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse
import seaborn as sns
from scipy import stats
import warnings, os
warnings.filterwarnings("ignore")
os.makedirs("results", exist_ok=True)

df = pd.read_csv("data/processed/features_acoustic_norm.csv")

VOWELS = ['a','e','i','o','u','y','E','O','@','2','9']
vowels_present = [v for v in VOWELS if v in df.phoneme.values]
dv = df[df.phoneme.isin(vowels_present)].copy()

# --- Summary stats table ---
summary = dv.groupby(["phoneme","l1_status"]).agg(
    mean_F1=("F1_lob","mean"), median_F1=("F1_lob","median"), std_F1=("F1_lob","std"),
    mean_F2=("F2_lob","mean"), median_F2=("F2_lob","median"), std_F2=("F2_lob","std"),
    IQR_F1=("F1_lob", lambda x: x.quantile(0.75)-x.quantile(0.25)),
    IQR_F2=("F2_lob", lambda x: x.quantile(0.75)-x.quantile(0.25)),
    CV_F1=("F1_lob", lambda x: x.std()/x.mean() if x.mean()!=0 else np.nan),
    n=("F1_lob","count")
).round(3)
summary.to_csv("results/summary_stats.csv")
print("Summary stats saved.")

# --- Vowel chart ---
fig, axes = plt.subplots(1,2, figsize=(14,6))
colors = {"fr":"steelblue","ru":"tomato"}
for ax, group in zip(axes, ["fr","ru"]):
    sub = dv[dv.l1_status==group]
    for ph in vowels_present:
        pts = sub[sub.phoneme==ph]
        if len(pts) < 5: continue
        ax.scatter(-pts.F2_lob, -pts.F1_lob, alpha=0.15, s=10, color=colors[group])
        cx, cy = -pts.F2_lob.mean(), -pts.F1_lob.mean()
        ax.annotate(ph, (cx,cy), fontsize=12, fontweight="bold", ha="center")
        # 95% confidence ellipse
        cov = np.cov(-pts.F2_lob, -pts.F1_lob)
        vals, vecs = np.linalg.eigh(cov)
        angle = np.degrees(np.arctan2(*vecs[:,1][::-1]))
        w, h = 2*1.96*np.sqrt(vals)
        ax.add_patch(Ellipse((cx,cy), w, h, angle=angle,
                     edgecolor=colors[group], facecolor="none", lw=1.5))
    ax.set_xlabel("−F2 (Lobanov)")
    ax.set_ylabel("−F1 (Lobanov)")
    ax.set_title(f"{'Native French' if group=='fr' else 'Russian L1'} speakers")
plt.suptitle("French Oral Vowels — F1/F2 Space by Speaker Group", fontsize=13)
plt.tight_layout()
plt.savefig("results/vowel_chart.png", dpi=150)
plt.close()
print("Vowel chart saved.")

# --- Box plots F1 and F2 by phoneme and L1 ---
fig, axes = plt.subplots(1,2, figsize=(16,6))
for ax, feat, label in zip(axes, ["F1_lob","F2_lob"], ["F1","F2"]):
    sns.boxplot(data=dv, x="phoneme", y=feat, hue="l1_status",
                palette={"fr":"steelblue","ru":"tomato"}, ax=ax)
    ax.set_title(f"{label} by phoneme and L1 status")
    ax.set_xlabel("Phoneme")
    ax.set_ylabel(f"{label} (Lobanov)")
plt.tight_layout()
plt.savefig("results/boxplots_F1_F2.png", dpi=150)
plt.close()
print("Box plots saved.")

# --- Variance decomposition for F1 ---
print("\nVariance decomposition F1 (per vowel):")
for ph in vowels_present:
    sub = dv[dv.phoneme==ph]
    total = sub.F1_lob.var()
    inter = sub.groupby("speaker_id").F1_lob.mean().var()
    intra = sub.groupby("speaker_id").F1_lob.var().mean()
    resid = max(0, total - inter - intra)
    print(f"  {ph}: total={total:.3f} inter={inter:.3f} intra={intra:.3f} resid={resid:.3f}")

print("\nAll Stage 6 Section 5.1 outputs saved to results/")

# =============================================================================
# SECTION 5.2 — Neural Representation Visualisations
# =============================================================================
import numpy as np

df = pd.read_csv("data/processed/features_acoustic_norm.csv")
w_pca = np.load("data/processed/features_whisper_pca.npz")
x_pca = np.load("data/processed/features_xlsr_pca.npz")

VOWELS_PLOT = ['a','e','i','o','u','y']
mask = df.phoneme.isin(VOWELS_PLOT)

palette_phoneme = dict(zip(VOWELS_PLOT, sns.color_palette("tab10", len(VOWELS_PLOT))))
palette_l1      = {"fr":"steelblue","ru":"tomato"}
palette_gender  = {"f":"orchid","m":"seagreen"}

def plot_pca(coords, labels_dict, title, fname):
    fig, axes = plt.subplots(1, 3, figsize=(18,5))
    for ax, (col, palette) in zip(axes, labels_dict.items()):
        cats = df[col][mask].values
        for cat in palette:
            idx = cats == cat
            ax.scatter(coords[mask][idx,0], coords[mask][idx,1],
                       c=palette[cat], label=cat, alpha=0.3, s=8)
        ax.legend(markerscale=2, fontsize=9)
        ax.set_title(f"Coloured by {col}")
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    plt.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.savefig(f"results/{fname}", dpi=150)
    plt.close()
    print(f"Saved {fname}")

labels = {"phoneme": palette_phoneme,
          "l1_status": palette_l1,
          "gender": palette_gender}

plot_pca(w_pca["layer_low_vis"],  labels, "Whisper Layer 4 — PCA", "pca_whisper_low.png")
plot_pca(w_pca["layer_high_vis"], labels, "Whisper Layer 20 — PCA", "pca_whisper_high.png")
plot_pca(x_pca["low_vis"],        labels, "XLS-R Layer 4 — PCA",   "pca_xlsr_low.png")
plot_pca(x_pca["mid_vis"],        labels, "XLS-R Layer 10 — PCA",  "pca_xlsr_mid.png")
plot_pca(x_pca["high_vis"],       labels, "XLS-R Layer 18 — PCA",  "pca_xlsr_high.png")

# --- Between-class variance ratio ---
print("\nBetween-class variance ratio (phoneme) in 2D PCA space:")
for name, coords in [("Whisper-low", w_pca["layer_low_vis"]),
                     ("Whisper-high",w_pca["layer_high_vis"]),
                     ("XLS-R-low",  x_pca["low_vis"]),
                     ("XLS-R-mid",  x_pca["mid_vis"]),
                     ("XLS-R-high", x_pca["high_vis"])]:
    sub = coords[mask]
    labels_ph = df.phoneme[mask].values
    grand_mean = sub.mean(axis=0)
    between = sum([(labels_ph==p).sum() * ((sub[labels_ph==p].mean(axis=0)-grand_mean)**2).sum()
                   for p in VOWELS_PLOT])
    total = ((sub - grand_mean)**2).sum()
    print(f"  {name}: {between/total:.3f}")

# --- Within vs between phoneme cosine similarity ---
print("\nCosine similarity ratio (within/between phoneme):")
from sklearn.metrics.pairwise import cosine_similarity
for name, key in [("Whisper-high","layer_high_clus"),("XLS-R-high","high_clus")]:
    full = np.load(f"data/processed/features_{'whisper' if 'whisper' in name.lower() else 'xlsr'}_pca.npz")[key]
    sub = full[mask]
    labs = df.phoneme[mask].values
    within, between = [], []
    for p in VOWELS_PLOT:
        idx = np.where(labs==p)[0][:50]
        other = np.where(labs!=p)[0][:50]
        if len(idx)>1:
            within.append(cosine_similarity(sub[idx]).mean())
        if len(other)>1:
            between.append(cosine_similarity(sub[idx[:10]], sub[other[:10]]).mean())
    print(f"  {name}: within={np.mean(within):.3f} between={np.mean(between):.3f} ratio={np.mean(within)/np.mean(between):.3f}")

print("\nSection 5.2 complete.")

# =============================================================================
# SECTION 5.3 — Representational Similarity Matrix + Mantel Test
# =============================================================================
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr

df = pd.read_csv("data/processed/features_acoustic_norm.csv")
VOWELS_PLOT = ['a','e','i','o','u','y']
mask = df.phoneme.isin(VOWELS_PLOT)
dv = df[mask].copy().reset_index(drop=True)

# per-phoneme mean vectors
def get_centroids(matrix, labels, vowels):
    return np.array([matrix[labels==p].mean(axis=0) for p in vowels])

labs = dv.phoneme.values

# acoustic centroids (F1_lob, F2_lob)
ac = dv[["F1_lob","F2_lob"]].values
ac_cent = get_centroids(ac, labs, VOWELS_PLOT)
D_ac = cdist(ac_cent, ac_cent, metric="euclidean")

# neural centroids
w_pca = np.load("data/processed/features_whisper_pca.npz")
x_pca = np.load("data/processed/features_xlsr_pca.npz")

w_high = w_pca["layer_high_clus"][mask.values]
x_high = x_pca["high_clus"][mask.values]

w_cent = get_centroids(w_high, labs, VOWELS_PLOT)
x_cent = get_centroids(x_high, labs, VOWELS_PLOT)

D_wh = cdist(w_cent, w_cent, metric="cosine")
D_xl = cdist(x_cent, x_cent, metric="cosine")

def mantel(D1, D2):
    n = D1.shape[0]
    idx = np.triu_indices(n, k=1)
    r, p = spearmanr(D1[idx], D2[idx])
    return r, p

r_ac_wh, p_ac_wh = mantel(D_ac, D_wh)
r_ac_xl, p_ac_xl = mantel(D_ac, D_xl)
r_wh_xl, p_wh_xl = mantel(D_wh, D_xl)

print("\nMantel Test Results:")
print(f"  Acoustic vs Whisper:  r={r_ac_wh:.3f} p={p_ac_wh:.3f}")
print(f"  Acoustic vs XLS-R:    r={r_ac_xl:.3f} p={p_ac_xl:.3f}")
print(f"  Whisper  vs XLS-R:    r={r_wh_xl:.3f} p={p_wh_xl:.3f}")

# save RSM heatmaps
fig, axes = plt.subplots(1,3, figsize=(15,4))
for ax, D, title in zip(axes,
    [D_ac, D_wh, D_xl],
    ["Acoustic","Whisper-high","XLS-R-high"]):
    im = ax.imshow(D, cmap="viridis")
    ax.set_xticks(range(len(VOWELS_PLOT)))
    ax.set_yticks(range(len(VOWELS_PLOT)))
    ax.set_xticklabels(VOWELS_PLOT)
    ax.set_yticklabels(VOWELS_PLOT)
    ax.set_title(f"RSM: {title}")
    plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig("results/rsm_comparison.png", dpi=150)
plt.close()
print("RSM heatmap saved.")
print("\nSection 5.3 complete.")

# =============================================================================
# SECTION 6.1 — Statistical Tests: L1 vs L2 Group Comparisons
# =============================================================================
from scipy.stats import shapiro, levene, ttest_ind, mannwhitneyu
from statsmodels.stats.multitest import multipletests

df = pd.read_csv("data/processed/features_acoustic_norm.csv")
VOWELS = ['a','e','i','o','u','y']

print("\n--- L1 vs L2 on Acoustic Features ---")
results = []
for ph in VOWELS:
    for feat in ["F1_lob","F2_lob"]:
        fr = df[(df.phoneme==ph)&(df.l1_status=="fr")][feat].dropna()
        ru = df[(df.phoneme==ph)&(df.l1_status=="ru")][feat].dropna()
        if len(fr)<5 or len(ru)<5: continue

        # normality
        _, p_norm_fr = shapiro(fr.sample(min(len(fr),50), random_state=42))
        _, p_norm_ru = shapiro(ru.sample(min(len(ru),50), random_state=42))
        normal = p_norm_fr>0.05 and p_norm_ru>0.05

        # homogeneity of variance
        _, p_lev = levene(fr, ru)
        equal_var = p_lev > 0.05

        # choose test
        if normal:
            stat, p = ttest_ind(fr, ru, equal_var=equal_var)
            test = "t-test"
        else:
            stat, p = mannwhitneyu(fr, ru, alternative="two-sided")
            test = "MWU"

        results.append({"phoneme":ph,"feature":feat,"test":test,
                        "stat":round(stat,3),"p":round(p,4),
                        "mean_fr":round(fr.mean(),3),"mean_ru":round(ru.mean(),3),
                        "normal":normal})

results_df = pd.DataFrame(results)

# BH correction
_, p_corrected, _, _ = multipletests(results_df.p, method="fdr_bh")
results_df["p_corrected"] = p_corrected.round(4)
results_df["significant"] = results_df.p_corrected < 0.05

results_df.to_csv("results/l1_l2_acoustic_tests.csv", index=False)
print(results_df[["phoneme","feature","test","mean_fr","mean_ru","p","p_corrected","significant"]].to_string())

# --- Gender residual effect after Lobanov ---
print("\n--- Residual Gender Effect After Lobanov ---")
for feat in ["F1_lob","F2_lob"]:
    f_vals = df[df.gender=="f"][feat].dropna()
    m_vals = df[df.gender=="m"][feat].dropna()
    stat, p = mannwhitneyu(f_vals, m_vals, alternative="two-sided")
    print(f"  {feat}: MWU stat={stat:.0f} p={p:.4f} {'significant' if p<0.05 else 'not significant'}")

# --- L1 vs L2 on Neural Representations (permutation test) ---
print("\n--- Permutation Test: L1 vs L2 on Neural Representations ---")
from sklearn.metrics.pairwise import cosine_distances

w_pca = np.load("data/processed/features_whisper_pca.npz")
x_pca = np.load("data/processed/features_xlsr_pca.npz")

df_reset = df.reset_index(drop=True)
perm_results = []

for model_name, arr in [("Whisper-high", w_pca["layer_high_clus"]),
                         ("XLS-R-high",   x_pca["high_clus"])]:
    for ph in VOWELS:
        mask_ph = df_reset.phoneme == ph
        if mask_ph.sum() < 10: continue
        sub = arr[mask_ph.values]
        labs = df_reset[mask_ph].l1_status.values

        fr_cent = sub[labs=="fr"].mean(axis=0)
        ru_cent = sub[labs=="ru"].mean(axis=0)
        obs = cosine_distances([fr_cent],[ru_cent])[0,0]

        # permutation
        np.random.seed(42)
        null = []
        for _ in range(1000):
            perm = np.random.permutation(labs)
            c1 = sub[perm=="fr"].mean(axis=0)
            c2 = sub[perm=="ru"].mean(axis=0)
            null.append(cosine_distances([c1],[c2])[0,0])
        p_perm = (np.array(null) >= obs).mean()
        perm_results.append({"model":model_name,"phoneme":ph,
                             "obs_dist":round(obs,4),"p_perm":round(p_perm,4)})

perm_df = pd.DataFrame(perm_results)
_, p_corr, _, _ = multipletests(perm_df.p_perm, method="fdr_bh")
perm_df["p_corrected"] = p_corr.round(4)
perm_df["significant"] = perm_df.p_corrected < 0.05
perm_df.to_csv("results/l1_l2_neural_tests.csv", index=False)
print(perm_df.to_string())
print("\nSection 6.1 complete.")

# =============================================================================
# SECTION 6.2 — Inter-phoneme Distance Matrices + Classifier
# =============================================================================
from sklearn.covariance import EmpiricalCovariance
from sklearn.neighbors import NearestCentroid
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_distances
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("data/processed/features_acoustic_norm.csv")
VOWELS = ['a','e','i','o','u','y']
mask = df.phoneme.isin(VOWELS)
dv = df[mask].copy().reset_index(drop=True)
labs = dv.phoneme.values

w_pca = np.load("data/processed/features_whisper_pca.npz")
x_pca = np.load("data/processed/features_xlsr_pca.npz")

ac    = dv[["F1_lob","F2_lob"]].values
w_arr = w_pca["layer_high_clus"][mask.values]
x_arr = x_pca["high_clus"][mask.values]

def get_centroids(arr, labs, vowels):
    return np.array([arr[labs==p].mean(axis=0) for p in vowels])

ac_cent = get_centroids(ac,    labs, VOWELS)
w_cent  = get_centroids(w_arr, labs, VOWELS)
x_cent  = get_centroids(x_arr, labs, VOWELS)

# Euclidean acoustic distance
D_ac = cdist(ac_cent, ac_cent, metric="euclidean")

# Mahalanobis acoustic distance
try:
    cov = EmpiricalCovariance().fit(ac)
    VI  = np.linalg.inv(cov.covariance_)
    D_mah = cdist(ac_cent, ac_cent, metric="mahalanobis", VI=VI)
except:
    D_mah = D_ac.copy()

# Cosine neural distances
D_wh = cosine_distances(w_cent)
D_xl = cosine_distances(x_cent)

def mantel(D1, D2):
    idx = np.triu_indices(D1.shape[0], k=1)
    r, p = spearmanr(D1[idx], D2[idx])
    return round(r,3), round(p,4)

print("\n--- Inter-phoneme Distance Mantel Tests ---")
print(f"  Acoustic(Eucl) vs Whisper: r={mantel(D_ac,D_wh)[0]} p={mantel(D_ac,D_wh)[1]}")
print(f"  Acoustic(Eucl) vs XLS-R:  r={mantel(D_ac,D_xl)[0]} p={mantel(D_ac,D_xl)[1]}")
print(f"  Whisper vs XLS-R:         r={mantel(D_wh,D_xl)[0]} p={mantel(D_wh,D_xl)[1]}")
print(f"  Acoustic(Mah)  vs Whisper: r={mantel(D_mah,D_wh)[0]} p={mantel(D_mah,D_wh)[1]}")
print(f"  Acoustic(Mah)  vs XLS-R:  r={mantel(D_mah,D_xl)[0]} p={mantel(D_mah,D_xl)[1]}")

# Bootstrap CIs on selected phoneme pairs
print("\n--- Bootstrap CIs on Distance (speaker-level, B=500) ---")
PAIRS = [("e","i"), ("o","u"), ("u","y")]
speakers = dv.speaker_id.values
np.random.seed(42)

for p1, p2 in PAIRS:
    for name, arr in [("Acoustic",ac),("Whisper",w_arr),("XLS-R",x_arr)]:
        boot = []
        for _ in range(500):
            spks = np.unique(speakers)
            sampled = np.random.choice(spks, len(spks), replace=True)
            idx = np.concatenate([np.where(speakers==s)[0] for s in sampled])
            sub = arr[idx]; sub_labs = labs[idx]
            if (sub_labs==p1).sum()<2 or (sub_labs==p2).sum()<2:
                continue
            c1 = sub[sub_labs==p1].mean(axis=0)
            c2 = sub[sub_labs==p2].mean(axis=0)
            if name=="Acoustic":
                boot.append(np.linalg.norm(c1-c2))
            else:
                boot.append(cosine_distances([c1],[c2])[0,0])
        lo, hi = np.percentile(boot,[2.5,97.5])
        print(f"  {p1}/{p2} {name}: [{lo:.4f}, {hi:.4f}]")

# --- Nearest-centroid classifier (leave-one-speaker-out) ---
print("\n--- Nearest-Centroid Classifier (Leave-One-Speaker-Out) ---")
groups = dv.speaker_id.values
logo = LeaveOneGroupOut()

for name, arr in [("Acoustic",ac),("Whisper",w_arr),("XLS-R",x_arr)]:
    y_true, y_pred = [], []
    for train_idx, test_idx in logo.split(arr, labs, groups):
        clf = NearestCentroid()
        clf.fit(arr[train_idx], labs[train_idx])
        y_true.extend(labs[test_idx])
        y_pred.extend(clf.predict(arr[test_idx]))
    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average="macro")
    print(f"  {name}: accuracy={acc:.3f} macro-F1={f1:.3f}")

    # confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=VOWELS)
    fig, ax = plt.subplots(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=VOWELS,
                yticklabels=VOWELS, cmap="Blues", ax=ax)
    ax.set_title(f"Confusion Matrix — {name}")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    plt.tight_layout()
    plt.savefig(f"results/confusion_{name.lower()}.png", dpi=150)
    plt.close()

print("\nSection 6.2 complete.")
