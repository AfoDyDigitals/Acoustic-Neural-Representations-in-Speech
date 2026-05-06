import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr, shapiro, levene, ttest_ind, mannwhitneyu
from sklearn.metrics.pairwise import cosine_distances
from sklearn.covariance import EmpiricalCovariance
from sklearn.neighbors import NearestCentroid
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from statsmodels.stats.multitest import multipletests
import statsmodels.formula.api as smf
import warnings, os
warnings.filterwarnings("ignore")
os.makedirs("results", exist_ok=True)

df = pd.read_csv("data/processed/features_acoustic_norm.csv")
VOWELS = ['a','e','i','o','u','y']
vowels_present = [v for v in VOWELS if v in df.phoneme.values]
dv = df[df.phoneme.isin(vowels_present)].copy()
w_pca = np.load("data/processed/features_whisper_pca.npz")
x_pca = np.load("data/processed/features_xlsr_pca.npz")

# =============================================================================
# SECTION 5.1 — Descriptive Statistics
# =============================================================================
summary = dv.groupby(["phoneme","l1_status"]).agg(
    mean_F1=("F1_lob","mean"), median_F1=("F1_lob","median"), std_F1=("F1_lob","std"),
    mean_F2=("F2_lob","mean"), median_F2=("F2_lob","median"), std_F2=("F2_lob","std"),
    IQR_F1=("F1_lob", lambda x: x.quantile(0.75)-x.quantile(0.25)),
    IQR_F2=("F2_lob", lambda x: x.quantile(0.75)-x.quantile(0.25)),
    n=("F1_lob","count")
).round(3)
summary.to_csv("results/summary_stats.csv")
print("Summary stats saved.")

palette_phoneme = dict(zip(VOWELS, sns.color_palette("tab10", len(VOWELS))))
palette_l1      = {"fr":"steelblue","ru":"tomato"}
palette_gender  = {"f":"orchid","m":"seagreen"}

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
        cov = np.cov(-pts.F2_lob, -pts.F1_lob)
        vals, vecs = np.linalg.eigh(cov)
        angle = np.degrees(np.arctan2(*vecs[:,1][::-1]))
        w, h = 2*1.96*np.sqrt(np.abs(vals))
        ax.add_patch(Ellipse((cx,cy), w, h, angle=angle,
                     edgecolor=colors[group], facecolor="none", lw=1.5))
    ax.set_xlabel("−F2 (Lobanov)"); ax.set_ylabel("−F1 (Lobanov)")
    ax.set_title(f"{'Native French' if group=='fr' else 'Russian L1'} speakers")
plt.suptitle("French Oral Vowels — F1/F2 Space", fontsize=13)
plt.tight_layout()
plt.savefig("results/vowel_chart.png", dpi=150); plt.close()
print("Vowel chart saved.")

fig, axes = plt.subplots(1,2, figsize=(16,6))
for ax, feat, label in zip(axes, ["F1_lob","F2_lob"], ["F1","F2"]):
    sns.boxplot(data=dv, x="phoneme", y=feat, hue="l1_status",
                palette={"fr":"steelblue","ru":"tomato"}, ax=ax)
    ax.set_title(f"{label} by phoneme and L1 status")
plt.tight_layout()
plt.savefig("results/boxplots_F1_F2.png", dpi=150); plt.close()
print("Box plots saved.")

print("\nVariance decomposition F1 (per vowel):")
for ph in vowels_present:
    sub = dv[dv.phoneme==ph]
    total = sub.F1_lob.var()
    inter = sub.groupby("speaker_id").F1_lob.mean().var()
    intra = sub.groupby("speaker_id").F1_lob.var().mean()
    resid = max(0, total - inter - intra)
    print(f"  {ph}: total={total:.3f} inter={inter:.3f} intra={intra:.3f} resid={resid:.3f}")
print("Section 5.1 complete.")

# =============================================================================
# SECTION 5.2 — Neural PCA Visualisations
# =============================================================================
mask = df.phoneme.isin(VOWELS)

def plot_pca(coords, labels_dict, title, fname):
    fig, axes = plt.subplots(1,3, figsize=(18,5))
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
    plt.savefig(f"results/{fname}", dpi=150); plt.close()
    print(f"Saved {fname}")

labels = {"phoneme":palette_phoneme,"l1_status":palette_l1,"gender":palette_gender}
plot_pca(w_pca["layer_low_vis"],  labels, "Whisper Layer 4 — PCA",  "pca_whisper_low.png")
plot_pca(w_pca["layer_high_vis"], labels, "Whisper Layer 20 — PCA", "pca_whisper_high.png")
plot_pca(x_pca["low_vis"],        labels, "XLS-R Layer 4 — PCA",    "pca_xlsr_low.png")
plot_pca(x_pca["mid_vis"],        labels, "XLS-R Layer 10 — PCA",   "pca_xlsr_mid.png")
plot_pca(x_pca["high_vis"],       labels, "XLS-R Layer 18 — PCA",   "pca_xlsr_high.png")

labs = dv.phoneme.values
print("\nBetween-class variance ratio:")
for name, npz, key in [
    ("Whisper-low",  "data/processed/features_whisper_pca.npz","layer_low_vis"),
    ("Whisper-high", "data/processed/features_whisper_pca.npz","layer_high_vis"),
    ("XLS-R-low",    "data/processed/features_xlsr_pca.npz","low_vis"),
    ("XLS-R-mid",    "data/processed/features_xlsr_pca.npz","mid_vis"),
    ("XLS-R-high",   "data/processed/features_xlsr_pca.npz","high_vis")]:
    coords = np.load(npz)[key][mask.values]
    l = dv.phoneme.values
    grand = coords.mean(axis=0)
    between = sum([(l==p).sum()*((coords[l==p].mean(axis=0)-grand)**2).sum()
                   for p in VOWELS if (l==p).sum()>0])
    total = ((coords-grand)**2).sum()
    print(f"  {name}: {between/total:.3f}" if total>0 else f"  {name}: nan")

print("\nCosine similarity ratio:")
for name, npz, key in [
    ("Whisper-high","data/processed/features_whisper_pca.npz","layer_high_clus"),
    ("XLS-R-high",  "data/processed/features_xlsr_pca.npz","high_clus")]:
    from sklearn.metrics.pairwise import cosine_similarity
    full = np.load(npz)[key]
    sub = full[mask.values]; l = dv.phoneme.values
    within, between = [], []
    for p in VOWELS:
        idx = np.where(l==p)[0][:50]; other = np.where(l!=p)[0][:50]
        if len(idx)>1:
            within.append(cosine_similarity(sub[idx]).mean())
            between.append(cosine_similarity(sub[idx[:10]],sub[other[:10]]).mean())
    print(f"  {name}: within={np.mean(within):.3f} between={np.mean(between):.3f} ratio={np.mean(within)/np.mean(between):.3f}")
print("Section 5.2 complete.")

# =============================================================================
# SECTION 5.3 — RSM + Mantel
# =============================================================================
def get_centroids(arr, labs, vowels):
    return np.array([arr[labs==p].mean(axis=0) for p in vowels])

labs = dv.phoneme.values
ac     = dv[["F1_lob","F2_lob"]].values
w_high = w_pca["layer_high_clus"][mask.values]
x_high = x_pca["high_clus"][mask.values]

ac_cent = get_centroids(ac,     labs, VOWELS)
w_cent  = get_centroids(w_high, labs, VOWELS)
x_cent  = get_centroids(x_high, labs, VOWELS)

D_ac = cdist(ac_cent, ac_cent, metric="euclidean")
D_wh = cosine_distances(w_cent)
D_xl = cosine_distances(x_cent)

def mantel(D1, D2):
    idx = np.triu_indices(D1.shape[0], k=1)
    r, p = spearmanr(D1[idx], D2[idx])
    return round(r,3), round(p,4)

print("\nMantel Test Results:")
print(f"  Acoustic vs Whisper: r={mantel(D_ac,D_wh)[0]} p={mantel(D_ac,D_wh)[1]}")
print(f"  Acoustic vs XLS-R:  r={mantel(D_ac,D_xl)[0]} p={mantel(D_ac,D_xl)[1]}")
print(f"  Whisper  vs XLS-R:  r={mantel(D_wh,D_xl)[0]} p={mantel(D_wh,D_xl)[1]}")

fig, axes = plt.subplots(1,3, figsize=(15,4))
for ax, D, title in zip(axes,[D_ac,D_wh,D_xl],["Acoustic","Whisper-high","XLS-R-high"]):
    im = ax.imshow(D, cmap="viridis")
    ax.set_xticks(range(len(VOWELS))); ax.set_yticks(range(len(VOWELS)))
    ax.set_xticklabels(VOWELS); ax.set_yticklabels(VOWELS)
    ax.set_title(f"RSM: {title}"); plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig("results/rsm_comparison.png", dpi=150); plt.close()
print("RSM saved. Section 5.3 complete.")

# =============================================================================
# SECTION 6.1 — Statistical Tests
# =============================================================================
print("\n--- L1 vs L2 Acoustic Tests ---")
results = []
for ph in VOWELS:
    for feat in ["F1_lob","F2_lob"]:
        fr = df[(df.phoneme==ph)&(df.l1_status=="fr")][feat].dropna()
        ru = df[(df.phoneme==ph)&(df.l1_status=="ru")][feat].dropna()
        if len(fr)<5 or len(ru)<5: continue
        _, p_norm_fr = shapiro(fr.sample(min(len(fr),50), random_state=42))
        _, p_norm_ru = shapiro(ru.sample(min(len(ru),50), random_state=42))
        normal = p_norm_fr>0.05 and p_norm_ru>0.05
        _, p_lev = levene(fr, ru)
        if normal:
            stat, p = ttest_ind(fr, ru, equal_var=p_lev>0.05)
            test = "t-test"
        else:
            stat, p = mannwhitneyu(fr, ru, alternative="two-sided")
            test = "MWU"
        results.append({"phoneme":ph,"feature":feat,"test":test,
                        "stat":round(stat,3),"p":round(p,4),
                        "mean_fr":round(fr.mean(),3),"mean_ru":round(ru.mean(),3)})

results_df = pd.DataFrame(results)
_, p_corr, _, _ = multipletests(results_df.p, method="fdr_bh")
results_df["p_corrected"] = p_corr.round(4)
results_df["significant"] = results_df.p_corrected < 0.05
results_df.to_csv("results/l1_l2_acoustic_tests.csv", index=False)
print(results_df[["phoneme","feature","test","mean_fr","mean_ru","p","p_corrected","significant"]].to_string())

print("\n--- Residual Gender Effect ---")
for feat in ["F1_lob","F2_lob"]:
    f_v = df[df.gender=="f"][feat].dropna()
    m_v = df[df.gender=="m"][feat].dropna()
    stat, p = mannwhitneyu(f_v, m_v, alternative="two-sided")
    print(f"  {feat}: p={p:.4f} {'significant' if p<0.05 else 'not significant'}")

print("\n--- Permutation Test Neural ---")
df_r = df.reset_index(drop=True)
perm_results = []
for model_name, arr in [("Whisper-high",w_pca["layer_high_clus"]),
                         ("XLS-R-high",  x_pca["high_clus"])]:
    for ph in VOWELS:
        mask_ph = df_r.phoneme==ph
        if mask_ph.sum()<10: continue
        sub = arr[mask_ph.values]; labs2 = df_r[mask_ph].l1_status.values
        fr_c = sub[labs2=="fr"].mean(axis=0); ru_c = sub[labs2=="ru"].mean(axis=0)
        obs = cosine_distances([fr_c],[ru_c])[0,0]
        np.random.seed(42)
        null = []
        for _ in range(1000):
            perm = np.random.permutation(labs2)
            null.append(cosine_distances([sub[perm=="fr"].mean(axis=0)],
                                         [sub[perm=="ru"].mean(axis=0)])[0,0])
        p_perm = (np.array(null)>=obs).mean()
        perm_results.append({"model":model_name,"phoneme":ph,
                             "obs_dist":round(obs,4),"p_perm":round(p_perm,4)})

perm_df = pd.DataFrame(perm_results)
_, p_c, _, _ = multipletests(perm_df.p_perm, method="fdr_bh")
perm_df["p_corrected"] = p_c.round(4)
perm_df["significant"]  = perm_df.p_corrected < 0.05
perm_df.to_csv("results/l1_l2_neural_tests.csv", index=False)
print(perm_df.to_string())
print("Section 6.1 complete.")

# =============================================================================
# SECTION 6.2 — Distance Matrices + Classifier
# =============================================================================
mask2 = df.phoneme.isin(VOWELS)
dv2 = df[mask2].copy().reset_index(drop=True)
labs2 = dv2.phoneme.values
ac2    = dv2[["F1_lob","F2_lob"]].values
w_arr  = w_pca["layer_high_clus"][mask2.values]
x_arr  = x_pca["high_clus"][mask2.values]

ac_cent2 = get_centroids(ac2,   labs2, VOWELS)
w_cent2  = get_centroids(w_arr, labs2, VOWELS)
x_cent2  = get_centroids(x_arr, labs2, VOWELS)

D_ac2 = cdist(ac_cent2, ac_cent2, metric="euclidean")
try:
    cov = EmpiricalCovariance().fit(ac2)
    VI  = np.linalg.inv(cov.covariance_)
    D_mah = cdist(ac_cent2, ac_cent2, metric="mahalanobis", VI=VI)
except:
    D_mah = D_ac2.copy()
D_wh2 = cosine_distances(w_cent2)
D_xl2 = cosine_distances(x_cent2)

print("\n--- Distance Mantel Tests ---")
print(f"  Acoustic(Eucl) vs Whisper: r={mantel(D_ac2,D_wh2)[0]} p={mantel(D_ac2,D_wh2)[1]}")
print(f"  Acoustic(Eucl) vs XLS-R:  r={mantel(D_ac2,D_xl2)[0]} p={mantel(D_ac2,D_xl2)[1]}")
print(f"  Whisper vs XLS-R:         r={mantel(D_wh2,D_xl2)[0]} p={mantel(D_wh2,D_xl2)[1]}")
print(f"  Acoustic(Mah)  vs Whisper: r={mantel(D_mah,D_wh2)[0]} p={mantel(D_mah,D_wh2)[1]}")
print(f"  Acoustic(Mah)  vs XLS-R:  r={mantel(D_mah,D_xl2)[0]} p={mantel(D_mah,D_xl2)[1]}")

print("\n--- Bootstrap CIs ---")
speakers2 = dv2.speaker_id.values
np.random.seed(42)
for p1, p2 in [("e","i"),("o","u"),("u","y")]:
    for name, arr in [("Acoustic",ac2),("Whisper",w_arr),("XLS-R",x_arr)]:
        boot = []
        for _ in range(500):
            samp = np.random.choice(np.unique(speakers2), len(np.unique(speakers2)), replace=True)
            idx  = np.concatenate([np.where(speakers2==s)[0] for s in samp])
            sl   = arr[idx]; sl2 = labs2[idx]
            if (sl2==p1).sum()<2 or (sl2==p2).sum()<2: continue
            c1 = sl[sl2==p1].mean(axis=0); c2 = sl[sl2==p2].mean(axis=0)
            boot.append(np.linalg.norm(c1-c2) if name=="Acoustic"
                        else cosine_distances([c1],[c2])[0,0])
        lo, hi = np.percentile(boot,[2.5,97.5])
        print(f"  {p1}/{p2} {name}: [{lo:.4f},{hi:.4f}]")

print("\n--- Nearest-Centroid Classifier ---")
groups2 = dv2.speaker_id.values
logo = LeaveOneGroupOut()
for name, arr in [("Acoustic",ac2),("Whisper",w_arr),("XLS-R",x_arr)]:
    y_true, y_pred = [], []
    for tr, te in logo.split(arr, labs2, groups2):
        clf = NearestCentroid()
        clf.fit(arr[tr], labs2[tr])
        y_true.extend(labs2[te]); y_pred.extend(clf.predict(arr[te]))
    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average="macro")
    print(f"  {name}: accuracy={acc:.3f} macro-F1={f1:.3f}")
    cm = confusion_matrix(y_true, y_pred, labels=VOWELS)
    fig, ax = plt.subplots(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=VOWELS,
                yticklabels=VOWELS, cmap="Blues", ax=ax)
    ax.set_title(f"Confusion Matrix — {name}")
    plt.tight_layout()
    plt.savefig(f"results/confusion_{name.lower()}.png", dpi=150); plt.close()
print("Section 6.2 complete.")

# =============================================================================
# SECTION 7 — Linear Mixed-Effects Models
# =============================================================================
print("\n--- Linear Mixed-Effects Models ---")
lme_results = []
for ph in VOWELS:
    sub = df[df.phoneme==ph].copy()
    sub["L2"]   = (sub.l1_status=="ru").astype(int)
    sub["Male"] = (sub.gender=="m").astype(int)
    if len(sub)<20: continue
    try:
        m0 = smf.mixedlm("F1_lob ~ 1", sub, groups=sub["speaker_id"]).fit(
             reml=True, method="lbfgs", warn_convergence=False)
        m1 = smf.mixedlm("F1_lob ~ L2 + Male", sub, groups=sub["speaker_id"]).fit(
             reml=True, method="lbfgs", warn_convergence=False)
        m2 = smf.mixedlm("F1_lob ~ L2 * Male", sub, groups=sub["speaker_id"]).fit(
             reml=True, method="lbfgs", warn_convergence=False)
        var_u = max(float(m0.cov_re.iloc[0,0]), 1e-10)
        var_e = m0.scale
        icc   = var_u / (var_u + var_e)
        fe    = m1.params["Intercept"] + m1.params.get("L2",0)*sub["L2"] + m1.params.get("Male",0)*sub["Male"]
        r2_m  = np.var(fe) / (var_u + var_e)
        lme_results.append({
            "phoneme":ph, "ICC":round(icc,3), "R2_marginal":round(r2_m,3),
            "L2_coef":round(m1.params.get("L2",np.nan),3),
            "L2_pval":round(m1.pvalues.get("L2",np.nan),4),
            "interaction_pval":round(m2.pvalues.get("L2:Male",np.nan),4),
            "AIC_null":round(m0.aic,1), "AIC_main":round(m1.aic,1)
        })
    except Exception as ex:
        print(f"  /{ph}/ skipped: {ex}")

lme_df = pd.DataFrame(lme_results)
lme_df.to_csv("results/lme_results.csv", index=False)
print(lme_df.to_string())

print("\n--- Neural PC1 LME ---")
for ph in ["a","i"]:
    mask_ph = df.phoneme==ph
    sub = df[mask_ph].copy().reset_index(drop=True)
    sub["L2"]   = (sub.l1_status=="ru").astype(int)
    sub["Male"] = (sub.gender=="m").astype(int)
    sub["PC1_w"] = w_pca["layer_high_clus"][mask_ph.values,0]
    sub["PC1_x"] = x_pca["high_clus"][mask_ph.values,0]
    for pc_col, mn in [("PC1_w","Whisper"),("PC1_x","XLS-R")]:
        try:
            m = smf.mixedlm(f"{pc_col} ~ L2 + Male", sub,
                            groups=sub["speaker_id"]).fit(
                            reml=True, method="lbfgs", warn_convergence=False)
            var_u = max(float(m.cov_re.iloc[0,0]),1e-10)
            var_e = m.scale
            icc   = var_u/(var_u+var_e)
            fe    = m.params["Intercept"]+m.params.get("L2",0)*sub["L2"]+m.params.get("Male",0)*sub["Male"]
            r2_m  = np.var(fe)/(var_u+var_e)
            print(f"  /{ph}/ {mn}: ICC={icc:.3f} R2={r2_m:.3f} L2_coef={m.params.get('L2',np.nan):.3f} L2_p={m.pvalues.get('L2',np.nan):.4f}")
        except Exception as ex:
            print(f"  /{ph}/ {mn}: skipped ({ex})")
print("Section 7 complete.")

# =============================================================================
# SECTION 8 — Confidence Intervals + ROPE
# =============================================================================
print("\n--- Acoustic CIs ---")
acoustic_cis = []
for ph in VOWELS:
    sub = df[df.phoneme==ph].copy()
    sub["L2"]   = (sub.l1_status=="ru").astype(int)
    sub["Male"] = (sub.gender=="m").astype(int)
    if len(sub)<20: continue
    for feat in ["F1_lob","F2_lob"]:
        try:
            m = smf.mixedlm(f"{feat} ~ L2 + Male", sub,
                            groups=sub["speaker_id"]).fit(
                            reml=True, method="lbfgs", warn_convergence=False)
            ci = m.conf_int()
            if "L2" in ci.index:
                lo, hi = ci.loc["L2"]
                acoustic_cis.append({"phoneme":ph,"feature":feat,
                                     "coef":round(m.params["L2"],3),
                                     "CI_low":round(lo,3),"CI_high":round(hi,3)})
        except: pass

ci_df = pd.DataFrame(acoustic_cis)
print(ci_df.to_string())
ci_df.to_csv("results/acoustic_cis.csv", index=False)

fig, axes = plt.subplots(1,2, figsize=(12,6))
for ax, feat in zip(axes,["F1_lob","F2_lob"]):
    sub_ci = ci_df[ci_df.feature==feat].reset_index(drop=True)
    y = range(len(sub_ci))
    ax.errorbar(sub_ci.coef, list(y),
                xerr=[sub_ci.coef-sub_ci.CI_low, sub_ci.CI_high-sub_ci.coef],
                fmt="o", color="steelblue", capsize=4)
    ax.axvline(0, color="red", linestyle="--", alpha=0.7)
    ax.set_yticks(list(y)); ax.set_yticklabels(sub_ci.phoneme.tolist())
    ax.set_xlabel("L2 coefficient"); ax.set_title(f"Forest Plot — {feat}")
plt.suptitle("L1/L2 Contrasts with 95% CI", fontsize=13)
plt.tight_layout()
plt.savefig("results/forest_acoustic.png", dpi=150); plt.close()
print("Forest plot saved.")

print("\n--- Neural Bootstrap CIs ---")
df_r2 = df.reset_index(drop=True)
neural_cis = []
np.random.seed(42)
for model_name, arr in [("Whisper",w_pca["layer_high_clus"]),
                         ("XLS-R",  x_pca["high_clus"])]:
    mask_v = df_r2.phoneme.isin(VOWELS)
    arr_v  = arr[mask_v.values]
    dv3    = df_r2[mask_v].reset_index(drop=True)
    spks3  = dv3.speaker_id.values
    for ph in VOWELS:
        mph = dv3.phoneme==ph
        sub = arr_v[mph.values]; labs3 = dv3[mph].l1_status.values; spk3 = dv3[mph].speaker_id.values
        if (labs3=="fr").sum()<3 or (labs3=="ru").sum()<3: continue
        boot = []
        for _ in range(500):
            samp = np.random.choice(np.unique(spk3), len(np.unique(spk3)), replace=True)
            idx  = np.concatenate([np.where(spk3==s)[0] for s in samp])
            sl   = sub[idx]; sl2 = labs3[idx]
            if (sl2=="fr").sum()<2 or (sl2=="ru").sum()<2: continue
            boot.append(cosine_distances([sl[sl2=="fr"].mean(axis=0)],
                                         [sl[sl2=="ru"].mean(axis=0)])[0,0])
        obs = cosine_distances([sub[labs3=="fr"].mean(axis=0)],
                               [sub[labs3=="ru"].mean(axis=0)])[0,0]
        lo, hi = np.percentile(boot,[2.5,97.5])
        neural_cis.append({"model":model_name,"phoneme":ph,
                           "obs":round(obs,4),"CI_low":round(lo,4),"CI_high":round(hi,4)})

nci_df = pd.DataFrame(neural_cis)
print(nci_df.to_string())
nci_df.to_csv("results/neural_cis.csv", index=False)

print("\n--- ROPE Classification ---")
ROPE_AC = 0.05
rope_results = []
for model_name, arr in [("Whisper",w_pca["layer_high_clus"]),
                         ("XLS-R",  x_pca["high_clus"])]:
    mask_v = df_r2.phoneme.isin(VOWELS)
    arr_v  = arr[mask_v.values]
    dv4    = df_r2[mask_v].reset_index(drop=True)
    intra  = []
    for spk in dv4.speaker_id.unique():
        for ph in VOWELS:
            idx = (dv4.speaker_id==spk)&(dv4.phoneme==ph)
            if idx.sum()<2: continue
            intra.append(cosine_distances(arr_v[idx.values]).mean())
    rope_neu = np.mean(intra)
    print(f"  {model_name} ROPE threshold: {rope_neu:.4f}")
    for _, row in nci_df[nci_df.model==model_name].iterrows():
        lo, hi = row.CI_low, row.CI_high
        cls = "Equivalent" if hi<rope_neu else ("Non-equivalent" if lo>rope_neu else "Indeterminate")
        rope_results.append({"model":model_name,"phoneme":row.phoneme,
                             "obs":row.obs,"CI_low":lo,"CI_high":hi,
                             "ROPE":round(rope_neu,4),"classification":cls})

rope_df = pd.DataFrame(rope_results)
rope_df.to_csv("results/rope_classification.csv", index=False)
print(rope_df[["model","phoneme","obs","CI_low","CI_high","ROPE","classification"]].to_string())

print("\n  Acoustic ROPE (±0.05):")
for _, row in ci_df.iterrows():
    lo, hi = row.CI_low, row.CI_high
    if hi < -ROPE_AC or lo > ROPE_AC: cls = "Non-equivalent"
    elif lo >= -ROPE_AC and hi <= ROPE_AC: cls = "Equivalent"
    else: cls = "Indeterminate"
    print(f"    {row.phoneme} {row.feature}: [{lo:.3f},{hi:.3f}] → {cls}")
print("Section 8 complete.")
