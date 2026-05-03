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
