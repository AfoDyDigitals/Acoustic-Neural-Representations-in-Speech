import pandas as pd
import numpy as np
import parselmouth
from parselmouth.praat import call
import yaml

params = yaml.safe_load(open("params.yaml"))
MAX_F_M = params["acoustics"]["max_formant_male"]
MAX_F_F = params["acoustics"]["max_formant_female"]
N_FORM  = params["acoustics"]["n_formants"]

df = pd.read_csv("data/processed/phonemes.csv")

def get_formants(row):
    try:
        snd = parselmouth.Sound(row.wav_path)
        snd = snd.extract_part(row.onset, row.offset, preserve_times=True)
        max_f = MAX_F_F if row.gender == "f" else MAX_F_M
        formants = call(snd, "To Formant (burg)", 0, N_FORM, max_f, 0.025, 50)
        mid = (row.onset + row.offset) / 2
        f1 = call(formants, "Get value at time", 1, mid, "hertz", "linear")
        f2 = call(formants, "Get value at time", 2, mid, "hertz", "linear")
        f3 = call(formants, "Get value at time", 3, mid, "hertz", "linear")
        return pd.Series({"F1": f1, "F2": f2, "F3": f3})
    except:
        return pd.Series({"F1": np.nan, "F2": np.nan, "F3": np.nan})

print("Extracting formants...")
df[["F1","F2","F3"]] = df.apply(get_formants, axis=1)
df.to_csv("data/processed/features_acoustic.csv", index=False)

total = len(df)
missing = df.F1.isna().sum()
print(f"Done: {total} tokens, {missing} missing F1 ({missing/total*100:.1f}%)")
