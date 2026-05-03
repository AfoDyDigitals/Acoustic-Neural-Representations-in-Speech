import os, glob, pandas as pd
import tgt

RAW = "data/raw/ru-fr_interference/2"
OUT = "data/processed/phonemes.csv"

meta = pd.read_csv(f"{RAW}/metadata_RUFR.csv", sep=";")
meta.columns = meta.columns.str.strip()
spk_info = meta.set_index("spk")[["L1","Gender"]].to_dict("index")

rows = []
for tg_path in glob.glob(f"{RAW}/wav_et_textgrids/FRcorp_textgrids_only/**/*.TextGrid", recursive=True):
    fname = os.path.basename(tg_path)
    parts = fname.replace(".TextGrid","").split("_")
    spk = parts[0].upper()
    sent_id = parts[-1]
    wav_path = tg_path.replace(".TextGrid", ".wav")
    if spk not in spk_info or not os.path.exists(wav_path):
        continue
    l1 = spk_info[spk]["L1"]
    gender = spk_info[spk]["Gender"]
    tg = tgt.io.read_textgrid(tg_path)
    tier = tg.get_tier_by_name("phones")
    for interval in tier.intervals:
        label = interval.text.strip()
        if not label or label in ["", "sil", "sp", "SIL", "<p:>"]:
            continue
        rows.append({
            "speaker_id": spk, "sent_id": sent_id, "rep_index": 0,
            "phoneme": label, "onset": interval.start_time,
            "offset": interval.end_time,
            "duration": (interval.end_time - interval.start_time) * 1000,
            "l1_status": l1, "gender": gender, "wav_path": wav_path
        })

pd.DataFrame(rows).to_csv(OUT, index=False)
print(f"Done: {len(rows)} phoneme tokens saved to {OUT}")
