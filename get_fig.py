import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# -------------------------
# 1) Load CSV
# -------------------------
CSV_PATH = "final_summary.csv"  # <- change if needed
df = pd.read_csv(CSV_PATH)

# Safety: ensure correct dtypes
for col in ["bleu", "long_sentence_bleu", "latency_ms"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df["exp"] = df["exp"].astype(str)
df["decoding"] = df["decoding"].astype(str)

# -------------------------
# 2) Add helper columns
# -------------------------
def infer_model_type(exp_name: str) -> str:
    name = exp_name.lower()
    if name.startswith("rnn_"):
        return "RNN"
    if name.startswith("trans_"):
        return "Transformer"
    if "t5" in name:
        return "T5"
    return "Other"

df["model_type"] = df["exp"].apply(infer_model_type)

# Make a nicer experiment label (optional)
# e.g., trans_abs_sin -> abs_sin, rnn_dot -> dot
def short_label(exp_name: str) -> str:
    name = exp_name
    if name.lower().startswith("rnn_"):
        return name[4:]
    if name.lower().startswith("trans_"):
        return name[6:]
    return name

df["exp_short"] = df["exp"].apply(short_label)

# Order experiments within each model group
# (You can customize this order)
ORDER_RNN = ["dot", "add", "mul", "tf_only", "schedule"]
ORDER_TRANS = ["base", "abs_sin", "abs_learn", "rms", "bs256"]
ORDER_T5 = ["finetune"]

def sort_key(row):
    mt = row["model_type"]
    es = row["exp_short"]
    if mt == "RNN":
        return (0, ORDER_RNN.index(es) if es in ORDER_RNN else 999, es)
    if mt == "Transformer":
        return (1, ORDER_TRANS.index(es) if es in ORDER_TRANS else 999, es)
    if mt == "T5":
        return (2, ORDER_T5.index(es) if es in ORDER_T5 else 999, es)
    return (3, 999, es)

df = df.sort_values(by=["model_type", "exp_short"], key=None).copy()
df["sort_tuple"] = df.apply(sort_key, axis=1)
df = df.sort_values("sort_tuple").drop(columns=["sort_tuple"])

# -------------------------
# 3) Plot 1: BLEU bar chart (grouped by decoding)
# -------------------------
# Pivot to have Greedy and Beam-3 side by side
pivot_bleu = df.pivot_table(
    index=["model_type", "exp_short"],
    columns="decoding",
    values="bleu",
    aggfunc="mean"
)

# Keep only common decoding names if present
cols = [c for c in ["Greedy", "Beam-3"] if c in pivot_bleu.columns]
pivot_bleu = pivot_bleu[cols]

# Flatten index for plotting
labels = [f"{mt}:{exp}" for (mt, exp) in pivot_bleu.index]
x = np.arange(len(labels))
width = 0.38 if len(cols) > 1 else 0.6

plt.figure(figsize=(max(12, len(labels) * 0.6), 5))
for i, c in enumerate(cols):
    plt.bar(x + (i - (len(cols)-1)/2)*width, pivot_bleu[c].values, width, label=c)

plt.xticks(x, labels, rotation=45, ha="right")
plt.ylabel("BLEU (as reported in CSV)")
plt.title("BLEU by Experiment and Decoding Strategy")
plt.legend()
plt.tight_layout()
plt.savefig("fig1_bleu_by_experiment.png", dpi=200)
plt.close()

# -------------------------
# 4) Plot 2: Latency bar chart (log scale)
# -------------------------
pivot_lat = df.pivot_table(
    index=["model_type", "exp_short"],
    columns="decoding",
    values="latency_ms",
    aggfunc="mean"
)
cols_lat = [c for c in ["Greedy", "Beam-3"] if c in pivot_lat.columns]
pivot_lat = pivot_lat[cols_lat]

labels_lat = [f"{mt}:{exp}" for (mt, exp) in pivot_lat.index]
x2 = np.arange(len(labels_lat))
width2 = 0.38 if len(cols_lat) > 1 else 0.6

plt.figure(figsize=(max(12, len(labels_lat) * 0.6), 5))
for i, c in enumerate(cols_lat):
    plt.bar(x2 + (i - (len(cols_lat)-1)/2)*width2, pivot_lat[c].values, width2, label=c)

plt.yscale("log")
plt.xticks(x2, labels_lat, rotation=45, ha="right")
plt.ylabel("Latency (ms, log scale)")
plt.title("Inference Latency by Experiment and Decoding Strategy (log scale)")
plt.legend()
plt.tight_layout()
plt.savefig("fig2_latency_logscale.png", dpi=200)
plt.close()

# -------------------------
# 5) Plot 3: BLEU vs Latency scatter (trade-off)
# -------------------------
# Each row is one experiment + decoding
plt.figure(figsize=(7, 5))

# Scatter by model type (no explicit colors to keep default cycling)
for mt, sub in df.groupby("model_type"):
    plt.scatter(sub["latency_ms"], sub["bleu"], label=mt)

plt.xscale("log")
plt.xlabel("Latency (ms, log scale)")
plt.ylabel("BLEU (as reported in CSV)")
plt.title("Quality–Latency Trade-off (BLEU vs Latency)")
plt.legend()
plt.tight_layout()
plt.savefig("fig3_bleu_vs_latency.png", dpi=200)
plt.close()

# -------------------------
# 6) Optional Plot 4: Long-sentence BLEU (grouped bars)
# -------------------------
pivot_long = df.pivot_table(
    index=["model_type", "exp_short"],
    columns="decoding",
    values="long_sentence_bleu",
    aggfunc="mean"
)
cols_long = [c for c in ["Greedy", "Beam-3"] if c in pivot_long.columns]
pivot_long = pivot_long[cols_long]

labels_long = [f"{mt}:{exp}" for (mt, exp) in pivot_long.index]
x3 = np.arange(len(labels_long))
width3 = 0.38 if len(cols_long) > 1 else 0.6

plt.figure(figsize=(max(12, len(labels_long) * 0.6), 5))
for i, c in enumerate(cols_long):
    plt.bar(x3 + (i - (len(cols_long)-1)/2)*width3, pivot_long[c].values, width3, label=c)

plt.xticks(x3, labels_long, rotation=45, ha="right")
plt.ylabel("Long-sentence BLEU (ref length > 20 words)")
plt.title("Long-sentence BLEU by Experiment and Decoding Strategy")
plt.legend()
plt.tight_layout()
plt.savefig("fig4_long_sentence_bleu.png", dpi=200)
plt.close()

print("Saved figures:")
print(" - fig1_bleu_by_experiment.png")
print(" - fig2_latency_logscale.png")
print(" - fig3_bleu_vs_latency.png")
print(" - fig4_long_sentence_bleu.png (optional)")
