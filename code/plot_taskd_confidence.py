"""Plot Task D confidence for exact, lenient-only and incorrect predictions by condition."""
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("manifests/task_d_predictions.csv", encoding="latin1")
# Query ID looks like "taskD-D1-carpet"; the "Condition" column is the ground-truth label.
df["prompt"] = df["Query ID"].str.split("-").str[1]
df["material"] = df["Query ID"].str.split("-").str[2]
pred = df["Predicted Label"].str.strip().str.lower()
truth = df["Condition"].str.strip().str.lower()
exact = pred == truth
lenient = [t in p for p, t in zip(pred, truth)]
df["outcome"] = "incorrect"
df.loc[lenient, "outcome"] = "lenient"
df.loc[exact, "outcome"] = "exact"

conditions = ["D1", "D2", "D3"]
# outcome: (color, marker, legend label, x offset)
styles = {
    "exact": ("tab:green", "o", "Exact label", -0.18),
    "lenient": ("gold", "o", "Lenient label (not exact)", 0.0),
    "incorrect": ("tab:red", "X", "Incorrect", 0.18),
}

fig, ax = plt.subplots(figsize=(7.5, 4.5))
for outcome, (color, marker, label, offset) in styles.items():
    sub = df[df["outcome"] == outcome]
    x = sub["prompt"].map(conditions.index) + offset
    ax.scatter(x, sub["Confidence"], c=color, marker=marker, s=35, label=label,
               edgecolor="black", linewidth=0.5, zorder=3)
    # one label per point; materials sharing the same confidence are joined
    for (cond, conf), names in sub.groupby(["prompt", "Confidence"])["material"]:
        ax.annotate(", ".join(names), (conditions.index(cond) + offset, conf),
                    textcoords="offset points", xytext=(6, -3), fontsize=7)
    # mean confidence per condition for this group
    for i, cond in enumerate(conditions):
        vals = sub.loc[sub["prompt"] == cond, "Confidence"]
        if len(vals):
            ax.hlines(vals.mean(), i + offset - 0.06, i + offset + 0.06,
                      colors=color, linewidth=2)

ax.set_xticks(range(len(conditions)))
ax.set_xticklabels(["D1 generic zero-shot", "D2 named definitions", "D3 + support examples"])
ax.set_ylabel("Model confidence (0-100)")
ax.set_ylim(60, 100)
ax.set_title("Task D: confidence of correct vs incorrect predictions")
ax.grid(axis="y", alpha=0.3)
ax.legend(loc="lower left", fontsize=8)
fig.tight_layout()
fig.savefig("taskd_confidence_by_condition.png", dpi=150)
print(df.groupby(["prompt", "outcome"])["Confidence"].agg(["count", "mean"]))
