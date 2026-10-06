"""Render the Task D prompt-sensitivity variants as a table image."""
import textwrap

import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("manifests/task_d_sensitivity.csv", encoding="cp1252")
df = df.rename(columns={"Evidence": "Share Link"})


def fill(text, width):
    """Wrap each line separately so the prompt keeps its own line breaks."""
    lines = [ln.strip() for ln in str(text).splitlines() if ln.strip()]
    return "\n".join(textwrap.fill(ln, width) for ln in lines)


# column: wrap width in characters (None = no wrapping)
wrap = {"Query ID": 22, "Condition": None, "Predicted Label": None, "Decision": None,
        "Confidence": None, "Visible Evidence": 50, "Exact Prompt": 80, "Share Link": 32}
cells = [[fill(v, w) if w else str(v) for v, w in zip(row, wrap.values())]
         for row in df.itertuples(index=False)]
headers = [h.replace(" ", "\n") if h in ("Predicted Label", "Visible Evidence") else h
           for h in df.columns]
col_widths = [0.09, 0.05, 0.05, 0.06, 0.06, 0.20, 0.32, 0.17]

# every cell in a row gets the height of the tallest cell in that row
row_lines = [2] + [max(v.count("\n") + 1 for v in row) for row in cells]
line_h = 0.22  # inches per text line
fig_h = line_h * sum(n + 0.8 for n in row_lines)
fig, ax = plt.subplots(figsize=(26, fig_h))
ax.axis("off")
table = ax.table(cellText=cells, colLabels=headers, colWidths=col_widths,
                 bbox=[0, 0, 1, 1], cellLoc="left")
table.auto_set_font_size(False)
table.set_fontsize(9)

total = sum(n + 0.8 for n in row_lines)
for (r, c), cell in table.get_celld().items():
    cell.set_height((row_lines[r] + 0.8) / total)
    cell.PAD = 0.02
    if r == 0:
        cell.set_facecolor("#d9d9d9")
        cell.get_text().set_fontweight("bold")
        cell.get_text().set_horizontalalignment("center")
    elif (r - 1) // 4 % 2 == 0:  # shade alternate material groups of 4 prompt variants
        cell.set_facecolor("#f2f2f2")
    if r > 0 and c == df.columns.get_loc("Predicted Label"):
        ok = df.iloc[r - 1]["Predicted Label"] == df.iloc[r - 1]["Condition"]
        cell.get_text().set_color("tab:green" if ok else "tab:red")

ax.set_title("Task D: prompt sensitivity across D2 prompt variants",
             fontsize=13, fontweight="bold", pad=12)
plt.savefig("taskd_sensitivity_table.png", dpi=200, bbox_inches="tight")
print("saved taskd_sensitivity_table.png")
