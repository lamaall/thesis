import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------
# LOAD DATA
# ----------------------------
df = pd.read_csv("results_log.txt")

df.columns = [
    "category",
    "filename",
    "iou",
    "chamfer",
    "fp",
    "fn"
]

categories = df["category"].unique()

iou_data = [df[df["category"] == c]["iou"].values for c in categories]
chamfer_data = [df[df["category"] == c]["chamfer"].values for c in categories]

iou_mean = df["iou"].mean()
chamfer_mean = df["chamfer"].mean()

# ============================================================
# IOU PLOT (HIGHER IS BETTER)
# ============================================================
plt.figure(figsize=(8, 6))

plt.boxplot(iou_data, labels=categories)
plt.title("IoU per Category")
plt.ylabel("IoU (higher is better)")
plt.xlabel("Category")

plt.axhline(iou_mean, color="red", linestyle="--", linewidth=2)

ymin, ymax = plt.ylim()

plt.annotate(
    "Better",
    xy=(1, ymin + (ymax - ymin) * 0.15),
    xytext=(1, ymin + (ymax - ymin) * 0.05),
    arrowprops=dict(arrowstyle="->", color="blue", lw=2),
    ha="center",
    color="blue"
)

plt.tight_layout()
plt.savefig("iou_boxplot.png", dpi=300)
plt.close()

# ============================================================
# CHAMFER PLOT (LOWER IS BETTER)
# ============================================================
plt.figure(figsize=(8, 6))

plt.boxplot(chamfer_data, labels=categories)
plt.title("Chamfer Distance per Category")
plt.ylabel("Chamfer Distance (lower is better)")
plt.xlabel("Category")

plt.axhline(chamfer_mean, color="red", linestyle="--", linewidth=2)

ymin, ymax = plt.ylim()

plt.annotate(
    "Better",
    xy=(1, ymin + (ymax - ymin) * 0.85),
    xytext=(1, ymin + (ymax - ymin) * 0.95),
    arrowprops=dict(arrowstyle="->", color="blue", lw=2),
    ha="center",
    color="blue"
)

plt.tight_layout()
plt.savefig("chamfer_boxplot.png", dpi=300)
plt.close()