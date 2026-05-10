import os
import pandas as pd
import re

# ----------------------------
# PARSE PROFILE FILE
# ----------------------------
def parse_profile(path):
    with open(path, "r") as f:
        text = f.read()

    def extract(stage):
        match = re.search(rf"{stage}.*?time\s*:\s*([0-9.]+)", text, re.S)
        return float(match.group(1)) if match else 0.0

    total = re.search(r"TOTAL TIME:\s*([0-9.]+)", text)
    total = float(total.group(1)) if total else 0.0

    detection = extract("whiteboard_detection")
    graph = extract("graph_construction")
    planning = extract("path_planning")
    execution = extract("robot_execution")

    return {
        "detection": detection,
        "graph": graph,
        "planning": planning,
        "execution": execution,
    }

# ----------------------------
# LOAD DATA
# ----------------------------
root = "./"
data = []

for r, _, files in os.walk(root):
    for f in files:
        if f.endswith("_profile.txt"):
            path = os.path.join(r, f)
            name = f.replace("_profile.txt", "")

            parsed = parse_profile(path)
            parsed["filename"] = name
            data.append(parsed)

df = pd.DataFrame(data)

# ----------------------------
# CATEGORY ASSIGNMENT
# ----------------------------
def get_category(filename):
    for r, _, files in os.walk(root):
        for f in files:
            if filename in f and "_cropped" in f:
                return os.path.basename(os.path.dirname(os.path.join(r, f)))
    return "unknown"

df["category"] = df["filename"].apply(get_category)

# ----------------------------
# COMPUTE MEAN ± STD
# ----------------------------
mean = df.groupby("category").mean(numeric_only=True)
std = df.groupby("category").std(numeric_only=True)

# ----------------------------
# SAVE 4 FIGURES (EQUAL WIDTH TABLES)
# ----------------------------
output_file = "runtime_figures.tex"

stages = ["detection", "graph", "planning", "execution"]
categories = list(mean.index)
chunk_size = 5

with open(output_file, "w", encoding="utf-8") as f:

    for stage in stages:

        f.write("\\begin{figure}[h]\n")
        f.write("\\centering\n")
        f.write("\\footnotesize\n")

        f.write(f"\\textbf{{{stage.capitalize()}}}\\\\[0.8em]\n")

        chunks = [categories[i:i + chunk_size] for i in range(0, len(categories), chunk_size)]

        for chunk in chunks:

            # FIXED WIDTH TABLE (KEY IMPROVEMENT)
            f.write("\\resizebox{\\textwidth}{!}{%\n")

            cols = "l" + "c" * len(chunk)
            f.write(f"\\begin{{tabular}}{{{cols}}}\n")

            f.write("\\toprule\n")

            # header
            header = "Drawing"
            for cat in chunk:
                header += f" & {cat}"
            header += " \\\\\n"

            f.write(header)
            f.write("\\midrule\n")

            # row
            row = "Time [s]"
            for cat in chunk:
                m = mean.loc[cat, stage]
                s = std.loc[cat, stage]
                row += f" & ${m:.3f} \\pm {s:.3f}$"
            row += " \\\\\n"

            f.write(row)
            f.write("\\bottomrule\n")
            f.write("\\end{tabular}%\n")
            f.write("}\n")  # close resizebox

            f.write("\\vspace{0.8em}\n")

        f.write(f"\\caption{{Runtime for {stage} stage across drawing categories (mean $\\pm$ std in seconds).}}\n")
        f.write(f"\\label{{fig:{stage}_runtime}}\n")
        f.write("\\end{figure}\n\n")

print(f"Saved aligned LaTeX figures to {output_file}")