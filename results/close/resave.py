import os
from PIL import Image

ROOT = "./"  # start folder

for root, dirs, files in os.walk(ROOT):
    for file in files:
        if file.lower().endswith(".ps"):
            ps_path = os.path.join(root, file)
            png_path = os.path.splitext(ps_path)[0] + ".png"

            try:
                with Image.open(ps_path) as img:
                    img.save(png_path)

                print(f"✔ Saved: {png_path}")

            except Exception as e:
                print(f"✘ Failed: {ps_path} -> {e}")