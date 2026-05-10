import os

root = "./"

delete_suffixes = ("_cropped.png", "_rescaled.png", "_overlay.png", "_cropped_for_thesis.png")

deleted = 0

for root_dir, _, files in os.walk(root):
    for file in files:
        if file.endswith(delete_suffixes):
            path = os.path.join(root_dir, file)
            os.remove(path)
            print("deleted:", path)
            deleted += 1

print(f"\nDone. Total deleted: {deleted}")