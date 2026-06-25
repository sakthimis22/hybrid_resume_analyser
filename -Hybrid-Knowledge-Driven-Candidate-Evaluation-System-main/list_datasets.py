import os

RAW_DATASET_PATH = "datasets/raw"

print("📂 Datasets found in raw folder:\n")

for root, dirs, files in os.walk(RAW_DATASET_PATH):
    for file in files:
        if file.endswith(".csv"):
            print(os.path.join(root, file))
