import pandas as pd
import os
import glob

# glob.glob(path2)[5].split("\\")[-1]
print(glob.glob("data/data_aug/*"))


for folder1 in glob.glob("data/data_aug/*"):
    images = []
    labels = []

    path = folder1 + "/*"
    for folder2 in glob.glob(path):
        path2 = folder2 + "/*"
        for file in glob.glob(path2):
            img = file.replace(r"\\", "/")
            images.append(file)
            if file.split("/")[-2] == "cancer":
                labels.append(1)
            elif file.split("/")[-2] == "normal":
                labels.append(0)

    df = pd.DataFrame({"image":images,
                  "label": labels})
    df.to_csv(f"{folder1}/info.csv", index=False, quoting=1)