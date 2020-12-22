import pandas as pd
from sklearn import model_selection

if __name__ == "__main__":
    df = pd.read_csv("input/train.csv")
    df["kfold"] = -1

    df.sample(frac=1).reset_index(drop=True)

    kf = model_selection.StratifiedKFoldn_splits=5, shuffle=False, random_state=42)
