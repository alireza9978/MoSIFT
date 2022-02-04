import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import chi2_kernel
from functools import partial

if __name__ == '__main__':
    mosift_df = pd.read_csv("../../dataset/final_features.csv", header=None)
    mosift_df.rename(columns={0: "video", 1: "category"}, inplace=True)
    mosift_df = mosift_df.sort_values(["category", "video"])
    mosift_df["video"] = np.arange(mosift_df.shape[0])
    mosift_df.drop(columns=["category"], inplace=True)
    mosift_df.set_index("video", inplace=True)

    df = pd.read_csv("../../dataset/main_final_features.csv")
    df.set_index(['video'], inplace=True)

    df = df.join(mosift_df).reset_index()
    label = df["category"]
    df = df.drop(columns=['video', "category"])
    df = df.values

    best = -1
    best_matrix = None

    result = []
    for i in range(10):

        X_train, X_test, y_train, y_test = train_test_split(df, label, test_size=0.20, random_state=i)

        # scaler = StandardScaler()
        # scaler.fit(X_train)  # Don't cheat - fit only on training data
        # X_train = scaler.transform(X_train)
        # X_test = scaler.transform(X_test)

        # model = make_pipeline(StandardScaler(), OneVsRestClassifier(RandomForestClassifier()))
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        matrix = confusion_matrix(y_test, y_pred)
        if acc > best:
            best = acc
            best_matrix = matrix
        result.append(acc)

    print(np.array(result).mean())
    if best != -1 and best_matrix is not None:
        print("best", best)
        print(best_matrix)
