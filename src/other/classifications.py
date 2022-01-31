# from sklearn.linear_model import SVM
import pandas as pd
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import NearestNeighbors, NearestCentroid

if __name__ == '__main__':
    df = pd.read_csv("../../dataset/final_features.csv", header=None)
    label = df[1]
    df = df.drop(columns=[0, 1])

    result = []
    for i in range(100):
        X_train, X_test, y_train, y_test = train_test_split(df, label, test_size=0.20, random_state=i)

        # scaler = StandardScaler()
        # scaler.fit(X_train)  # Don't cheat - fit only on training data
        # X_train = scaler.transform(X_train)
        # X_test = scaler.transform(X_test)

        # model = make_pipeline(StandardScaler(), OneVsRestClassifier(SVC()))
        model =RandomForestClassifier()
        # model = SGDClassifier()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(acc, i)
        result.append(acc)
        print(confusion_matrix(y_test, y_pred))

    print(np.array(result).mean())
