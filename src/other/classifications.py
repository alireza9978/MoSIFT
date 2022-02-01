import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import chi2_kernel
from functools import partial

if __name__ == '__main__':
    df = pd.read_csv("../../dataset/final_features.csv", header=None)
    label = df[1]
    df = df.drop(columns=[0, 1])

    best = -1
    best_matrix = None

    result = []
    for i in range(10):

        X_train, X_test, y_train, y_test = train_test_split(df, label, test_size=0.20, random_state=i)

        # scaler = StandardScaler()
        # scaler.fit(X_train)  # Don't cheat - fit only on training data
        # X_train = scaler.transform(X_train)
        # X_test = scaler.transform(X_test)

        # model = make_pipeline(StandardScaler(), OneVsRestClassifier(SVC()))
        # model = RandomForestClassifier()
        my_chi2_kernel = partial(chi2_kernel, gamma=1)
        model = SVC(kernel=my_chi2_kernel)
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
