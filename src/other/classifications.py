# from sklearn.linear_model import SVM
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import NearestNeighbors, NearestCentroid

if __name__ == '__main__':
    df = pd.read_csv("../../dataset/final_features.csv", header=None)
    label = df[1]
    df = df.drop(columns=[0, 1])

    X_train, X_test, y_train, y_test = train_test_split(df, label, test_size=0.15, random_state=42)

    # scaler = StandardScaler()
    # scaler.fit(X_train)  # Don't cheat - fit only on training data
    # X_train = scaler.transform(X_train)
    # X_test = scaler.transform(X_test)

    # model = make_pipeline(StandardScaler(), SVC())
    model = SVC()
    # model = SGDClassifier(loss="log")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(accuracy_score(y_test, y_pred))
