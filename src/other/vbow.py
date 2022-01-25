from sklearn.cluster import MiniBatchKMeans
import numpy as np
import pandas as pd
import glob
import os

from src.load_dataset import load_all_features

cols = []
for i in range(500):
    cols.append("feature" + str(i))
cols.append("label")


def clustering(input_data, n_clusters, batch_size):
    model = MiniBatchKMeans(n_clusters=n_clusters, batch_size=batch_size).fit(input_data)
    return model


def gen_histogram(feature_vectors, kmeans_model):
    histogram = np.zeros(len(kmeans_model.cluster_centers_))
    cluster_result = kmeans_model.predict(feature_vectors)
    for i in cluster_result:
        histogram[i] += 1.0
    return histogram


def gen_vbow(input_path, output_name, kmeans_model, class_id):
    vbow = []
    listing = glob.glob(input_path + "*.csv")
    for csv_name in listing:
        try:
            feature_vectors = pd.read_csv(input_path + csv_name.split('\\')[1])
            histogram = gen_histogram(feature_vectors, kmeans_model)
            histogram = np.append(histogram, class_id)
            vbow.append(histogram)
        except:
            print('Exception in ' + input_path + csv_name)
    df = pd.DataFrame(vbow, columns=cols)
    df.to_csv(output_name + listing[0].split('/')[2].split('\\')[0] + ".csv", index=False)
    # np.savetxt(output_name+".csv", vbow, delimiter=",")


if __name__ == '__main__':
    data_frame = load_all_features()
    data_frame.fillna(data_frame.mean(), inplace=True)
    kmeans_model = clustering(data_frame, 500, 32)
    print("# clustering ended")

