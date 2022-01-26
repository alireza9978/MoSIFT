import multiprocessing

from joblib import Parallel, delayed
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import pairwise_distances
import numpy as np
import pandas as pd
import glob
import math
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


def get_adjacency_matrix(temp_df: pd.DataFrame):
    temp_labels = temp_df["word_label"]
    result = pairwise_distances(temp_df.drop(columns=[259, 260, "word_label"]), metric=word_neighbor, n_jobs=1)
    result = result[~np.eye(result.shape[0], dtype=bool)].reshape(result.shape[0], -1)
    x, y = np.where(result == 1)
    result_df = pd.DataFrame({"from": temp_labels[x].values, "to": temp_labels[y].values})
    result_df["video"] = temp_df[259].values[0]
    result_df["category"] = temp_df[260].values[0]
    return result_df


def word_neighbor(x, y):
    temp_d = np.abs(x - y)
    if temp_d[0] < 5 and temp_d[1] < 5 and temp_d[2] < 60:
        return 1
    else:
        return 0


if __name__ == '__main__':
    data_frame, labels = load_all_features()
    spatio_temporal_df = data_frame[[256, 257, 258, 259, 260]]
    data_frame.drop(columns=[256, 257, 258, 259, 260], inplace=True)
    data_frame.fillna(data_frame.mean(), inplace=True)
    kmeans_model = clustering(data_frame, 500, 32)
    word_label = kmeans_model.predict(data_frame)
    print("# clustering ended")
    spatio_temporal_df.loc[:, "word_label"] = word_label
    matrix = spatio_temporal_df.groupby([259, 260], group_keys=False).apply(get_adjacency_matrix)
    tf = matrix.groupby(["from", "to"]).count()
    idf = matrix.groupby(["from", "to", "video", "category"]).sum()
    print(matrix)
    # Parallel(n_jobs=int(multiprocessing.cpu_count()))(
    #     delayed(get_adjacency_matrix)(group) for group, name in spatio_temporal_df.groupby(columns=[259]))
