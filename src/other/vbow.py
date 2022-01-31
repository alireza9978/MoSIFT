import multiprocessing

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import euclidean_distances

from src.load_dataset import load_all_features

g = 0.1


def distance_cost(row):
    from_pt_x, from_pt_y, from_pt_f, to_pt_x, to_pt_y, to_pt_f = row.values
    point_a, point_b = np.array([[from_pt_x, from_pt_y, from_pt_f]]), np.array([[to_pt_x, to_pt_y, to_pt_f]])
    distance = euclidean_distances(point_a, point_b)
    return np.exp(-g * distance)[0][0]


def gen_histogram(temp_df: pd.DataFrame):
    histogram = np.zeros(500)
    for k in temp_df["prediction"]:
        histogram[k] += 1.0
    return pd.Series(histogram)


def get_adjacency_matrix(temp_df: pd.DataFrame):
    temp_labels = temp_df["word_label"]
    temp_labels = temp_labels.reset_index(drop=True)
    points = temp_df.drop(columns=["video", "category", "word_label"])
    points = points.reset_index(drop=True)
    result = pairwise_distances(points, metric=word_neighbor, n_jobs=1)
    result[np.eye(result.shape[0], dtype=bool)] = 0
    y, x = np.where(result == 1)
    result_df = pd.DataFrame(
        {"from": temp_labels[x].values, "to": temp_labels[y].values, "from_pt_x": points.loc[x, "x"].values,
         "from_pt_y": points.loc[x, "y"].values, "from_pt_f": points.loc[x, "frame"].values,
         "to_pt_x": points.loc[y, "x"].values, "to_pt_y": points.loc[y, "y"].values,
         "to_pt_f": points.loc[y, "frame"].values})
    result_df["video"] = temp_df["video"].values[0]
    result_df["category"] = temp_df["category"].values[0]
    return result_df


def word_neighbor(x, y):
    temp_d = x - y
    if np.abs(temp_d[0]) <= 5 and np.abs(temp_d[1]) <= 5 and 0 <= temp_d[2] <= 60:
        return 1
    else:
        return 0


def vectorise(temp_list):
    result = temp_list[["from", "to"]].groupby(["from", "to"]).size().reset_index()
    return result[["from", "to"]]


def inner_generate_bi_gram_histogram(temp_df: pd.DataFrame):
    return temp_df[["from-to", "weight"]].groupby(["from-to"]).sum()


def generate_bi_gram_histogram(temp_spatio_temporal_df):
    adjacency_matrix = apply_parallel(temp_spatio_temporal_df.groupby(["video", "category"]), get_adjacency_matrix)
    tf = adjacency_matrix.groupby(["from", "to"]).size()

    df = apply_parallel(adjacency_matrix.groupby(["video", "category"]), vectorise)
    df = df.groupby(["from", "to"]).size()
    tf_idf_matrix = adjacency_matrix.set_index(["from", "to"])
    tf_idf_matrix["tf"] = tf
    tf_idf_matrix = tf_idf_matrix.reset_index().set_index(["from", "to"])
    tf_idf_matrix["df"] = df
    tf_idf_matrix = tf_idf_matrix.reset_index()
    n = adjacency_matrix.groupby(["video", "category"]).size().shape[0]
    tf_idf_matrix["tf_idf"] = tf_idf_matrix["tf"] * np.log(n / tf_idf_matrix["df"])

    grouped_tf_idf_matrix = tf_idf_matrix[["from", "to", "tf_idf"]].groupby(["from", "to"]).first().reset_index()
    grouped_tf_idf_matrix = grouped_tf_idf_matrix.sort_values("tf_idf", ascending=False)
    grouped_tf_idf_matrix = grouped_tf_idf_matrix.iloc[:200][['from', "to"]].reset_index(drop=True)
    grouped_tf_idf_matrix["from-to"] = grouped_tf_idf_matrix["from"].astype(str) + "-" + grouped_tf_idf_matrix[
        "to"].astype(str)
    adjacency_matrix["from-to"] = adjacency_matrix["from"].astype(str) + "-" + adjacency_matrix["to"].astype(str)
    adjacency_matrix = adjacency_matrix[adjacency_matrix["from-to"].isin(grouped_tf_idf_matrix["from-to"])]
    adjacency_matrix["weight"] = adjacency_matrix[['from_pt_x', 'from_pt_y', 'from_pt_f',
                                                   'to_pt_x', 'to_pt_y', 'to_pt_f']].apply(distance_cost, axis=1)
    temp_bi_gram_histogram = adjacency_matrix.groupby(["video", "category"]).apply(inner_generate_bi_gram_histogram)
    temp_bi_gram_histogram = temp_bi_gram_histogram.reset_index("from-to")
    temp_bi_gram_histogram = pd.pivot_table(temp_bi_gram_histogram, values=['weight'], index=["video", "category"],
                                            columns=["from-to"], aggfunc=np.sum).fillna(0)

    return temp_bi_gram_histogram


def apply_parallel(data_frame_grouped, func):
    result_list = Parallel(n_jobs=int(multiprocessing.cpu_count() - 5))(
        delayed(func)(group) for name, group in data_frame_grouped)
    return pd.concat(result_list)


if __name__ == '__main__':
    spatio_temporal_columns = ["x", "y", "frame", "video", "category"]

    data_frame, labels = load_all_features()
    data_frame.rename(columns={256: "x", 257: "y", 258: "frame"}, inplace=True)
    spatio_temporal_df = data_frame[spatio_temporal_columns]
    data_frame.drop(columns=spatio_temporal_columns, inplace=True)
    data_frame.fillna(data_frame.mean(), inplace=True)

    print("# data loaded")

    clustering_model = MiniBatchKMeans(n_clusters=500, batch_size=32)
    clustering_model.fit(data_frame)
    word_label = clustering_model.predict(data_frame)
    np.savetxt('../../dataset/word_label.csv', word_label, delimiter=',')
    # word_label = np.loadtxt('../../dataset/word_label.csv', delimiter=',').astype(int)
    print("# clustered")

    spatio_temporal_df = spatio_temporal_df.reset_index(drop=True)
    spatio_temporal_df["prediction"] = pd.Series(word_label, index=spatio_temporal_df.index)
    vector_bag_of_word_histogram = spatio_temporal_df[["video", "category", "prediction"]].groupby(
        ["video", "category"]).apply(gen_histogram)

    print("# Histogram of features generated")

    spatio_temporal_df.loc[:, "word_label"] = word_label
    bi_gram_histogram = generate_bi_gram_histogram(spatio_temporal_df)

    print("# Histogram of bi-gram words")

    vector_bag_of_word_histogram.index.rename(["video", "category"], inplace=True)
    final_df = vector_bag_of_word_histogram.join(bi_gram_histogram).fillna(0)
    final_df.reset_index().to_csv("../../dataset/final_features.csv", header=None, index=None)
