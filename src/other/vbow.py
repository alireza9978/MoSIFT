import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import pairwise_distances

from src.load_dataset import load_all_features


def clustering(input_data, n_clusters, batch_size):
    model = MiniBatchKMeans(n_clusters=n_clusters, batch_size=batch_size).fit(input_data)
    return model


def gen_histogram(temp_df: pd.DataFrame):
    histogram = np.zeros(500)
    for k in temp_df["prediction"]:
        histogram[k] += 1.0
    return pd.Series(histogram)


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
    temp_d = x - y
    if np.abs(temp_d[0]) <= 5 and np.abs(temp_d[1]) <= 5 and 0 <= temp_d[2] <= 60:
        return 1
    else:
        return 0


def vectorise(temp_list):
    result = temp_list[["from", "to"]].groupby(["from", "to"]).size().reset_index()
    return result[["from", "to"]]


def generate_bi_gram_histogram(temp_df: pd.DataFrame):
    return temp_df.groupby(["from-to"]).size()


if __name__ == '__main__':
    data_frame, labels = load_all_features()
    spatio_temporal_df = data_frame[[256, 257, 258, 259, 260]]
    data_frame.drop(columns=[256, 257, 258, 259, 260], inplace=True)
    data_frame.fillna(data_frame.mean(), inplace=True)
    kmeans_model = clustering(data_frame, 500, 32)
    word_label = kmeans_model.predict(data_frame)

    print("# clustering ended")

    spatio_temporal_df["prediction"] = word_label
    vector_bag_of_word_histogram = spatio_temporal_df[[259, 260, "prediction"]].groupby(
        [259, 260]).apply(gen_histogram)
    vector_bag_of_word_histogram.index = vector_bag_of_word_histogram.index.rename({259: "video", 260: "category"})

    print("# Histogram of features generated")

    spatio_temporal_df.loc[:, "word_label"] = word_label
    matrix = spatio_temporal_df.groupby([259, 260], group_keys=False).apply(get_adjacency_matrix)
    tf = matrix.groupby(["from", "to"]).size()

    df = matrix.groupby(["video", "category"]).apply(vectorise)
    df = df.reset_index().drop(columns=["level_2"])
    df = df.groupby(["from", "to"]).size()
    tf_idf_matrix = matrix.set_index(["from", "to"])
    tf_idf_matrix["tf"] = tf
    tf_idf_matrix = tf_idf_matrix.reset_index().set_index(["from", "to", "video", "category"])
    tf_idf_matrix["df"] = df
    tf_idf_matrix = tf_idf_matrix.reset_index()
    tf_idf_matrix["tf_idf"] = tf_idf_matrix["tf"] / tf_idf_matrix["df"]
    tf_idf_matrix = tf_idf_matrix[["from", "to", "tf_idf"]].groupby(["from", "to"]).first().reset_index()
    tf_idf_matrix = tf_idf_matrix.sort_values("tf_idf", ascending=False)
    tf_idf_matrix = tf_idf_matrix.iloc[:200][['from', "to"]].reset_index(drop=True)
    tf_idf_matrix["from-to"] = tf_idf_matrix["from"].astype(str) + "-" + tf_idf_matrix["to"].astype(str)
    matrix["from-to"] = matrix["from"].astype(str) + "-" + matrix["to"].astype(str)

    bi_gram_histogram = matrix[matrix["from-to"].isin(tf_idf_matrix["from-to"])].groupby(["video", "category"]).apply(
        generate_bi_gram_histogram)
    bi_gram_histogram = bi_gram_histogram.reset_index("from-to")
    bi_gram_histogram = pd.pivot_table(bi_gram_histogram, values=[0], index=["video", "category"], columns=["from-to"],
                                       aggfunc=np.sum).fillna(0)

    final_df = vector_bag_of_word_histogram.join(bi_gram_histogram).fillna(0)

    final_df.reset_index().to_csv("../../dataset/final_features.csv", header=None, index=None)
    # Parallel(n_jobs=int(multiprocessing.cpu_count()))(
    #     delayed(get_adjacency_matrix)(group) for group, name in spatio_temporal_df.groupby(columns=[259]))
