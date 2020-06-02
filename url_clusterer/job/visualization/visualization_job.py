import sys

import numpy as np
from bokeh.io import show
from bokeh.palettes import Turbo256
from bokeh.plotting import figure
from bokeh.transform import linear_cmap
from pyspark.sql import SparkSession
from sklearn.manifold import TSNE

from job.batch_job import BatchJob
from util.url_io import URL_IO


class VisualizationJob(BatchJob):
    app_name: str = "VisualizationJob"
    __clustering_algorithm: str

    def __init__(self, clustering_algorithm):
        self.__clustering_algorithm = clustering_algorithm

    def apply_additional_configs(self, builder: SparkSession.Builder) -> SparkSession.Builder:
        return builder

    def run(self, spark: SparkSession) -> None:
        clustered_urls_and_vectors = URL_IO.read_clustered_urls(spark, self.__clustering_algorithm)
        clustered_urls_and_vectors = clustered_urls_and_vectors.toPandas().sort_values("id")
        labeled_urls = URL_IO.read_labeled_urls().sort_values("id")
        labeled_url_ids = labeled_urls["id"].to_list()
        clustered_url_ids = clustered_urls_and_vectors["id"].to_list()
        clustered_url_cluster_ids = clustered_urls_and_vectors["cluster_id"].to_list()

        labels = labeled_urls["label"].to_list()
        labels_of_clustered_urls = []

        i = 0
        for j in range(len(labeled_url_ids)):
            if clustered_url_ids[i] == labeled_url_ids[j]:
                labels_of_clustered_urls.append(labels[j])
                i += 1

        clustered_urls_and_vectors["label"] = labels_of_clustered_urls

        vectors = np.asarray(list(map(lambda e: e.toArray(), clustered_urls_and_vectors["vector"].to_list())))
        tsne = TSNE()
        vectors_2d = tsne.fit_transform(vectors)

        clustered_urls_and_vectors['x'] = list(map(lambda e: e[0], vectors_2d))
        clustered_urls_and_vectors['y'] = list(map(lambda e: e[1], vectors_2d))

        clustered_urls_and_vectors = clustered_urls_and_vectors[["x", "y", "id", "label", "cluster_id"]]

        tooltips = [
            ('url', '@url'),
            ('label', '@label'),
            ('cluster id', '@cluster_id')
        ]
        colors = linear_cmap(field_name='cluster_id', palette=Turbo256, low=min(clustered_url_cluster_ids),
                             high=max(clustered_url_cluster_ids))
        p = figure(plot_width=1300, plot_height=700, tooltips=tooltips)
        p.circle(x="x", y="y", source=clustered_urls_and_vectors, size=7, color=colors, fill_alpha=1)
        show(p)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("kmeans or bisecting_kmeans must be passed to specify which clustering results to be visualized.")
    else:
        VisualizationJob(sys.argv[1]).start()
