import logging
import sys
from collections import defaultdict
from functools import partial
from itertools import count

from pyspark.sql import SparkSession
from sklearn.metrics import v_measure_score

from job.batch_job import BatchJob
from util.url_io import URL_IO


class ClusterEvaluatorJob(BatchJob):
    def __init__(self, clustering_algorithm):
        self.__clustering_algorithm = clustering_algorithm

    app_name: str = "ClusterEvaluatorJob"
    __clustering_algorithm: str

    def apply_additional_configs(self, builder: SparkSession.Builder) -> SparkSession.Builder:
        return builder

    def run(self, spark: SparkSession) -> None:
        logging.getLogger("ClusterEvaluatorJob.run()").info("started")
        clustered_urls_and_vectors = URL_IO.read_clustered_urls(spark,
                                                                self.__clustering_algorithm)\
            .toPandas().sort_values("id")
        labeled_urls = URL_IO.read_labeled_urls().sort_values("id")
        labeled_url_ids = labeled_urls["id"].to_list()
        clustered_url_ids = clustered_urls_and_vectors["id"].to_list()
        clustered_url_cluster_ids = clustered_urls_and_vectors["cluster_id"].to_list()
        cluster_ids = []

        j = 0
        for i in range(len(labeled_url_ids)):
            if labeled_url_ids[i] == clustered_url_ids[j]:
                cluster_ids.append(clustered_url_cluster_ids[j])
                j += 1
            else:
                cluster_ids.append(-1)

        labels = labeled_urls["label"].to_list()
        label_ids = defaultdict(partial(next, count(1)))

        label_ids = [(label_ids[label], label) for label in labels]
        label_ids = list(map(lambda e: e[0], label_ids))

        v_measure = v_measure_score(label_ids, cluster_ids)
        print("v-measure-score: ", v_measure)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("kmeans or bisecting_kmeans must be passed to specify which clustering results to be evaluated.")
    else:
        ClusterEvaluatorJob(sys.argv[1]).start()
