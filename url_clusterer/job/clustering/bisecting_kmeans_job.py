from typing import Callable

from pyspark.ml.clustering import BisectingKMeans, KMeansModel
from pyspark.sql import SparkSession, DataFrame

from config.config_loader import get_configs
from job.batch_job import BatchJob
from util.url_feature_extractor import URLFeatureExtractor
from util.url_io import URL_IO


class BisectingKMeansJob(BatchJob):
    def __init__(self, k: int, distance_measure: str, s: float, window_size: int,
                 additional_weight_function: Callable[[int], float] = lambda e: 1):
        self.__k = k
        self.__distance_measure = distance_measure
        self.__s = s
        self.__window_size = window_size
        self.__additional_weight_function = additional_weight_function

    app_name: str = "KMeansJob"

    def apply_additional_configs(self, builder: SparkSession.Builder) -> SparkSession.Builder:
        return builder

    def __additional_weight_function(self, position):
        return 1

    def run(self, spark: SparkSession) -> None:
        urls_and_vectors = URLFeatureExtractor \
            .get_urls_and_vectors(spark, self.__window_size, self.__s, self.__additional_weight_function)
        clustered_url_vectors = self.get_clusters(urls_and_vectors)
        URL_IO.write_clustered_urls(clustered_url_vectors.select("id", "url", "cluster_id", "vector"), "kmeans")

    def get_clusters(self, urls_and_vectors: DataFrame) -> DataFrame:
        model = BisectingKMeansJob.get_model(urls_and_vectors, self.__k, self.__distance_measure)
        clustered_url_vectors = model.transform(urls_and_vectors)
        return clustered_url_vectors

    @staticmethod
    def get_model(urls_and_vectors: DataFrame, k: int, distance_measure: str) -> KMeansModel:
        return BisectingKMeans().setMaxIter(get_configs()["clustering"]["max-iters"]).setK(k).setDistanceMeasure(distance_measure) \
            .setFeaturesCol("vector").setPredictionCol("cluster_id").fit(urls_and_vectors)


if __name__ == "__main__":
    BisectingKMeansJob(10, "cosine", 2.9668130681806715, 4).start()
