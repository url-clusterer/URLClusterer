import logging

from pyspark.sql import SparkSession, DataFrame

from config.config_loader import get_configs
from job.batch_job import BatchJob
from job.clustering.bisecting_kmeans_job import BisectingKMeansJob
from job.clustering.profiling.profiler import Profiler
from util.url_feature_extractor import URLFeatureExtractor


class BisectingKMeansProfilerJob(BatchJob, Profiler):
    __urls_and_vectors: DataFrame
    __number_of_samples: int

    def __init__(self, number_of_experiments: int, k: int, distance_measure: str, s: float, window_size: int) -> None:
        self.__k = k
        self.__distance_measure = distance_measure
        self.__s = s
        self.__window_size = window_size
        self.__additional_weight_function = lambda e: 1
        super().__init__(number_of_experiments)

    app_name: str = "BisectingKMeansProfilerJob"

    def apply_additional_configs(self, builder: SparkSession.Builder) -> SparkSession.Builder:
        return builder

    def run(self, spark: SparkSession) -> None:
        logging.getLogger("BisectingKMeansProfilerJob.run()").info("started.")
        self.__urls_and_vectors = URLFeatureExtractor \
            .get_urls_and_vectors(spark, self.__window_size, self.__s, self.__additional_weight_function).cache()
        self.__number_of_samples = self.__urls_and_vectors.count()
        self.run_experiments()
        self.__urls_and_vectors.unpersist()
        self.log_profiling_results()

    def log_profiling_results(self) -> None:
        logger = "BisectingKMeansProfilerJob.log_profiling_results()"
        logging.getLogger(logger).info(str(self._number_of_experiments) + " experiment had run for K-means on " +
                                       str(self.__number_of_samples) + " samples.")
        logging.getLogger(logger).info("Mean execution time: " + str(self.get_mean_execution_time()))
        logging.getLogger(logger).info("Standard deviation of execution times: " +
                                       str(self.get_standard_deviation_of_execution_times()))

    def get_clusters(self) -> DataFrame:
        return BisectingKMeansJob \
            .get_model(self.__urls_and_vectors, self.__k, self.__distance_measure) \
            .transform(self.__urls_and_vectors)


if __name__ == "__main__":
    BisectingKMeansProfilerJob(get_configs()["profiler"]["number-of-experiments"], 10, "euclidean", 3, 5).start()
