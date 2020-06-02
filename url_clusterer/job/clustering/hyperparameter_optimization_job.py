import logging
from abc import ABC, abstractmethod

import hyperopt
from hyperopt.pyll.base import Apply
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.sql import SparkSession, DataFrame

from config.config_loader import get_configs
from url_clusterer.job.batch_job import BatchJob
from util.url_io import URL_IO
from util.url_vector_calculator import URLVectorCalculator


class HyperparameterOptimizationJob(BatchJob, ABC):
    app_name: str
    search_space: Apply
    spark: SparkSession
    split_urls: DataFrame
    split_urls_and_word_frequency_orders: DataFrame
    term_counts: DataFrame
    __best_score = -1

    def apply_additional_configs(self, builder: SparkSession.Builder) -> SparkSession.Builder:
        return builder

    def run(self, spark: SparkSession) -> None:
        self.spark = spark
        self.split_urls_and_word_frequency_orders = URL_IO.read_split_urls_and_word_frequency_orders(spark)
        logging.getLogger("run()").info("Running " + self.app_name + " for " + get_configs()["dataset"]["name"])
        best_hyperparameters = self.get_best_hyperparameters()
        logging.getLogger("run()").info(
            "best hyperparameters: " + str(hyperopt.space_eval(self.search_space, best_hyperparameters)))

    def additional_weight_function(self, position):
        return 1

    @abstractmethod
    def get_clusters(self, parameters: dict, urls_and_vectors: DataFrame) -> DataFrame:
        """
        Apply a clustering algorithm and assign clusters to each URL in the urls_and_vectors DataFrame.

        :param parameters: A dict of parameters to run clustering algorithm.
        :param urls_and_vectors: A DataFrame of URLs and their corresponding feature vectors with columns: id, url,
                                 split_url, coefficients, vector
        :return: A DataFrame of URLs and the cluster id of the URLs are assigned by the clustering algorithm with
                 columns: id, url, split_url, coefficients, vector, cluster_id.
        """
        pass

    def get_silhouette_score(self, clustered_url_vectors: DataFrame, distance_measure: str = "euclidean") -> float:
        """
        Calculates the silhouette score of the given cluster in the parameter and returns.

        :param distance_measure: The distance measure that is used for clustering.
        :param clustered_url_vectors: A DataFrame of URLs and the cluster id of the URLs are assigned by the clustering
                                      algorithm with columns: id, url, split_url, coefficients, vector, cluster_id.
        :return: silhouette score of the clustering of clustered URLs.
        """
        if distance_measure == "euclidean":
            distance_measure = "squaredEuclidean"
        evaluator = ClusteringEvaluator(predictionCol="cluster_id", featuresCol="vector",
                                        distanceMeasure=distance_measure)
        return evaluator.evaluate(clustered_url_vectors)

    def objective(self, args: tuple) -> float:
        clustering_algorithm = args[0]
        parameters = args[1]
        logging.getLogger("objective()").info("parameters: " + str(parameters))
        word_vectors = URL_IO.read_word_vectors(self.spark, parameters["window_size"])
        urls_and_vectors = URLVectorCalculator \
            .get_urls_and_vectors(self.split_urls_and_word_frequency_orders,
                                  word_vectors,
                                  parameters["s"],
                                  self.additional_weight_function)
        clustered_url_vectors = self.get_clusters(parameters, urls_and_vectors)
        cluster_score = self.get_silhouette_score(clustered_url_vectors, parameters["distance_measure"])
        if cluster_score > self.__best_score:
            self.__best_score = cluster_score
            URL_IO.write_clustered_urls(clustered_url_vectors.select("id", "url", "cluster_id", "vector"),
                                        clustering_algorithm)
        logging.getLogger("objective()").info("cluster score: " + str(cluster_score))
        return -cluster_score

    def get_best_hyperparameters(self) -> dict:
        return hyperopt.fmin(
            fn=self.objective,
            space=self.search_space,
            algo=hyperopt.tpe.suggest,
            max_evals=get_configs()["hyperparameter-optimization"]["number-of-evaluations"])
