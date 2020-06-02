from typing import Callable

from hyperopt import hp
from hyperopt.pyll import Apply
from pyspark.ml.clustering import BisectingKMeans
from pyspark.sql import DataFrame

from job.clustering.hyperparameter_optimization_job import HyperparameterOptimizationJob


class BisectingKMeansOptimizationJob(HyperparameterOptimizationJob):
    def __init__(self, get_cluster_score: Callable[[DataFrame], float] = None,
                 additional_weight_function: Callable[[int], float] = lambda e: 1.0) -> None:
        self.additional_weight_function = additional_weight_function
        if get_cluster_score:
            self.get_cluster_score = get_cluster_score

    app_name: str = "BisectingKMeansOptimizationJob"

    search_space: Apply = hp.choice('model', [('bisecting_kmeans', {'k': hp.uniformint('k', 4, 20),
                                                                    'distance_measure': hp.choice("distance_measure",
                                                                                                  ['euclidean',
                                                                                                   'cosine']),
                                                                    'window_size': hp.uniformint('window_size', 2, 5),
                                                                    's': hp.uniform('s', .5, 5)})])

    def get_clusters(self, parameters: dict, urls_and_vectors: DataFrame) -> DataFrame:
        urls_and_vectors = urls_and_vectors.cache()
        bisecting_kmeans = BisectingKMeans().setK(parameters['k']).setDistanceMeasure(
            parameters['distance_measure']).setFeaturesCol("vector").setPredictionCol("cluster_id")
        model = bisecting_kmeans.fit(urls_and_vectors)
        clustered_url_vectors = model.transform(urls_and_vectors)
        urls_and_vectors.unpersist()
        return clustered_url_vectors


if __name__ == "__main__":
    BisectingKMeansOptimizationJob().start()
