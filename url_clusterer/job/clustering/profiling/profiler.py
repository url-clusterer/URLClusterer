import logging
from abc import ABC, abstractmethod
from statistics import mean, stdev
from time import time
from typing import List

from pyspark.sql import DataFrame


class Profiler(ABC):
    __execution_times: List[int]
    _number_of_experiments: int

    def __init__(self, number_of_experiments):
        self._number_of_experiments = number_of_experiments
        self.__execution_times = [0] * self._number_of_experiments

    @abstractmethod
    def get_clusters(self) -> DataFrame:
        pass

    def run_experiments(self):
        for i in range(self._number_of_experiments):
            start_time_of_experiment = time()
            self.get_clusters().count()  # It is run for a Spark action to happen.
            self.__execution_times[i] = int((time() - start_time_of_experiment) * 1000)
            logging.getLogger("Profiler.run_experiments()").info("Execution time of experiment " +
                                                                 str(i + 1) + ": " +
                                                                 str(self.__execution_times[i]) + " ms.")

    def get_mean_execution_time(self):
        return mean(self.__execution_times)

    def get_standard_deviation_of_execution_times(self):
        return stdev(self.__execution_times)
