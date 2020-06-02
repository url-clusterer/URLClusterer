from abc import ABC, abstractmethod

from pyspark.sql import SparkSession

from logger.logging_manager import LoggingManager


class BatchJob(ABC):
    def start(self) -> None:
        LoggingManager.configure_logger()
        builder = SparkSession \
            .builder \
            .appName(self.app_name) \
            .config("spark.sql.execution.arrow.enabled", "true")

        spark = self.apply_additional_configs(builder).getOrCreate()
        self.run(spark)

    app_name: str

    @abstractmethod
    def apply_additional_configs(self, builder: SparkSession.Builder) -> SparkSession.Builder:
        pass

    @abstractmethod
    def run(self, spark: SparkSession) -> None:
        pass
