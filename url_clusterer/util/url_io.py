import pandas as pd
from pyspark.sql import SparkSession, DataFrame

from config.config_loader import get_configs
from job.tuple.url_tuple import URL


class URL_IO:
    @staticmethod
    def read_urls(spark: SparkSession) -> DataFrame:
        return spark.read.csv("data/" + get_configs()["dataset"]["unlabeled"]["filename"], header=True, schema=URL)

    @staticmethod
    def write_split_urls(urls: DataFrame) -> None:
        urls.write.parquet(
            "out/" + get_configs()["dataset"]["name"] + "/" + get_configs()["parquet-filenames"]["split-url"],
            mode="overwrite")

    @staticmethod
    def read_split_urls(spark: SparkSession) -> DataFrame:
        return spark.read.parquet(
            "out/" + get_configs()["dataset"]["name"] + "/" + get_configs()["parquet-filenames"]["split-url"])

    @staticmethod
    def write_split_urls_and_word_frequency_orders(split_urls_and_word_frequency_orders: DataFrame) -> None:
        split_urls_and_word_frequency_orders \
            .write \
            .parquet("out/" + get_configs()["dataset"]["name"] + "/" + get_configs()["parquet-filenames"][
            "split-urls-and-word-frequency-orders"],
                     mode="overwrite")

    @staticmethod
    def write_word_vectors(word_vectors: DataFrame, window_size: int) -> None:
        word_vectors.write.parquet(
            "out/" + get_configs()["dataset"]["name"] + "/word_vectors_" + str(window_size) + ".parquet",
            mode="overwrite")

    @staticmethod
    def read_word_vectors(spark: SparkSession, window_size: int) -> DataFrame:
        return spark.read.parquet(
            "out/" + get_configs()["dataset"]["name"] + "/" + get_configs()["parquet-filename-prefixes"][
                "word-vectors"] + str(window_size) + ".parquet")

    @staticmethod
    def read_split_urls_and_word_frequency_orders(spark: SparkSession) -> DataFrame:
        return spark.read.parquet(
            "out/" + get_configs()["dataset"]["name"] + "/" + get_configs()["parquet-filenames"][
                "split-urls-and-word-frequency-orders"])

    @staticmethod
    def write_clustered_urls(clustered_urls: DataFrame, clustering_algorithm: str) -> None:
        clustered_urls.write.parquet(
            "out/" + get_configs()["dataset"]["name"] + "/" + get_configs()["parquet-filename-prefixes"][
                "clustered-urls-by"] + clustering_algorithm + ".parquet",
            mode="overwrite")

    @staticmethod
    def read_clustered_urls(spark: SparkSession, clustering_algorithm: str) -> DataFrame:
        return spark.read.parquet(
            "out/" + get_configs()["dataset"]["name"] + "/" + get_configs()["parquet-filename-prefixes"][
                "clustered-urls-by"] + clustering_algorithm + ".parquet")

    @staticmethod
    def read_labeled_urls() -> pd.DataFrame:
        return pd.read_csv("data/" + get_configs()["dataset"]["labeled"]["filename"])
