from typing import Iterable

from pyspark.ml.feature import Word2Vec
from pyspark.sql import SparkSession, DataFrame

from job.batch_job import BatchJob
from util.url_io import URL_IO


class Word2VecJob(BatchJob):
    def __init__(self, window_size_range: Iterable[int]) -> None:
        self.__window_size_range = window_size_range

    app_name: str = "Word2VecJob"

    def apply_additional_configs(self, builder: SparkSession.Builder) -> SparkSession.Builder:
        return builder

    def run(self, spark: SparkSession) -> None:
        split_urls = URL_IO.read_split_urls_and_word_frequency_orders(spark).select("split_url")
        for window_size in self.__window_size_range:
            word_vectors = self.calculate_word_vectors(split_urls, window_size)
            URL_IO.write_word_vectors(word_vectors, window_size)

    @staticmethod
    def calculate_word_vectors(split_urls: DataFrame, window_size: int) -> DataFrame:
        word2vec = Word2Vec(windowSize=window_size, vectorSize=100, minCount=0, inputCol="split_url", seed=1,
                            outputCol="url_vector")
        model = word2vec.fit(split_urls)
        word_vectors = model.getVectors()
        return word_vectors


if __name__ == "__main__":
    Word2VecJob(range(2, 6)).start()
