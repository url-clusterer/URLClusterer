from typing import Callable

from pyspark.sql import SparkSession, DataFrame

from util.url_io import URL_IO
from util.url_vector_calculator import URLVectorCalculator


class URLFeatureExtractor:
    @staticmethod
    def get_urls_and_vectors(spark: SparkSession, window_size: int, s: float,
                             additional_weight_function: Callable[[int], float] = lambda e: 1) -> DataFrame:
        """
        Extracts feature vectors with the given parameters and returns.

        :param spark: SparkSession object.
        :param window_size: window_size parameter of Word2Vec.
        :param s: s parameter of Zipf distribution.
        :param additional_weight_function: additional weight function to be used for weighting.
        :return: A DataFrame of URLS and vectors with columns: id, url, split_url, coefficients, vector.
        """
        split_urls_and_word_frequency_orders = URL_IO.read_split_urls_and_word_frequency_orders(spark)
        word_vectors = URL_IO.read_word_vectors(spark, window_size)
        urls_and_vectors = URLVectorCalculator \
            .get_urls_and_vectors(split_urls_and_word_frequency_orders,
                                  word_vectors,
                                  s,
                                  additional_weight_function)
        return urls_and_vectors
