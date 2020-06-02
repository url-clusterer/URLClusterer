import operator
from typing import Callable

import numpy as np
from pyspark.ml.feature import StandardScaler
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql.functions import posexplode
from pyspark.sql.types import Row

from url_clusterer.job.tuple.url_tuple import CombinedSplitURLsAndCoefficients, IDURLPosWordCoefficient, \
    IDURLPosWordVectorCoefficient


class URLVectorCalculator:
    @staticmethod
    def get_zipf_coefficient(order, s):
        return 1 / (order ** s)

    @staticmethod
    def get_coefficients(split_urls_and_word_frequency_orders: DataFrame, s: float,
                         additional_weight_function: Callable[[int], float] = lambda e: 1) -> DataFrame:
        """

        :param split_urls_and_word_frequency_orders: A DataFrame of split URLs and word frequency orders with columns:
                                                     id, url, split_url, word_frequency_orders.
        :param s: s parameter of Zipf distribution.
        :param additional_weight_function: additional weight function to be applied additional weight beside Zipf to
                                           word vector.
        :return: A DataFrame of split URLs and coefficient of each term with columns: id, url, split_url, coefficients
        """

        def calculate_coefficients(word_frequency_orders):
            coefficients = []
            for i in range(len(word_frequency_orders)):
                coefficients.append(
                    additional_weight_function(i) * URLVectorCalculator.get_zipf_coefficient(word_frequency_orders[i],
                                                                                             s))
            return coefficients

        get_coefficients_udf = F.udf(calculate_coefficients, T.ArrayType(T.DoubleType()))
        split_urls_and_coefficients = split_urls_and_word_frequency_orders \
            .select("id",
                    "url",
                    "split_url",
                    get_coefficients_udf("word_frequency_orders").alias("coefficients"))
        return split_urls_and_coefficients

    @staticmethod
    def combine_split_urls_and_coefficients(split_urls_and_coefficients: Row) -> CombinedSplitURLsAndCoefficients:
        combined_split_urls_and_coefficients = []
        for i in range(len(split_urls_and_coefficients.split_url)):
            combined_split_urls_and_coefficients.append(
                (split_urls_and_coefficients.split_url[i], split_urls_and_coefficients.coefficients[i]))
        return CombinedSplitURLsAndCoefficients(split_urls_and_coefficients.id, split_urls_and_coefficients.url,
                                                combined_split_urls_and_coefficients)

    @staticmethod
    def join_urls_and_weighted_word_vectors(split_urls_and_coefficients: DataFrame,
                                            word_vectors: DataFrame) -> DataFrame:
        """
        Explodes split URLs to words and calculates each word's weighted word vector.

        :param split_urls_and_coefficients: A DataFrame of split URLs and coefficients with columns: id, url, split_url,
                                            coefficients.
        :param word_vectors: A DataFrame of words and vectors with columns: word, vector.
        :return: A DataFrame of URLs and weighted word vectors with columns: id, url, pos, word, weighted_word_vector,
                 coefficient.
        """
        combined_split_urls_and_coefficients = split_urls_and_coefficients \
            .rdd.map(URLVectorCalculator.combine_split_urls_and_coefficients).toDF()
        return combined_split_urls_and_coefficients \
            .select("id", "url", posexplode("split_urls_and_coefficients")) \
            .withColumnRenamed("col", "word_and_coefficient") \
            .rdd.map(lambda e:
                     IDURLPosWordCoefficient(
                         e.id, e.url, e.pos, e.word_and_coefficient[0], e.word_and_coefficient[1])) \
            .toDF().join(word_vectors, "word") \
            .rdd.map(lambda e:
                     IDURLPosWordVectorCoefficient(id=e.id, url=e.url, pos=e.pos, word=e.word,
                                                   coefficient=e.coefficient,
                                                   weighted_word_vector=list([float(x) for x in
                                                                              np.multiply(e.vector.values,
                                                                                          e.coefficient)]))) \
            .toDF()

    @staticmethod
    def sort_list_of_2_tuples_by_0th_item(list_of_2_tuples: list) -> list:
        return [item[1] for item in sorted(list_of_2_tuples, key=operator.itemgetter(0))]

    @staticmethod
    def sum_word_vectors(urls_and_weighted_word_vectors: DataFrame) -> DataFrame:
        """
        Sums weighted word vectors and their corresponding coefficients for each URL.

        :param urls_and_weighted_word_vectors: A DataFrame of URLs and weighted word vectors with columns: id, url, pos,
                                               word, weighted_word_vector, coefficient.
        :return: A DataFrame of URLs and their corresponding sum of word vectors and sum of coefficients with columns:
                 id, url, split_url, coefficients, summed_vectors, summed_coefficients.
        """

        word_array_sorter_udf = F.udf(URLVectorCalculator.sort_list_of_2_tuples_by_0th_item,
                                      T.ArrayType(T.StringType()))
        coefficient_array_sorter_udf = F.udf(URLVectorCalculator.sort_list_of_2_tuples_by_0th_item,
                                             T.ArrayType(T.DoubleType()))

        vector_size = len(urls_and_weighted_word_vectors.select('weighted_word_vector').first()[0])
        return urls_and_weighted_word_vectors \
            .groupBy("id", "url") \
            .agg(F.collect_list(F.struct("pos", "word")).alias("positions_and_words"),
                 F.collect_list(F.struct("pos", "coefficient")).alias("positions_and_coefficients"),
                 F.sum("coefficient").alias("summed_coefficients"),
                 F.array(*[F.sum(F.col("weighted_word_vector")[i])
                           for i in range(vector_size)]).alias("summed_vectors")) \
            .select("id", "url", "summed_coefficients", "summed_vectors",
                    word_array_sorter_udf("positions_and_words").alias("split_url"),
                    coefficient_array_sorter_udf("positions_and_coefficients").alias("coefficients"))

    @staticmethod
    def divide_summed_vectors_by_summed_coefficient(urls_and_summed_vectors: DataFrame) -> DataFrame:
        """
        Divides sum of vectors with their corresponding sum of coefficient for each URL.

        :param urls_and_summed_vectors: A DataFrame of URLs and their corresponding sum of vectors and sum of
                                        coefficients with columns: id, url, split_url, coefficients, summed_vectors,
                                        summed_coefficients.
        :return: A DataFrame of URLs and their corresponding vectors with columns: id, url, split_url, coefficients,
                 vector.
        """
        vector_divider_udf = F.udf(lambda e: Vectors.dense(np.divide(np.asarray(e[0]), e[1])), VectorUDT())
        return urls_and_summed_vectors \
            .select("id", "url", "split_url", "coefficients",
                    vector_divider_udf(
                        F.struct("summed_vectors", "summed_coefficients").alias("vector_and_coefficient")
                    ).alias("vector"))

    @staticmethod
    def standardize_url_vectors(urls_and_vectors: DataFrame) -> DataFrame:
        """
        Standardizes URLs and vectors DataFrame.
        :param urls_and_vectors: A DataFrame of URLs and vectors with columns: id, url, split_url, coefficients, vector.
        :return: A DataFrame of URLS and standardized vectors with columns: id, url, split_url, coefficients, vector.
        """
        standard_scaler = StandardScaler(inputCol="vector", outputCol="scaled_vector")
        standard_scaler_model = standard_scaler.fit(urls_and_vectors)
        return standard_scaler_model \
            .transform(urls_and_vectors) \
            .select("id", "url", "split_url", "coefficients", "scaled_vector") \
            .withColumnRenamed("scaled_vector", "vector")

    @staticmethod
    def get_urls_and_vectors(split_urls_and_word_frequency_orders: DataFrame, word_vectors: DataFrame, s: float,
                             additional_weight_function: Callable[[int], float] = lambda e: 1):
        """
        Calculates vectors for each URL.

        :param split_urls_and_word_frequency_orders: A DataFrame of split URLs and word frequency orders with columns:
                                                     id, url, split_url, word_frequency_orders.
        :param word_vectors: A DataFrame of words and vectors with columns: word, vector.
        :param s: s parameter of Zipf distribution.
        :param additional_weight_function: additional weight function to be used for weighting.
        :return: A DataFrame of URLS and vectors with columns: id, url, split_url, coefficients, vector.
        """
        split_urls_and_coefficients = URLVectorCalculator \
            .get_coefficients(split_urls_and_word_frequency_orders, s, additional_weight_function)
        urls_and_weighted_word_vectors = URLVectorCalculator \
            .join_urls_and_weighted_word_vectors(split_urls_and_coefficients, word_vectors)
        urls_and_summed_vectors = URLVectorCalculator.sum_word_vectors(urls_and_weighted_word_vectors)
        urls_and_vectors = URLVectorCalculator.divide_summed_vectors_by_summed_coefficient(urls_and_summed_vectors)
        urls_and_vectors = URLVectorCalculator.standardize_url_vectors(urls_and_vectors)
        return urls_and_vectors
