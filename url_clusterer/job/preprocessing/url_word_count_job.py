import operator

from pyspark.sql import SparkSession, DataFrame, Window
from pyspark.sql import functions as F
from pyspark.sql.functions import posexplode
from pyspark.sql.types import IntegerType, ArrayType, StringType

from url_clusterer.job.batch_job import BatchJob
from util.url_io import URL_IO


class URLWordCountJob(BatchJob):
    def __init__(self, min_count: int = 10, max_ratio: float = .8):
        self.__min_count = min_count
        self.__max_ratio = max_ratio

    app_name: str = "URLWordCountJob"
    __max_count: int

    def apply_additional_configs(self, builder: SparkSession.Builder) -> SparkSession.Builder:
        return builder

    def collect_split_urls_order_by_positions(self, urls_exploded_by_split_urls: DataFrame,
                                              words_and_frequency_orders: DataFrame) -> DataFrame:
        def sorter(l: list) -> list:
            res = sorted(l, key=operator.itemgetter(0))
            return [item[1] for item in res]

        string_sorter_udf = F.udf(sorter, ArrayType(StringType()))
        integer_sorter_udf = F.udf(sorter, ArrayType(IntegerType()))
        split_urls_and_word_frequency_orders = urls_exploded_by_split_urls \
            .join(words_and_frequency_orders, "word", "inner") \
            .select("id", "url", "word", "pos", "word_frequency_order") \
            .groupBy("id", "url") \
            .agg(F.collect_list(F.struct("pos", "word")).alias("pos_and_words"),
                 F.collect_list(F.struct("pos", "word_frequency_order")).alias("pos_and_word_frequency_orders")) \
            .select("id",
                    "url",
                    string_sorter_udf("pos_and_words").alias("split_url"),
                    integer_sorter_udf("pos_and_word_frequency_orders").alias("word_frequency_orders"))
        return split_urls_and_word_frequency_orders

    def get_words_and_frequency_orders(self, urls_exploded_by_split_urls: DataFrame) -> DataFrame:
        window = Window.orderBy(F.col("count").desc())
        words_and_frequency_orders = urls_exploded_by_split_urls \
            .groupBy("word") \
            .agg(F.count(F.col("word")).alias("count")) \
            .orderBy("count", ascending=False) \
            .filter(F.col("count") >= self.__min_count) \
            .filter(F.col("count") < self.__max_count) \
            .withColumn("word_frequency_order", F.row_number().over(window)) \
            .select("word", "word_frequency_order")
        return words_and_frequency_orders

    def get_urls_exploded_by_split_urls(self, split_urls: DataFrame) -> DataFrame:
        return split_urls \
            .select("id", "url", posexplode(split_urls.split_url)) \
            .withColumnRenamed("col", "word")

    def run(self, spark: SparkSession) -> None:
        split_urls = URL_IO.read_split_urls(spark)
        self.__max_count = int(split_urls.count() * self.__max_ratio)

        urls_exploded_by_split_urls = self.get_urls_exploded_by_split_urls(split_urls)

        words_and_frequency_orders = self.get_words_and_frequency_orders(urls_exploded_by_split_urls)

        split_urls_and_word_frequency_orders = self.collect_split_urls_order_by_positions(urls_exploded_by_split_urls,
                                                                                          words_and_frequency_orders)

        URL_IO.write_split_urls_and_word_frequency_orders(split_urls_and_word_frequency_orders)


if __name__ == "__main__":
    URLWordCountJob().start()
