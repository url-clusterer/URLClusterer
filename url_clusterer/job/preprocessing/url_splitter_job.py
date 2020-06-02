from typing import Callable, List

from pyspark.sql import SparkSession

from url_clusterer.job.batch_job import BatchJob
from url_clusterer.job.tuple.url_tuple import SplitURL
from url_clusterer.util.url_splitter import URLSplitter
from util.url_io import URL_IO


class URLSplitterJob(BatchJob):
    def __init__(self, term_sorter: Callable[[List[str]], List[str]] = None) -> None:
        self.__term_sorter = term_sorter

    def apply_additional_configs(self, builder: SparkSession.Builder) -> SparkSession.Builder:
        return builder

    app_name: str = "UrlSplitterJob"

    def run(self, spark: SparkSession) -> None:
        urls = URL_IO.read_urls(spark)
        url_splitter = URLSplitter(self.__term_sorter)
        split_urls = spark.createDataFrame(urls.rdd.map(lambda e: SplitURL(e.id, e.url, url_splitter.split_url(e.url))))
        split_urls.show()
        URL_IO.write_split_urls(split_urls)


if __name__ == "__main__":
    URLSplitterJob().start()
