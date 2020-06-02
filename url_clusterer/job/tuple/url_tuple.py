from collections import namedtuple

from pyspark.sql.types import StructType, IntegerType, StringType

URL = StructType().add("id", IntegerType(), False).add("url", StringType(), False)
SplitURL = namedtuple("SplitURL", ["id", "url", "split_url"])
CombinedSplitURLsAndCoefficients = namedtuple("CombinedSplitURLsAndCoefficients",
                                              ["id", "url", "split_urls_and_coefficients"])
IDURLPosWordCoefficient = namedtuple("IDURLPosWordCoefficient", ["id", "url", "pos", "word", "coefficient"])
IDURLPosWordVectorCoefficient = namedtuple("IDURLPosWordVectorCoefficient",
                                           ["id", "url", "pos", "word", "weighted_word_vector", "coefficient"])
