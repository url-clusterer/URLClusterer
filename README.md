# URL Clusterer

## Overview

This repository contains an implementation of a Spark pipeline that clusters URLs for any website. The stages of the pipeline and the related Spark job file of each stage is as follows:

- Splitting: `url_clusterer/job/preprocessing/url_splitter_job.py`
- Counting words: `url_clusterer/job/preprocessing/url_word_count_job.py`
- Applying Word2vec: `url_clusterer/job/preprocessing/word2vec_job.py`
- Finding clusters by searching for best hyperparameters (using K-means or Bisecting K-means)
	- K-means: `url_clusterer/job/clustering/kmeans_optimization_job.py`
	- Bisecting K-means: `url_clusterer/job/clustering/bisecting_kmeans_optimization_job.py`

To evaluate the clustering result given a ground truth, `url_clusterer/job/evaluation/cluster_evaluator_job.py` can be used. Then to visualize the resulting clustering, `url_clusterer/job/visualization/visualization_job.py` can be used.

Input and output data locations, and some of the parameters of the algorithms in the pipeline for these Spark jobs can be configured in `config.yml`. `example_config.yml` file is an example configuration file.

## Prerequisites

```
pip3 install pyspark numpy pandas scikit-learn bokeh pyarrow hyperopt PyYAML
```

## Usage

To use the pipeline, attributes in the `config.yml` file must be set. Then splitting, counting words, Word2vec, and one of the optimizations jobs (K-means or Bisecting K-means) must be run respectively. The resulting cluster ids for each URL are written to a location in the file system that is specified in the `config.yml` file.

## More about the Project

See the details about the project from the [white paper](https://github.com/url-clusterer/white-paper) repository.
