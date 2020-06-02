import itertools
import re
from typing import Pattern, Callable, List
from urllib.parse import urlparse


class URLSplitter:
    def __init__(self, term_sorter: Callable[[List[str]], List[str]] = None):
        self.__term_sorter = term_sorter

    split_pattern_for_path: Pattern[str] = re.compile(r'[/\- \+]')

    def split_url(self, url):
        parsed_url = urlparse(url)
        url_parts = []
        url_parts += parsed_url.netloc.split('.')[:-2]
        url_parts += filter(lambda e: e, URLSplitter.split_pattern_for_path.split(parsed_url.path))
        query_part = list(filter(lambda e: e, itertools.chain.from_iterable(
            map(lambda q: q.split('='), parsed_url.query.split('&')))))
        if query_part:
            url_parts += ['?']
            url_parts += query_part
        url_parts = list(filter(lambda e: e, url_parts))
        if self.__term_sorter:
            url_parts = self.__term_sorter(url_parts)
        return url_parts
