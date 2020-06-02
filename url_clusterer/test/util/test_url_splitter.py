from unittest import TestCase

from url_clusterer.util.url_splitter import URLSplitter


class TestUrlSplitter(TestCase):
    def setUp(self) -> None:
        self.__url_splitter = URLSplitter()

    def test_split_url_1(self):
        url = 'https://www.example.com'
        expected_split_url = ['www']
        split_url = self.__url_splitter.split_url(url)
        self.assertEqual(expected_split_url, split_url)

    def test_split_url_2(self):
        url = 'https://www.asdf.asdf-32.f.example.com'
        expected_split_url = ['www', 'asdf', 'asdf-32', 'f']
        split_url = self.__url_splitter.split_url(url)
        self.assertEqual(expected_split_url, split_url)

    def test_split_url_3(self):
        url = 'https://www.asdf.asdf-32.f.example.com/'
        expected_split_url = ['www', 'asdf', 'asdf-32', 'f']
        split_url = self.__url_splitter.split_url(url)
        self.assertEqual(expected_split_url, split_url)

    def test_split_url_4(self):
        url = 'https://example.com/'
        expected_split_url = []
        split_url = self.__url_splitter.split_url(url)
        self.assertEqual(expected_split_url, split_url)

    def test_split_url_5(self):
        url = 'https://asdf.example.com/?q=v'
        expected_split_url = ['asdf', '?', 'q', 'v']
        split_url = self.__url_splitter.split_url(url)
        self.assertEqual(expected_split_url, split_url)

    def test_split_url_6(self):
        url = 'https://asdf.example.com/c1/c2-3/c4/5/6-adsf-fasd?q=v&pg=1&cns'
        expected_split_url = ['asdf', 'c1', 'c2', '3', 'c4', '5', '6', 'adsf', 'fasd', '?', 'q', 'v', 'pg', '1', 'cns']
        split_url = self.__url_splitter.split_url(url)
        self.assertEqual(expected_split_url, split_url)

    def test_split_url_7(self):
        url = 'https://asdf.example.com/c1/c2-3/c4/5/6-adsf-fasd'
        expected_split_url = ['asdf', 'c1', 'c2', '3', 'c4', '5', '6', 'adsf', 'fasd']
        split_url = self.__url_splitter.split_url(url)
        self.assertEqual(expected_split_url, split_url)

    def test_split_url_8(self):
        url = 'https://asdf.example.com/c1/c2-3/c4/5/6-adsf-fasd?asd'
        expected_split_url = ['asdf', 'c1', 'c2', '3', 'c4', '5', '6', 'adsf', 'fasd', '?', 'asd']
        split_url = self.__url_splitter.split_url(url)
        self.assertEqual(expected_split_url, split_url)

    def test_split_url_9(self):
        url = 'https://asdf.example.com/c1/c2-3/c4/5/6-adsf-fasd?q=v&w&pg=1&cns'
        expected_split_url = ['asdf', 'c1', 'c2', '3', 'c4', '5', '6', 'adsf', 'fasd', '?', 'q', 'v', 'w', 'pg', '1',
                              'cns']
        split_url = self.__url_splitter.split_url(url)
        self.assertEqual(expected_split_url, split_url)
