import unittest
from unittest import mock
from marqo import config
from marqo import enums
from marqo.client import Client
from marqo import utils
from tests.marqo_test import MarqoTestCase


class TestConfig(MarqoTestCase):

    def setUp(self) -> None:
        self.endpoint = self.authorized_url

    def test_init_custom_devices(self):
        c = config.Config(url=self.endpoint,indexing_device="cuda:3", search_device="cuda:4")
        assert c.indexing_device == "cuda:3"
        assert c.search_device == "cuda:4"

    def test_set_url_localhost(self):
        @mock.patch("urllib3.disable_warnings")
        def run(mock_dis_warnings):
            c = config.Config(url="https://localhost:8882")
            assert not c.cluster_is_remote
            mock_dis_warnings.assert_called()
            return True
        assert run()

    def test_set_url_0000(self):
        @mock.patch("urllib3.disable_warnings")
        def run(mock_dis_warnings):
            c = config.Config(url="https://0.0.0.0:8882")
            assert not c.cluster_is_remote
            mock_dis_warnings.assert_called()
            return True
        assert run()

    def test_set_url_remote(self):
        @mock.patch("urllib3.disable_warnings")
        @mock.patch("warnings.resetwarnings")
        def run(mock_reset_warnings, mock_dis_warnings):
            c = config.Config(url="https://some-cluster-somewhere:8882")
            assert c.cluster_is_remote
            mock_dis_warnings.assert_not_called()
            mock_reset_warnings.assert_called()
            return True
        assert run()

    def test_url_is_s2search(self):
        c = config.Config(url="https://s2search.io/abdcde:8882")
        assert c.cluster_is_s2search

    def test_url_is_not_s2search(self):
        c = config.Config(url="https://som_random_cluster/abdcde:8882")
        assert not c.cluster_is_s2search
