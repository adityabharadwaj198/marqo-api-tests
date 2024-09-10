import abc
import argparse
import pytest
import subprocess
from typing import List, Optional

class BaseTestCase(abc.ABC):
    @abc.abstractmethod
    def prepare(self):
        """
        Runs on from_version. Prepare Marqo state for test execution on to_version.
        Common actions are creating indexes and adding documents.
        """
        pass

@pytest.mark.marqo_from_version('2.5')
class TestPartialUpdateExistingIndex(BaseTestCase):
    def prepare(self):
        # Create structured and unstructured indexes and add some documents
        pass

    def test_partialUpdate_scoreModifiers_success(self):
        # This runs on to_version
        pass
