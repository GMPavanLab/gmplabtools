from unittest import mock
import pytest
import os
from gmplabtools import PammCommander
from tests.conftest import InputDictFixture


class TestPammCommander:

    @mock.patch('os.path.isfile', lambda x: True)
    def test_init(self, input_dict):
        # when:
        pamm = PammCommander(input_dict)

        # then:
        assert pamm.dimension == 3

    @InputDictFixture(fspread=0.2)
    @mock.patch('os.path.isfile', lambda x: True)
    def test_format_fpoints_and_fspread(self, input_dict):
        # when:
        pamm = PammCommander(input_dict)

        # then:
        with pytest.raises(ValueError, match="Must provide only"):
            pamm.format()

    @InputDictFixture(fpoints=None, fspread=None)
    @mock.patch('os.path.isfile', lambda x: True)
    def test_format_fpoints_and_fspread_missing(self, input_dict):
        # when:
        pamm = PammCommander(input_dict)

        # then:
        with pytest.raises(ValueError, match="Must provide only"):
            pamm.format()

    @InputDictFixture(gridfile="some_gridfile")
    def test_grid_file_not_found(self, input_dict):
        # when:
        pamm = PammCommander(input_dict)

        # then:
        with pytest.raises(ValueError, match="Grid file `some_gridfile` was not found"):
            pamm.format()