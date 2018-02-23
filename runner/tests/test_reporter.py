import io
import pytest
import tempfile
import numpy as np
import autorun.reporter as rep
from unittest.mock import MagicMock, patch, call


@pytest.mark.parametrize(
    "string,seconds", [
        ('0.15ms', 0.00015),
        ('25.25ms', 0.02525),
        ('1750.5ms', 1.7505),
        ('55.25s', 55.25),
        ('135.10s', 135.10),
        ('0m20.5s', 20.5),
        ('1m20.25s', 80.25),
        ('1m70.75s', 130.75),
    ])
def test_go_timing(string, seconds):
    assert rep.parse_duration(string) == seconds
