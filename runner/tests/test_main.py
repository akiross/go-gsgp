import io
import pytest
import tempfile
import autorun.main as arm
from unittest.mock import MagicMock, patch


@pytest.mark.parametrize(
    "elems,expected", [
        ([], [()]),
        ([1], [(), (1,)]),
        ([1, 2], [(), (1,), (2,), (1, 2)]),
        ([1, 2, 3], [(), (1,), (2,), (3,), (1, 2), (1, 3), (2, 3), (1, 2, 3)]),
    ])
def test_powerset(elems, expected):
    '''Testing poweset helper function'''
    assert set(arm.powerset(elems)) == set(expected)

def test_split():
    '''Testing split helper function'''
    assert False

def test_Dataset():
    '''Testing the Dataset class'''
    with tempfile.TemporaryDirectory() as td:
        data = '1 2 3\n4 5 6'
        pseudo_file = io.StringIO(data)
        # This will write to file
        ds = arm.Dataset(pseudo_file, 2, out_path=td)
        with open(f'{td}/train_1.dat') as f1, open(f'{td}/test_1.dat') as f2:
            # Check if dataset size is correct
            n_vars, n_rows = int(f1.readline()), int(f1.readline())
            assert n_vars == 2
            assert n_rows == 1

            n_vars, n_rows = int(f2.readline()), int(f2.readline())
            assert n_vars == 2
            assert n_rows == 1

            d1 = f1.read()
            d2 = f2.read()

            assert len(d1.split()) == n_vars + 1
            assert len(d2.split()) == n_vars + 1

            assert d1.split() == ['1', '2', '3']
            assert d2.split() == ['4', '5', '6']

def test_Runner():
    '''Testing the Runner class'''
    assert False

