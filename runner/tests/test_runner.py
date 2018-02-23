import io
import pytest
import tempfile
import numpy as np
import autorun.runner as arm
from unittest.mock import MagicMock, patch, call


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


def test_zopen():
    with patch('gzip.open') as gzopen:
        arm.zopen('test.gz')
        gzopen.assert_called_once_with('test.gz', 'rt')
    with patch('builtins.open') as nopen:
        arm.zopen('test.txt')
        nopen.assert_called_once_with('test.txt', 'rt')


def test_split():
    '''Testing split helper function'''
    pass


def test_row_average():
    dat = [[1, 2, 3], [1, 4, -3], [1, 6, 0]]
    assert all(arm.row_average(dat) == [1, 4, 0])


# def test_best_cv():
#     fits = [
#             # Mod1    Mod2    Mod3
#             [(1, 4), (1, 1), (1, 4)],  # Run 1
#             [(2, 8), (2, 4), (2, 6)],  # Run 2
#             [(3, 6), (3, 3), (3, 8)],  # Run 3
#             [(4, 2), (4, 2), (4, 4)],  # Run 4
#     ]
#     assert arm.best_cv(fits) == 1  # Index of best model
# 

def test_load_semantic_file():
    rows = [
        '1.50,2.00,3.25',
        '2.25,4.25,3.75',
        '3.35,6.75,9.00',
    ]
    sem = [
        [1.5, 2.0, 3.25],
        [2.25, 4.25, 3.75],
        [3.35, 6.75, 9.0],
    ]
    assert np.array_equal(arm.load_semantic_file(rows), sem)


def test_run_path():
    assert arm.get_run_path('foo', 0) == 'foo/foo0'
    assert arm.get_run_path('foo/', 1) == 'foo/foo1'
    assert arm.get_run_path('./foo', 2) == './foo/foo2'
    assert arm.get_run_path('./foo/', 2) == './foo/foo2'
    assert arm.get_run_path('../foo', 3) == '../foo/foo3'
    assert arm.get_run_path('../foo/', 4) == '../foo/foo4'
    assert arm.get_run_path('../.././foo', 5) == '../.././foo/foo5'
    assert arm.get_run_path('../.././foo/', 6) == '../.././foo/foo6'


@patch('autorun.runner.load_dataset')
@patch('autorun.runner.mkdir')
def test_Dataset(mkdir, load_dataset):
    '''Testing the Dataset class.'''
    # Prepare a dataset that would be loaded from file
    dat = [
        ['1', '2', '3'],
        ['4', '5', '6'],
        ['7', '8', '9']]
    load_dataset.return_value = dat
    # Build dataset
    with patch.object(arm.Dataset, '_write_dataset') as wd:
        ds = arm.Dataset(None, 3, 'dir')
        ds.generate_folds(False)
        calls = [
                call('dir/dataset/train_0.dat', 2, 2, [dat[1], dat[2]]),
                call('dir/dataset/test_0.dat', 2, 1, [dat[0]]),
                call('dir/dataset/train_1.dat', 2, 2, [dat[0], dat[2]]),
                call('dir/dataset/test_1.dat', 2, 1, [dat[1]]),
                call('dir/dataset/train_2.dat', 2, 2, [dat[0], dat[1]]),
                call('dir/dataset/test_2.dat', 2, 1, [dat[2]]),
        ]
        assert wd.mock_calls == calls


def test_Runner():
    '''Testing the Runner class'''
    pass


@pytest.mark.parametrize(
    "algo,models,data,gens,k_folds,outdir,bindir,conf", [
        ('algo', ['m1', 'm2'], 'data.txt', 100, 3, 'outd', 'bind', 'conf.ini'),
    ])
@patch('autorun.runner.Runner')
@patch('autorun.runner.Dataset')
@patch('autorun.runner.Logger')
@patch('autorun.runner.load_avg_semantic')
def test_forrest(sem_loader, Logger, Dataset, Runner,
                 algo, models, data, gens,
                 k_folds, outdir, bindir, conf):
    '''Tests that a single run will produce all the right files.'''
    # Get objects
    logger = Logger()
    dataset = Dataset()
    # dataset.datafile = data
    runner = Runner()

    Logger.reset_mock()
    # Dataset.reset_mock()
    Runner.reset_mock()

    # Test setup
    forrest = arm.Forrest('test', algo, models, dataset, k_folds,
                          outdir, bindir, conf)

    Logger.assert_called_once_with(f'{outdir}/test')
    # Dataset.assert_called_once_with(data, k_folds, out_dir=outdir)
    Runner.assert_called_once_with(algo, dataset,
                                   out_dir=outdir,
                                   bin_dir=bindir,
                                   conf_path=conf,
                                   error_measure='RMSE')
    # Setup mock objects being used
    sem_loader.side_effect = [100, 200]
    runner.run.side_effect = zip(range(1, k_folds+1), range(10, k_folds+10))
    logger.out_sem_train = lambda k: f'{outdir}/train-sem/{k}'
    logger.out_sem_test = lambda k: f'{outdir}/test-sem/{k}'

    # Test run
    k_fits, _, avg_sem_train, avg_sem_test = forrest.run(gens)

    # Ensure load_avg_semantic was called with right files
    assert sem_loader.mock_calls == [
        call([f'{outdir}/train-sem/{k}' for k in range(k_folds)]),
        call([f'{outdir}/test-sem/{k}' for k in range(k_folds)]),
    ]
    # Ensure fitness are returned correctly
    assert k_fits == [(1, 10), (2, 11), (3, 12)]
    # Ensure semantics are correct
    assert avg_sem_train == 100
    assert avg_sem_test == 200


@pytest.mark.parametrize(
    "algo,models,data,gens,k_folds,outdir,bindir,conf", [
        ('algo', ['m1', 'm2'], 'data.txt', 100, 3, 'outd', 'bind', 'conf.ini'),
    ])
@patch('autorun.runner.Runner')
@patch('autorun.runner.Dataset')
@patch('autorun.runner.load_avg_semantic')
def test_integration_forrest(sem_loader, Dataset, Runner,
                 algo, models, data, gens,
                 k_folds, outdir, bindir, conf):
    '''Tests that a single run will produce all the (real) right files.'''
    # Get objects
    dataset = Dataset()
    runner = Runner()

    # Dataset.reset_mock()
    Runner.reset_mock()

    # Test setup
    with patch('autorun.runner.mkdir'):
        forrest = arm.Forrest('integr', algo, models, dataset, k_folds,
                              outdir, bindir, conf)

    logger = forrest._logger

    logdir = f'{outdir}/integr'
    assert logger._dir == logdir
    # Dataset.assert_called_once_with(data, k_folds, out_dir=outdir)
    Runner.assert_called_once_with(algo, dataset,
                                   out_dir=outdir,
                                   bin_dir=bindir,
                                   conf_path=conf,
                                   error_measure='RMSE')
    # Setup mock objects being used
    sem_loader.side_effect = [100, 200]
    runner.run.side_effect = zip(range(1, k_folds+1), range(10, k_folds+10))

    # Test run
    k_fits, _, avg_sem_train, avg_sem_test = forrest.run(gens)

    # Ensure load_avg_semantic was called with right files
    assert sem_loader.mock_calls == [
        call([f'{logdir}/sem_train_{k}.txt.gz' for k in range(k_folds)]),
        call([f'{logdir}/sem_test_{k}.txt.gz' for k in range(k_folds)]),
    ]
    # Ensure fitness are returned correctly
    assert k_fits == [(1, 10), (2, 11), (3, 12)]
    # Ensure semantics are correct
    assert avg_sem_train == 100
    assert avg_sem_test == 200


