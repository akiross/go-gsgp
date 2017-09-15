#!/usr/bin/env python3

import argparse
from subprocess import run

def powerset(iterable):
    from itertools import chain, combinations
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def split(dataset, k):
    '''Returns two file names containing the k-th fold of cross-validation'''
    return 'foo{} bar{}'.format(k, k).split()

class Dataset:
    '''Represents a dataset to be used with k-fold cross-validation'''
    def __init__(self, datafile, k, randomize=True): # TODO , out_path='./'):
        from random import shuffle
        self._ds_path = datafile
        self._k = k
        self._written = [] # Dataset files

        self._prepare_datasets(k)


    def get_fold_path(self, i):
        '''Returns the name of the i-th train/test files'''
        return self._written[i]

    def _prepare_datasets(self, k):
        '''Prepare datafiles for k-fold cross validation,
        producing train_*.dat and test_*.dat files in current
        directory'''

        # Load the data
        with self._ds_path as df:
            self._ds = [l.strip().split() for l in df]
        # Number of variables in the dataset
        n_vars = len(self._ds[0]) - 1

        # Shuffle rows
        if randomize:
            shuffle(self._ds)

        # Generate files
        dsl = len(self._ds) 
        size = (dsl + k - 1) // k

        # Sequence number for split
        n = 0
        for i in range(0, dsl, size):
            n += 1
            j = min(dsl, i+size)
            # Path of files to write
            train_file, test_file = f'train_{n}.dat', f'test_{n}.dat'
            self._written.append((train_file, test_file))

            with open(train_file, 'wt') as train, open(test_file, 'wt') as test:
                # Write train dataset
                print(n_vars, file=train)
                print(dsl - j + i, file=train)
                print('\n'.join('\t'.join(r) for r in self._ds[:i] + self._ds[j:]), file=train)
                # Write test dataset
                print(n_vars, file=test)
                print(j-i, file=test)
                print('\n'.join('\t'.join(r) for r in self._ds[i:j]), file=test)

class Runner:
    '''Class responsible to perform runs'''
    def __init__(self, algo, dataset):
        self._algo = algo
        self._ds = dataset

    def run(k, mods, n_gens):
        ds_train, ds_test = self._ds.get_fold_path(k)
        print('Running simulation', k, 'with models', mods, 'on datasets', ds_train, ds_test, 'for', n_gens, 'generations')
        return 123, 123


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='Run tests with CV model selection')
    parser.add_argument('--k_fold', '-k', type=int, default=10, help='Number of folds')
    parser.add_argument('datafile', type=open, help='Dataset file to load')
    args = parser.parse_args()

    # How many CV folds
    k_fold = 10
    # How many runs
    num_runs = 30
    # Short run
    short_gens = 100
    # Long run
    long_gens = 1000
    # Models we are applying
    models = ['../models/nl_lr_sem.py', '../models/nl_mlp_sem.py', '../models/nl_svr_sem.py']
    # Full dataset
    #datafile = args.datafile # './dataset.txt'
    # Algorithm to use
    algo = '../go-gsgp-cpu'

    # A friendly reminder
    print('We are going to run', 10 * 2**len(models), 'times the short version')

    # Load the dataset and prepare the k-fold
    dataset = Dataset(args.datafile, k_fold)

    # Prepare for running
    runner = Runner(algo, dataset)

    # Where fitnesses are saved for CV
    cv_fits = []

    # Perform k-fold cross validation
    for k in range(k_fold):
        ## Split the dataset and get two file names
        #ds_train, ds_test = dataset.get_fold_path(k)
        # Where fitnesses are saved for this fold
        k_fits = []
        # For every combination of models
        for mods in powerset(models):
            # Run simulation gathering results
            train_fit, test_fit = runner.run(k, mods, short_gens)
            # Accumulate fitnesses for this fold
            k_fits.append((train_fit, test_fit))
        # Accumulate fitnesses for cross validation
        cv_fits.append(k_fits)

    # Compute cross validation
    best_models = select_models(best_cv(cv_fits), powerset(models))

    # Perform long run
    for k in range(k_fold):
        ds_train, ds_test = dataset.get_fold_path(k)
        run_sim(k, best_models, ds_train, ds_test, long_gens)












