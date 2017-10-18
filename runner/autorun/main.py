#!/usr/bin/env python3

"""Run simulations with various options."""

import os
import time
import pickle
import shutil
import logging
import argparse
import subprocess
import numpy as np
from collections import Counter


global_stats = dict()


def logi(log, message):
    """Log the message in the specified log."""
    logging.getLogger(log).info(message)


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)."
    from itertools import chain, combinations
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


def cv_average(cv_fits): 
    """Compute average for CV fitness results."""
    fits = np.array(cv_fits) # Convert to numpy array
    return np.mean(fits, axis=0) # Compute average across CV runs


def best_cv(cv_fits):
    '''Given a cv_fits, computes the average for the CV process,
    `then determines which models will to be used using the test
    fitness results. Selects the best of the combinations used'''

    average = cv_average(cv_fits)
    tf = average[:,1] # Take only test fitnesses
    return tf.argmin()


class Dataset:
    '''Represents a dataset to be used with k-fold cross-validation'''
    def __init__(self, datafile, k, randomize=True, out_dir='.'):
        self._ds_obj = datafile
        self._k = k
        self._written = [] # Dataset files
        self._out_path = out_dir

        os.mkdir(os.path.join(out_dir, 'dataset'))

        self._prepare_datasets(k, randomize)

    def get_fold_path(self, i):
        '''Returns the name of the i-th train/test files'''
        return self._written[i]

    def get_out_path(self):
        return self._out_path

    def get_train_path(self, k):
        return os.path.join(self._out_path, 'dataset', f'train_{k}.dat')

    def get_test_path(self, k):
        return os.path.join(self._out_path, 'dataset', f'test_{k}.dat')

    def _prepare_datasets(self, k, randomize):
        '''Prepare datafiles for k-fold cross validation,
        producing train_*.dat and test_*.dat files in current
        directory'''

        from random import shuffle

        # Load the data
        with self._ds_obj as df:
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
            j = min(dsl, i+size)
            # Path of files to write
            train_file, test_file = self.get_train_path(n), self.get_test_path(n)
            self._written.append((train_file, test_file))

            with open(train_file, 'wt') as train, open(test_file, 'wt') as test:
                # TODO move writing to separate procedure to make testing easier
                # Write train dataset
                print(n_vars, file=train)
                print(dsl - j + i, file=train)
                print('\n'.join('\t'.join(r) for r in self._ds[:i] + self._ds[j:]), file=train)
                # Write test dataset
                print(n_vars, file=test)
                print(j-i, file=test)
                print('\n'.join('\t'.join(r) for r in self._ds[i:j]), file=test)

            n += 1


class Logger:
    def __init__(self, basedir):
        self._dir = basedir
        # Create directory (assuming it's not existing)
        os.mkdir(basedir)

    def out_file_timing(self, k):
        return os.path.join(self._dir, f'timing{k}.txt')

    def out_fit_train(self, k):
        return os.path.join(self._dir, f'fit_train_{k}.txt')

    def out_fit_test(self, k):
        return os.path.join(self._dir, f'fit_test_{k}.txt')
    
    def out_log_stdout(self, k):
        return os.path.join(self._dir, f'log{k}.stdout')

    def out_log_stderr(self, k):
        return os.path.join(self._dir, f'log{k}.stderr')

    def out_log_model_stderr(self, k):
        return os.path.join(self._dir, f'mod_log{k}.stderr')

    def get_mod_file(self, n, k):
        '''Path of the mod file for combination n and cross validation k'''
        return os.path.join(self._dir, f'mod_{n}_{k}.dat')

    def open_log_stdout(self, k):
        return open(self.out_log_stdout(k), 'at')

    def open_log_stderr(self, k):
        return open(self.out_log_stderr(k), 'at')

    def open_log_model_stderr(self, k):
        return open(self.out_log_model_stderr(k), 'at')


class Runner:
    '''Class responsible to perform runs'''

    def __init__(self, algo, dataset, out_dir, bin_dir, error_measure='RMSE'):
        self._algo = algo
        self._ds = dataset
        self._dir = out_dir
        self._err = error_measure
        self._bin_path = bin_dir

    def run(self, k, mods, n_gens, logger):
        '''Run the simulation for the k-th CV fold, and return '''
        ds_train, ds_test = self._ds.get_fold_path(k)
        print('Running simulation', k, 'with models', mods, 'on datasets', ds_train, ds_test, 'for', n_gens, 'generations')

        # The number of generations goes into config file

        if self._algo == 'mauro':
            return self._run_with_temp_files(k, mods, n_gens, logger)
        else:
            return self._run_direct(k, mods, n_gens, logger)

    def _run_with_temp_files(self, k, mods, n_gens):
        '''Runs algorithms that write to fixed paths'''
        raise NotImplementedError('Not implemented yet! Why are you not using the faster algos? :P')

    def _run_direct(self, k, mods, n_gens, logger):
        '''Run algorithms that write to custom paths'''

        dsdir = self._ds.get_out_path()
        if_train = self._ds.get_train_path(k)
        if_test = self._ds.get_test_path(k)

        log_mode = 'wt' # We overwrite because in the end we need only the last run (FIXME?)

        # Run the models to get semantics
        mod_sems = []
        for n, mod in enumerate(mods):
            mod_file = logger.get_mod_file(n, k)
            with open(mod_file, 'w') as omod, logger.open_log_model_stderr(k) as merr:
                subprocess.run([mod, if_train, if_test], stdout=omod, stderr=merr)
            mod_sems.append(mod_file)

        run_args = [os.path.join(self._bin_path, self._algo),
            '-error_measure', self._err,
            '-out_file_exec_timing', logger.out_file_timing(k),
            '-out_file_train_fitness', logger.out_fit_train(k),
            '-out_file_test_fitness', logger.out_fit_test(k),
            '-train_file', if_train,
            '-test_file', if_test,
            '-max_number_generations', n_gens,
            *mod_sems,
            ]
        #print('RUNNING', f'{bin_path}/{algo}', *run_args, '2>', log_path)
        with logger.open_log_stdout(k) as lout, logger.open_log_stderr(k) as lerr:
            subprocess.run([str(a) for a in run_args], stdout=lout, stderr=lerr)

        # Get the last fitness value
        with open(logger.out_fit_train(k)) as ftrain:
            last_line = None
            for row in ftrain:
                last_line = row
            train_fit = float(last_line)
        with open(logger.out_fit_test(k)) as tfit:
            last_line = None
            for row in tfit:
                last_line = row
            test_fit = float(last_line)
        return train_fit, test_fit


def run_sim(args, dataset, out_dir):
    # A friendly reminder
    logger_other.info(f'We are going to run {args.k_fold * 2**len(models)} times the short version and save output to {out_dir}')

    # Prepare for running
    runner = Runner(args.algorithm, dataset, out_dir=out_dir, bin_dir=args.bindir)

    logger_selection = Logger(os.path.join(out_dir, 'selection'))
    logger_longrun = Logger(os.path.join(out_dir, 'longrun'))

    # Where fitnesses are saved for CV
    cv_fits = []

    if args.all:
        best_models = models2[-1] # Use all models 
        global_stats['sel_time'] = global_stats.get('sel_time', 0)
    else:
        if False:
            #with open('salvato_risultati', 'rb') as fpi:
            #    cv_fits = pickle.load(fpi)
            pass
        else:
            t_start = time.perf_counter()
            # Perform k-fold cross validation
            for k in range(args.k_fold):
                ## Split the dataset and get two file names
                #ds_train, ds_test = dataset.get_fold_path(k)
                # Where fitnesses are saved for this fold
                k_fits = []
                # For every combination of models
                for mods in models2:
                    # Run simulation gathering results
                    train_fit, test_fit = runner.run(k, mods, args.shortg, logger_selection)
                    # Accumulate fitnesses for this fold
                    k_fits.append((train_fit, test_fit))
                # Accumulate fitnesses for cross validation
                cv_fits.append(k_fits)
            t_tot = time.perf_counter() - t_start
            logi('stats.selection.walltimes', f'Time for running selection: {t_tot}')
            global_stats['sel_time'] = global_stats.get('sel_time', 0) + t_tot

            # Save for convenience
            #with open('salvato_risultati', 'wb') as fpo:
            #    pickle.dump(cv_fits, fpo)

        logi('stats.selection.cv.fitness.average', f'Average fitnesses of CV tests (models combinations on rows)\n{cv_average(cv_fits)}')

        t_start = time.perf_counter()
        # Compute cross validation
        bm = best_cv(cv_fits)
        best_models = models2[bm]

        logi('stats.selection.models.best', f'{bm} {best_models}')
        global_stats['best_models'] = global_stats.get('best_models', Counter()) + Counter({bm: 1})

    # TODO testare l'argomento --all

    # Perform long run, using only selected models
    k_fits = []
    for k in range(args.k_fold):
        #ds_train, ds_test = dataset.get_fold_path(k)
        train_fit, test_fit = runner.run(k, best_models, args.longg, logger_longrun)
        k_fits.append((train_fit, test_fit))

    logi('stats.longrun.cv.fitness.average', f'Average CV: {cv_average(k_fits)}')
    t_tot = time.perf_counter() - t_start
    logi('stats.longrun.walltimes', f'Total time for longruns: {t_tot}')
    global_stats['lon_time'] = global_stats.get('lon_time', 0) + t_tot

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='Run tests with CV model selection')
    parser.add_argument('--all', type=bool, default=False, help='Use all models without selection')
    parser.add_argument('--k_fold', '-k', type=int, default=10, help='Number of folds')
    parser.add_argument('--runs', '-r', type=int, default=30, help='Number of runs')
    parser.add_argument('--config', '-C', type=str, default=None, help='Configuration file to use')
    parser.add_argument('--bindir', '-B', type=str, default='..', help='Directory containing binaries')
    parser.add_argument('--shortg', '-s', type=int, default=100, help='Number of generations for short runs')
    parser.add_argument('--longg', '-l', type=int, default=1000, help='Number of generations for long runs')
    parser.add_argument('--algorithm', '-A', type=str, default='go-gsgp-cpu', help='Path of algo to use')
    parser.add_argument('datafile', type=open, help='Dataset file to load')
    #parser.add_argument('modeldir', type=str, help='Models directory')
    parser.add_argument('outdir', type=str, help='Output directory')
    args = parser.parse_args()

    # Models we are applying
    models = ['../models/nl_lr_sem.py', '../models/nl_mlp_sem.py', '../models/nl_svr_sem.py']
    models2 = list(powerset(models))

    # Create root directory for all the results
    os.mkdir(args.outdir)

    logging.basicConfig(filename=os.path.join(args.outdir, 'stats.log'), level=logging.INFO)

    logger_stats = logging.getLogger('stats')
    logger_other = logging.getLogger('other')

    # If provided, copy configuration file
    if args.config is not None:
        cfg = os.path.join(args.outdir, 'configuration.ini')
        shutil.copy(args.config, cfg)
        args.config = cfg # Replace old choice
        print(f'This is the time to review your configuration file in {cfg}')
        print(f'Generation counts will be {args.shortg} (short) and {args.longg} (long)')
        input(f'Press Enter when ready to go.')

    # Load the dataset and prepare the k-fold
    dataset = Dataset(args.datafile, args.k_fold, out_dir=args.outdir)

    for r in range(args.runs):
        # FIXME one log per run, then use log module
        # FIXME this breaks when using paths like '../somedir'
        outdir = os.path.join(args.outdir, os.path.basename(args.outdir) + str(r))
        # Create output directory
        os.mkdir(outdir)
        # Run algo
        run_sim(args, dataset, outdir)

    logi('stats.walltimes', f'Total selection and longrun wallclock time: {global_stats["sel_time"]} {global_stats["lon_time"]}')
    logi('stats.counters', f'')
    logi('stats.selection.models.frequency', f'{global_stats["best_models"]}')

