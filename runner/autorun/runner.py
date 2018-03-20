#!/usr/bin/env python3

"""Run simulations with various options."""

import os
import sys
import gzip
import json
import time
import pickle
import logging
import argparse
import subprocess
import numpy as np
from os import mkdir
from random import shuffle
from collections import Counter
from shutil import copy as file_copy


# Store relevant statistics in a dictionary
# 
global_stats = dict()


def subprocess_run(args, **kwargs):
    logi('subprocess', f'Running {" ".join(str(a) for a in args)}')
    return subprocess.run(args, **kwargs)


def fprint(fp, *args):
    print(*args, file=fp)


def zopen(path, mode='rt'):
    """Open the file for reading. If gzipped, open it anyway."""
    if path.endswith('.gz'):
        return gzip.open(path, mode)
    return open(path, mode)


def logi(log, message):
    """Log the message in the specified log."""
    logging.getLogger(log).info(message)


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)."
    from itertools import chain, combinations
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


def row_average(table): 
    """Compute average along first axis (rows)."""
    fits = np.array(table) # Convert to numpy array
    return np.mean(fits, axis=0) # Compute average across rows


def load_semantic_file(fp):
    """Return the semantic from the specified file-like object.
    
    Return value has shape (R, S) where R is the number of rows accumulated in
    time, while S is the size of the semantic (dataset).
    """
    sem = []
    for l in fp:
        sem.append([float(v) for v in l.split(',')])
    return np.array(sem)


def load_avg_semantic(sem_files):
    """Given a list of paths, return the average semantic."""
    sem = []
    for path in sem_files:
        with zopen(path) as fp:
            sem.append(load_semantic_file(fp))
    return row_average(np.array(sem))


# def load_semantic_files(sem_files):
#     """Given a list of paths, return loaded semantics, as nested lists.
# 
#     Returned list has length len(sem_files), one item per semantic.
#     Each semantic has length R, where is the number of points in time (rows).
#     Each row has length S, the number of values in the semantic (dataset).
#     """
#     sem = []
#     for path in sem_files:
#         with zopen(path) as fp:
#             sem.append([float(v) for l in fp for v in l.split(',')])
#     return sem


def load_dataset(path, skip=0):
    """Load a dataset: rows of space-separated values."""
    with zopen(path) as df:
        for _ in range(skip):
            next(df)
        return [l.strip().split() for l in df]


class Dataset:
    """Represents a dataset to be used with k-fold cross-validation.
    It will create a 'dataset' folder in outpath/ and put there the datafiles.
    """

    def __init__(self, datafile, k, outdir, skip_header=0):
        self._datafile = datafile
        self._k = k
        self._written = [] # Dataset files
        self._outdir = os.path.join(outdir, 'dataset')

        mkdir(self._outdir)

        # Load the data as a list of strings
        self._ds = load_dataset(self._datafile, skip_header)
        # self._prepare_datasets(k, randomize)

    def get_fold_path(self, i):
        """Returns the name of the i-th train/test files"""
        return self._written[i]

    def get_out_path(self):
        return self._outdir

    def get_train_path(self, k):
        return os.path.join(self._outdir, f'train_{k}.dat')

    def get_test_path(self, k):
        return os.path.join(self._outdir, f'test_{k}.dat')

    def is_consistent(self):
        """Check if the dataset can be divided in k equal partitions."""
        dsl = len(self._ds)  # (total) dataset length
        k = self._k  # Partitions
        if k > dsl:
            msg = f'Dataset has size {dsl}, k cannot be larger than {dsl//2}.'
            return False, msg

        # We might want to have semantic consistency among the folds
        # (i.e. all folds of the same size, so we can average their semantics)
        # Check if the number of rows is exactly divisible
        if dsl % k != 0:
            ok_sizes = [i for i in range(2, dsl // 2) if dsl % i == 0]
            msg = f'Dataset has size {dsl}, not divisible in {k} folds. '
            msg += f'Some valid K-values: {ok_sizes}'
            return False, msg
        return True, None

    def generate_folds(self, randomize):
        """Prepare datafiles for K-fold cross validation, resampling the data.
        
        Produce train_*.dat and test_*.dat files in output directory.
        If K is not consistent
        If resample is True, a dataset that is not consistent will be resampled
        to the largest consistent subset for the given K.
        """
        k = self._k
        # Number of variables in the dataset
        n_vars = len(self._ds[0]) - 1
        dsl = len(self._ds)  # (total) dataset length

        size, rem = divmod(dsl, k)  # Size of each fold

        # Shuffle all the rows if necessary, **before** resampling
        if randomize:
            shuffle(self._ds)

        if rem != 0:
            # If not divisible, resample excluding reminder
            data = self._ds[:-rem]
            dsl -= rem  # New size
        else:
            # Or use all the data
            data = self._ds

        # Generate files
        self.n_train_samples = dsl - size
        self.n_test_samples = size

        # Iterate on starting points of each fold (0, size, 2*size, ...)
        for k, i in enumerate(range(0, dsl, size)):
            # Ending point in the dataset
            j = min(dsl, i+size)
            # Path of files to write
            train_file = self.get_train_path(k)
            test_file = self.get_test_path(k)
            self._written.append((train_file, test_file))

            # Write train dataset
            n_rows = dsl - j + i
            rows = data[:i] + data[j:]
            assert len(rows) == n_rows
            self._write_dataset(train_file, n_vars, n_rows, rows)

            # Write test dataset
            n_rows = j - i
            rows = data[i:j]
            assert len(rows) == n_rows
            self._write_dataset(test_file, n_vars, n_rows, rows)

    def _write_dataset(self, path, n_vars, n_rows, data):
        """Write data on file."""
        with zopen(path, 'wt') as fp:
            fprint(fp, n_vars)
            fprint(fp, n_rows)
            fprint(fp, '\n'.join('\t'.join(r) for r in data))


class Logger:
    '''Provide paths and utilities for output and log files.'''

    def __init__(self, basedir):
        self._dir = basedir
        # Create directory (assuming it's not existing)
        mkdir(basedir)

    def out_dump_file(self, k):
        return os.path.join(self._dir, f'dump_{k}.proto')

    def out_file_timing(self, k):
        return os.path.join(self._dir, f'timing{k}.txt')

    def out_file_contrib(self, k):
        return os.path.join(self._dir, f'contribs_{k}.txt.gz')

    def out_fit_train(self, k):
        return os.path.join(self._dir, f'fit_train_{k}.txt')

    def out_fit_test(self, k):
        return os.path.join(self._dir, f'fit_test_{k}.txt')

    def out_avg_sem_train(self):
        return os.path.join(self._dir, 'sem_train_avg.txt')

    def out_avg_sem_test(self):
        return os.path.join(self._dir, 'sem_test_avg.txt')

    def out_sem_train(self, k):
        return os.path.join(self._dir, f'sem_train_{k}.txt.gz')
    
    def out_sem_test(self, k):
        return os.path.join(self._dir, f'sem_test_{k}.txt.gz')
    
    def out_log_stdout(self, k):
        return os.path.join(self._dir, f'log{k}.stdout')

    def out_log_stderr(self, k):
        return os.path.join(self._dir, f'log{k}.stderr')

    def out_log_model_stderr(self, n, k):
        return os.path.join(self._dir, f'mod_{n}_{k}.stderr')

    def get_mod_file(self, n, k):
        """Path of the mod file for combination n and cross validation k"""
        return os.path.join(self._dir, f'mod_{n}_{k}.dat')

    def open_log_stdout(self, k):
        return zopen(self.out_log_stdout(k), 'at')

    def open_log_stderr(self, k):
        return zopen(self.out_log_stderr(k), 'at')

    def open_log_model_stderr(self, n, k):
        return zopen(self.out_log_model_stderr(n, k), 'at')


def file_last_line(path):
    """Return the last line of a file."""
    with zopen(path) as fp:
        last_line = None
        for row in fp:
            last_line = row
        return last_line


# def float_list(css: str):
#     """Return the comma-separated-list of floats as a list of floats."""
#     return [float(v) for v in css.split(',')]


class Runner:
    """Class responsible to perform runs."""

    def __init__(self, algo, dataset, out_dir, bin_dir, conf_path='.', error_measure='RMSE'):
        self._algo = algo
        self._ds = dataset
        self._dir = out_dir
        self._err = error_measure
        self._bin_path = bin_dir
        self._conf_path = conf_path

    def run(self, k, mods, n_gens, logger):
        """Run the simulation for the k-th CV fold, and return """
        ds_train, ds_test = self._ds.get_fold_path(k)
        logi('runner', f'Running simulation {k} with models {mods} on datasets {ds_train} {ds_test} for {n_gens} generations')

        # The number of generations goes into config file

        if self._algo == 'mauro':
            return self._run_with_temp_files(k, mods, n_gens, logger)
        else:
            return self._run_direct(k, mods, n_gens, logger)

    def _run_with_temp_files(self, k, mods, n_gens):
        """Runs algorithms that write to fixed paths"""
        raise NotImplementedError('Not implemented yet! Why are you not using the faster algos? :P')

    def _run_direct(self, k, mods, n_gens, logger):
        """Run algorithms that write to custom paths"""

        # Get paths for
        dsdir = self._ds.get_out_path()  # Directory containing output
        if_train = self._ds.get_train_path(k)  # Training dataset
        if_test = self._ds.get_test_path(k)  # Testing dataset

        log_mode = 'wt' # We overwrite because in the end we need only the last run (FIXME?)

        # Run each machine learning algo and get its semantics
        mod_sems = []
        for n, mod in enumerate(mods):
            mod_file = logger.get_mod_file(n, k)
            with zopen(mod_file, 'w') as omod, logger.open_log_model_stderr(n, k) as merr:
                subprocess_run([mod, if_train, if_test], stdout=omod, stderr=merr)
            mod_sems.append(mod_file)

        run_args = [os.path.join(self._bin_path, self._algo),
            '-config', self._conf_path,
            '-error_measure', self._err,
            '-out_file_exec_timing', logger.out_file_timing(k),
            '-out_file_train_fitness', logger.out_fit_train(k),
            '-out_file_test_fitness', logger.out_fit_test(k),
            '-out_file_train_semantic', logger.out_sem_train(k),
            '-out_file_test_semantic', logger.out_sem_test(k),
            '-out_file_contributions', logger.out_file_contrib(k),
            # '-proto_dump', logger.out_dump_file(k),
            '-train_file', if_train,
            '-test_file', if_test,
            '-max_number_generations', n_gens,
            *mod_sems,
            ]
        with logger.open_log_stdout(k) as lout, logger.open_log_stderr(k) as lerr:
            subprocess_run([str(a) for a in run_args], stdout=lout, stderr=lerr)

        # Get the last fitness value
        train_fit = float(file_last_line(logger.out_fit_train(k)))
        test_fit = float(file_last_line(logger.out_fit_test(k)))
        # Get the last semantic value
        # train_sem = float_list(file_last_line(logger.out_sem_train(k)))
        # test_sem = float_list(file_last_line(logger.out_sem_test(k)))
        return train_fit, test_fit  # , train_sem, test_sem


class Forrest:
    '''Run a cross-validated simulation with results saved in outdir.'''

    def __init__(self, name, algo, models, dataset, k_folds, outdir, bindir, config, error_measure='RMSE'):
        self._name = name
        self._models = models
        self._k_folds = k_folds
        self._dataset = dataset
        # Build a logger to give us data files paths
        self._logger = Logger(os.path.join(outdir, name))
        # Setup runner to use the specified algorithm and dataset
        self._runner = Runner(algo,
                              self._dataset,
                              out_dir=outdir,
                              bin_dir=bindir,
                              conf_path=config,
                              error_measure=error_measure)

    def run(self, n_gens):
        '''Run the simulation k times for n_gens and returns results.'''
        k_fits = []  # Stores best fitnesses for each run
        k_timings = []  # Timing for each run
        self._k_sem_train, self._k_sem_test = [], []  # Semantics for each run

        for k in range(self._k_folds):
            logi(f'run.{self._name}', f'Starting CV fold {k}')
            # Run simulation measuring time
            t_start = time.perf_counter()
            fits = self._runner.run(k, self._models, n_gens, self._logger)
            t_end = time.perf_counter() - t_start
            # Build data
            k_fits.append(fits)
            k_timings.append(t_end)
            # Save paths of semantic files produced
            self._k_sem_train.append(self._logger.out_sem_train(k))
            self._k_sem_test.append(self._logger.out_sem_test(k))

        # Open semantic files and compute average
        avg_sem_train = load_avg_semantic(self._k_sem_train)
        avg_sem_test = load_avg_semantic(self._k_sem_test)

        # Return all data
        return k_fits, k_timings, avg_sem_train, avg_sem_test

    def save_files(self, avg_sem_train, avg_sem_test):
        '''Save the semantics on text files.'''
        np.savetxt(self._logger.out_avg_sem_train(), avg_sem_train)
        np.savetxt(self._logger.out_avg_sem_test(), avg_sem_test)

    def clean_sem_files(self, ):
        # Remove semantic files if necessary
        for sem_fp in self._k_sem_train + self._k_sem_test:
            os.remove(sem_fp)

    # FIXME this is not right
    def _write_stats(self, k_fits):
        '''Broken.'''
        logi(f'stats.{self._name}.cv.fitness.average', f'Average CV: {row_average(k_fits)}')
        t_tot = time.perf_counter() - t_start
        logi(f'stats.{self._name}.walltimes', f'Total time for longruns: {t_tot}')
        global_stats['lon_time'] = global_stats.get('lon_time', 0) + t_tot
        global_stats.setdefault('lon_times', []).append(t_tot)
        # Compute average semantic for the cross-validation set
        print('Average of k-folded semantics!', self._k_sem_train, self._k_sem_test)


def load_models(modeldir):
    '''Load models from a directory and return them along with the powerset'''
    models = []
    for f in os.listdir(modeldir):
        fp = os.path.join(modeldir, f)
        if os.path.isfile(fp):
            models.append(fp)

    if len(models) > 6:
        print('There are more than 6 models: the process could be very slow.')
        print(models[:3])
        if input('Are you sure you want to continue? (y to go on) ') != 'y':
            sys.exit(0)

    models2 = list(powerset(models))
    return models, models2


def parse_arguments():
    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Run tests with CV model selection')
    parser.add_argument('--dry', action='store_true',
                        help='Perform a dry run, writing actions on stdout')
    parser.add_argument('--all', action='store_true',
                        help='Use all models without selection')
    parser.add_argument('--none', action='store_true',
                        help='Use no models without selection')
    parser.add_argument('--noask', action='store_true',
                        help='Do not ask for config checking')
    parser.add_argument('--keep', action='store_true',
                        help='Keep semantic files for selection')
    parser.add_argument('--j_folds', '-j', type=int, default=5,
                        help='Number of nested folds')
    parser.add_argument('--k_folds', '-k', type=int, default=5,
                        help='Number of folds')
    parser.add_argument('--runs', '-r', type=int, default=30,
                        help='Number of runs')
    parser.add_argument('--config', '-C', type=str, default=None,
                        help='Configuration file to use')
    parser.add_argument('--bindir', '-B', type=str, default='..',
                        help='Directory containing binaries')
    parser.add_argument('--modeldir', '-M', type=str, default='..',
                        help='Directory containing models')
    parser.add_argument('--shortg', '-s', type=int, default=100,
                        help='Number of generations for short runs')
    parser.add_argument('--longg', '-l', type=int, default=1000,
                        help='Number of generations for long runs')
    parser.add_argument('--algorithm', '-A', type=str, default='go-gsgp-cpu',
                        help='Path of algo to use')
    parser.add_argument('datafile', type=str,
                        help='Dataset file to load')
    parser.add_argument('outdir', type=str,
                        help='Output directory')
    return parser.parse_args()


def write_stats(args, models):
    '''Save some arguments in the global stats, including model names.'''
    # Save arguments
    types = [int, float, bool]  # Types of arguments to save
    args_items = vars(args).items()  # args is a namespace, vars is needed
    global_stats['args'] = {k: v for k, v in args_items if type(v) in types}
    # Save model names
    mod_names = [os.path.basename(m).split('.')[0] for m in models]
    global_stats['models'] = mod_names
    # Save powerset
    global_stats['models2'] = list(powerset(mod_names))
    # Number of runs to perform
    global_stats['n_runs'] = args.runs
    # Number of k-folds
    global_stats['k_folds'] = args.k_folds
    # Number of j-folds
    global_stats['j_folds'] = args.j_folds

def get_run_path(outdir, r):
    '''Return the path for the r-th run, given outdir as base directory.'''
    # Remove trailing slash
    if outdir[-1] == '/':
        outdir = outdir[:-1]
    return os.path.join(outdir, os.path.basename(outdir) + str(r))


def setup_dry_run():
    '''Redefine global functions to have no effect.'''
    global mkdir, zopen, subprocess_run, file_copy, fprint, file_last_line
    global logi, load_dataset

    mkdir = lambda path: print(f'DRY RUN: mkdir({path})')

    from io import BytesIO, StringIO
    def _zo(path, mode='rt'):
        print(f'DRY RUN: zopen({path}, {mode})')
        if 't' in mode:
            return StringIO()
        return BytesIO()
    zopen = _zo

    def _sr(*args, **kwargs):
        print(f'DRY RUN: subprocess.run({args}, {kwargs})')
    subprocess_run = _sr
    file_copy = lambda src, dest: print(f'DRY RUN: copy({src}, {dest})')

    def _fp(fp, *args, **kwargs):
        data = f'{args}, {kwargs}'
        if len(data) > 50:
            data = data[:20] + '...' + data[-20:]
        print(f'DRY RUN: fprint({data})')
    fprint = _fp

    def _fll(path):
        print(f'DRY RUN: file_last_line({path})')
        return '3.14'
    file_last_line = _fll

    logi = lambda k, a: print('DRY RUN LOG:', k, a)

    def _lds(path, skip):
        # Emulate file opening
        _zo(path)
        # 120 rows (120 is divisible by 2, 3, 4, 5, 6, 8, 10, 15), 4 columns
        r, c = 120, 4
        return [[f'{j}.{i+1}' for j in range(i*c, (i+1)*c)] for i in range(r)]
    load_dataset = _lds


def stateller(key, value, description):
    """Write stats in the global dictionary."""
    global_stats[key] = value
    global_stats['description:' + key] = description


def main():
    '''Main function to get a fresh scope.'''
    args = parse_arguments()
    # Setup functions for dry run if necessary, else start logging
    if args.dry:
        print('Performing dry run!')
        setup_dry_run()
    else:
        # Create root directory for all the results
        mkdir(args.outdir)
        # Setup logging
        logging.basicConfig(filename=os.path.join(args.outdir, 'stats.log'),
                            level=logging.INFO)
    # Models we are applying
    models, models2 = load_models(args.modeldir)
    # Save some interesting data in global stats
    write_stats(args, models)

    logger_stats = logging.getLogger('stats')
    logger_other = logging.getLogger('other')

    # If provided, copy configuration file
    # TODO also, read values from the config file instead of using cli arguments
    if args.config is not None:
        cfg = os.path.join(args.outdir, 'configuration.ini')
        file_copy(args.config, cfg)
        args.config = cfg # Replace old choice
        if not args.noask:
            print(f'This is the time to review your configuration file in {cfg}')
            print(f'Generation counts will be {args.shortg} (short) and {args.longg} (long)')
            input(f'Press Enter when ready to go.')

        # Save config file to stats, for reference
        with zopen(cfg, 'rt') as cfgfp:
            global_stats['configuration.ini'] = cfgfp.read()

    # TODO
    # per poter fare le analisi sul tempo, bisogna avere una media delle semantiche
    # dopo tutti i run. per non recuperare le semantiche al momento delle analisi dati
    # conviene fare qui la media e la produzione di un file di output medio che sia facilmente usabile nelle analisi

    for r in range(args.runs):
        print(f'Performing run {r}')
        # Prepare output directory for this run
        outdir = get_run_path(args.outdir, r)  # somepath/sim += /sim{r}
        mkdir(outdir)

        # Prepare dataset in somepath/sim/sim{r}/dataset
        # This single run will have the datafile partitioned in k folds
        dataset = Dataset(args.datafile, args.k_folds, outdir)
        cons, cons_msg = dataset.is_consistent()
        if not cons:
            # Emit a signal if K does not evenly partition the dataset
            print('Warning! Selected K cannot produce consistent semantics!')
            print(cons_msg)
            logi('run.dataset', cons_msg)
        dataset.generate_folds(True)

        # Model selection
        if args.all:
            best_models = models2[-1]  # Use all models 
            bm = len(models2) - 1  # Last combination
            t_tot = 0  # No time spent
        elif args.none:
            best_models = models2[0] # Use no models 
            bm = 0
            t_tot = 0  # No time spent
        else:
            t_start = time.perf_counter()
            # Create somepath/sim/sim{r}/selection
            mkdir(os.path.join(outdir, 'selection'))
            # Nested cross validation fitness
            ncv_fits = []
            # For every combination of models
            for m, mods in enumerate(models2):
                print('Testing performances of models combo', mods)
                # We need to perform J-folded cross-validation for every K-fold
                seldir = os.path.join(outdir, 'selection', f'selection{m}')
                mkdir(seldir)
                avg_j_fits = []
                for k in range(args.k_folds):
                    # Prepare directory for this nested cross validation
                    # somepath/sim/sim{r}/selection/selection{k}
                    nestdir = os.path.join(seldir, f'selection{m}_{k}')
                    mkdir(nestdir)
                    # Build a dataset from the k-th fold train file
                    nested_dataset = Dataset(dataset.get_train_path(k),
                                             args.j_folds, nestdir,
                                             skip_header=2)
                    # Check if J is consistent
                    cons, cons_msg = nested_dataset.is_consistent()
                    if not cons:
                        print('Warning! Selected J produces inconsistent semantics!')
                        print(cons_msg)
                        logi('run.nested_dataset', cons_msg)
                    nested_dataset.generate_folds(True)
                    # Prepare run, using 
                    forrest = Forrest(f'shortrun',
                                      args.algorithm,
                                      mods,
                                      nested_dataset,
                                      args.j_folds,  # Use a different fold number
                                      nestdir,
                                      args.bindir,
                                      args.config)

                    # Run short simulation
                    k_fits, _, _, _ = forrest.run(args.shortg)

                    # Average fitness over J-folds
                    avg_fit = row_average(k_fits)
                    avg_j_fits.append(avg_fit)

                    if not args.keep:
                        # There is no need to keep the semantic files here
                        forrest.clean_sem_files()

                # Compute average for each model
                avg_model_fit = row_average(avg_j_fits)
                ncv_fits.append(avg_model_fit)

            # Compute selection time for this run and save it
            t_tot = time.perf_counter() - t_start

            # Log average selection fitness
            logi('stats.selection.cv.fitness.average', f'Average fitnesses of NCV tests (models combinations on rows)\n{row_average(ncv_fits)}')

            # Use average validation fitness to determine best model
            bm = int(np.array(ncv_fits)[:,1].argmin())
            best_models = models2[bm]

        # Save selection time
        global_stats['sel_time'] = global_stats.get('sel_time', 0) + t_tot
        global_stats.setdefault('sel_times', []).append(t_tot)
        logi('stats.selection.walltimes', f'Time for running selection: {t_tot}')
        # Increment best model usage
        global_stats['best_models'] = global_stats.get('best_models', Counter()) + Counter({str(bm): 1})
        # Save combination
        global_stats.setdefault('bm_hist', []).append(bm)
        logi('stats.selection.models.best', f'{bm} {best_models}')

        print('Performing long run with best models', models2[bm])
        # Prepare simulation, storing data in somepath/sim/sim{r}/longrun
        forrest = Forrest(f'longrun',
                          args.algorithm,
                          models2[bm],
                          dataset,
                          args.k_folds,
                          outdir,
                          args.bindir,
                          args.config)

        # Run simulation
        k_fits, k_timing, avg_sem_train, avg_sem_test = forrest.run(args.longg)

        # Write average semantic data
        forrest.save_files(avg_sem_train, avg_sem_test)

        # Save logs and stats
        logi('stats.longrun.cv.fitness.average', f'Average CV: {row_average(k_fits)}')
        t_tot = sum(k_timing)  # Total time for executing K-fold CV
        logi('stats.longrun.walltimes', f'Total time for longruns: {t_tot}')
        global_stats['lon_time'] = global_stats.get('lon_time', 0) + t_tot
        global_stats.setdefault('lon_times', []).append(t_tot)

    logi('stats.selection.models.frequency', f'{global_stats["best_models"]}')

    with zopen(os.path.join(args.outdir, 'stats.json'), 'wt') as statfile:
        #for k, v in global_stats.items():
        #    print(f'Writing stat {k} = {v}')
        json.dump(global_stats, statfile)
        #pickle.dump(statfile, global_stats)


if __name__ == '__main__':
    main()
