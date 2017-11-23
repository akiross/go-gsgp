#!/usr/bin/env python3

"""Run simulations with various options."""

import os
import sys
import gzip
import json
import time
import pickle
import shutil
import logging
import argparse
import subprocess
import numpy as np
from collections import Counter


# Store relevant statistics in a dictionary
# 
global_stats = dict()


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


def best_cv(cv_fits):
    """Given a cv_fits, computes the average for the CV process,
    `then determines which models will to be used using the test
    fitness results. Selects the best of the combinations used"""

    average = row_average(cv_fits)
    tf = average[:,1] # Take only test fitnesses
    return tf.argmin()


def semantic_distance(s1, s2):
    """Return L2 distance between semantics."""
    s1 = np.array(s1)
    s2 = np.array(s2)
    return sum(s1 - s2) ** 0.5


def load_semantic_file(fp):
    """Return the semantic from the specified file-like object."""
    sem = []
    for l in fp:
        sem.append(np.array([float(v) for v in l.split(',')]))
    return sem


def load_avg_semantic(sem_files):
    """Given a list of paths, returns the average semantic."""
    acc = np.array(load_semantic_file(zopen(sem_files[0])))
    for path in sem_files[1:]:
        with zopen(path) as fp:
            acc += np.array(load_semantic_file(fp))
    return acc / len(sem_files)


def load_semantic(name, run, fold, dataset):
    """Return the semantic from the specified fold in a longrun."""
    sem = []
    with open(f'{name}/{name}{run}/longrun/sem_{dataset}_{fold}.txt', 'rt') as fp:
        return load_semantic_file(fp)


def semantic_distance(s1, s2):
    """Compute L2-norm between two semantics."""
    return sum((s1 - s2) ** 2)


def load_sem_distance(name, run, fold, dataset):
    """Load the k-fold semantic and compute distance among first semantic and all the others, returning it."""
    sem = load_semantic(name, run, fold, dataset)
    return [semantic_distance(sem[0], s) for s in sem]


def load_folded_sem_distance(name, run, n_folds, dataset):
    """Return the k-fold average of semantic distance."""
    # Load semantic of
    dist = np.array(load_sem_distance(name, run, 0, dataset))
    # Compute average distances
    for i in range(1, n_folds):
        dist += np.array(load_sem_distance(name, run, i, dataset))
    return dist / n_folds


class Dataset:
    """Represents a dataset to be used with k-fold cross-validation"""
    def __init__(self, datafile, k, randomize=True, out_dir='.'):
        self._ds_obj = datafile
        self._k = k
        self._written = [] # Dataset files
        self._out_path = out_dir

        os.mkdir(os.path.join(out_dir, 'dataset'))

        self._prepare_datasets(k, randomize)

    def get_fold_path(self, i):
        """Returns the name of the i-th train/test files"""
        return self._written[i]

    def get_out_path(self):
        return self._out_path

    def get_train_path(self, k):
        return os.path.join(self._out_path, 'dataset', f'train_{k}.dat')

    def get_test_path(self, k):
        return os.path.join(self._out_path, 'dataset', f'test_{k}.dat')

    def _prepare_datasets(self, k, randomize, force_semantic_consistency=True):
        """Prepare datafiles for k-fold cross validation,
        producing train_*.dat and test_*.dat files in current
        directory.
        """
        from random import shuffle

        # Load the data as a list of strings
        with self._ds_obj as df:
            self._ds = [l.strip().split() for l in df]
        # Number of variables in the dataset
        n_vars = len(self._ds[0]) - 1
        dsl = len(self._ds)  # (total) dataset length
        size, rem = divmod(dsl, k)  # Size of each fold
        # We might want to have semantic consistency among the folds
        # (i.e. all folds of the same size, so we can average their semantics)
        # Check if the number of rows is exactly divisible
        if force_semantic_consistency and rem != 0:
            ok_sizes = [i for i in range(2, dsl // 2) if dsl % i == 0]
            print(f'Dataset has size {dsl} which cannot divided in {k} folds')
            print(f'Here some valid k values you can use:\n{ok_sizes}')
            sys.exit(1) # Do not proceed: we rely on this

        # size = dsl + k - 1 // k

        # Shuffle rows
        if randomize:
            shuffle(self._ds)

        # Generate files
        self.n_train_samples = dsl - size
        self.n_test_samples = size

        n = 0  # Sequence number for split
        # Iterate on starting points of each fold (0, size, 2*size, ...)
        for i in range(0, dsl, size):
            # Ending point in the dataset
            j = min(dsl, i+size)
            # Path of files to write
            train_file, test_file = self.get_train_path(n), self.get_test_path(n)
            self._written.append((train_file, test_file))

            with open(train_file, 'wt') as train, open(test_file, 'wt') as test:
                # TODO move writing to file in a separate procedure to make testing easier
                # Write train dataset
                n_rows = dsl - j + i
                print(n_vars, file=train)
                print(n_rows, file=train)
                print('\n'.join('\t'.join(r) for r in self._ds[:i] + self._ds[j:]), file=train)

                # Write test dataset
                n_rows = j - i
                print(n_vars, file=test)
                print(n_rows, file=test)
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

    def out_log_model_stderr(self, k):
        return os.path.join(self._dir, f'mod_log{k}.stderr')

    def get_mod_file(self, n, k):
        """Path of the mod file for combination n and cross validation k"""
        return os.path.join(self._dir, f'mod_{n}_{k}.dat')

    def open_log_stdout(self, k):
        return open(self.out_log_stdout(k), 'at')

    def open_log_stderr(self, k):
        return open(self.out_log_stderr(k), 'at')

    def open_log_model_stderr(self, k):
        return open(self.out_log_model_stderr(k), 'at')


def file_last_line(path):
    """Return the last line of a file."""
    with open(path) as fp:
        last_line = None
        for row in fp:
            last_line = row
        return last_line


def float_list(css: str):
    """Return the comma-separated-list of floats as a list of floats."""
    return [float(v) for v in css.split(',')]


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
        print('Running simulation', k, 'with models', mods, 'on datasets', ds_train, ds_test, 'for', n_gens, 'generations')

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
            with open(mod_file, 'w') as omod, logger.open_log_model_stderr(k) as merr:
                subprocess.run([mod, if_train, if_test], stdout=omod, stderr=merr)
            mod_sems.append(mod_file)

        run_args = [os.path.join(self._bin_path, self._algo),
            '-config', self._conf_path,
            '-error_measure', self._err,
            '-out_file_exec_timing', logger.out_file_timing(k),
            '-out_file_train_fitness', logger.out_fit_train(k),
            '-out_file_test_fitness', logger.out_fit_test(k),
            '-out_file_train_semantic', logger.out_sem_train(k),
            '-out_file_test_semantic', logger.out_sem_test(k),
            '-train_file', if_train,
            '-test_file', if_test,
            '-max_number_generations', n_gens,
            *mod_sems,
            ]
        #print('RUNNING', f'{bin_path}/{algo}', *run_args, '2>', log_path)
        with logger.open_log_stdout(k) as lout, logger.open_log_stderr(k) as lerr:
            subprocess.run([str(a) for a in run_args], stdout=lout, stderr=lerr)

        # Get the last fitness value
        train_fit = float(file_last_line(logger.out_fit_train(k)))
        test_fit = float(file_last_line(logger.out_fit_test(k)))
        # Get the last semantic value
        # train_sem = float_list(file_last_line(logger.out_sem_train(k)))
        # test_sem = float_list(file_last_line(logger.out_sem_test(k)))
        return train_fit, test_fit  # , train_sem, test_sem


def run_sim(args, dataset, out_dir):
    """Run simulation with a preliminary phase of model selection."""
    # A friendly reminder
    logger_other.info(f'We are going to run {args.k_fold * 2**len(models)} times the short version and save output to {out_dir}')
    logger_other.info(f'Using config file {args.config}')

    # Prepare for running
    runner = Runner(args.algorithm, dataset, out_dir=out_dir, bin_dir=args.bindir, conf_path=args.config)

    logger_selection = Logger(os.path.join(out_dir, 'selection'))
    logger_longrun = Logger(os.path.join(out_dir, 'longrun'))

    # Where fitnesses and semantics are saved for CV
    cv_fits, cv_sems_train, cv_sems_test = [], [], []

    if args.all:
        best_models = models2[-1] # Use all models 
        global_stats['best_models'] = global_stats.get('best_models', Counter()) + Counter({-1: 1})
        global_stats['sel_time'] = global_stats.get('sel_time', 0)
    elif args.none:
        best_models = models2[0] # Use no models 
        global_stats['best_models'] = global_stats.get('best_models', Counter()) + Counter({0: 1})
        global_stats['sel_time'] = global_stats.get('sel_time', 0)
    else:
        t_start = time.perf_counter()
        # Perform k-fold cross validation
        for k in range(args.k_fold):
            ## Split the dataset and get two file names
            #ds_train, ds_test = dataset.get_fold_path(k)
            k_fits = []  # Saved fitnesses for this fold
            k_sems_train, k_sems_test = [], []  # Saved semantics for this fold

            # For every combination of models
            for mods in models2:
                # Run simulation gathering results
                train_fit, test_fit = runner.run(k, mods, args.shortg, logger_selection)
                # Accumulate fitnesses for this fold
                k_fits.append((train_fit, test_fit))
            # Accumulate fitnesses and semantics for cross validation
            cv_fits.append(k_fits)
        t_tot = time.perf_counter() - t_start
        logi('stats.selection.walltimes', f'Time for running selection: {t_tot}')
        global_stats['sel_time'] = global_stats.get('sel_time', 0) + t_tot
        global_stats.setdefault('sel_times', []).append(t_tot)

        logi('stats.selection.cv.fitness.average', f'Average fitnesses of CV tests (models combinations on rows)\n{row_average(cv_fits)}')

        # Compute cross validation
        bm = best_cv(cv_fits)
        best_models = models2[bm]

        # We are relying on the fact that k-folded dataset have same length
        # assert cv_sems_train.shape == (args.k_fold, len(models2), dataset.n_train_samples)

        ## TESTING
        # Does cv average works on cv_sems as well?
        # avg_sem_train = row_average(cv_sems_train)
        # avg_sem_test = row_average(cv_sems_test)
        # print('avg_sem_train shape is', avg_sem_train.shape)

        # bm_sem = 
        # individuare la semantica (media) del migliore individuo
        # e fare la distanza tra quella e quella del migliore individuo man mano che procedo

        logi('stats.selection.models.best', f'{bm} {best_models}')
        global_stats['best_models'] = global_stats.get('best_models', Counter()) + Counter({bm: 1})

    # TODO testare l'argomento --all

    # Best semantic
    # best_train_sem = avg_sem_train[bm]
    # best_test_sem = avg_sem_test[bm]

    # Start timer for long run
    t_start = time.perf_counter()

    # Perform long run, using only selected models
    k_fits = []
    k_sem_train, k_sem_test = [], []
    # k_sem_dist = []
    for k in range(args.k_fold):
        #ds_train, ds_test = dataset.get_fold_path(k)
        # train_fit, test_fit, train_sem, test_sem = runner.run(k, best_models, args.longg, logger_longrun)
        train_fit, test_fit = runner.run(k, best_models, args.longg, logger_longrun)
        k_fits.append((train_fit, test_fit))
        # Compute distance between semantics
        # dtr = semantic_distance(train_sem, best_train_sem)
        # dte = semantic_distance(test_sem, best_test_sem)
        # k_sem_dist.append((dtr, dte))

        # Save files containing semantics
        k_sem_train.append(logger_longrun.out_sem_train(k))
        k_sem_test.append(logger_longrun.out_sem_test(k))

    # Open the semantic files and compute average
    avg_sem_train = load_avg_semantic(k_sem_train)
    avg_sem_test = load_avg_semantic(k_sem_test)
    np.savetxt(logger_longrun.out_avg_sem_train(), avg_sem_train)
    np.savetxt(logger_longrun.out_avg_sem_test(), avg_sem_test)

    # Remove semantic files if necessary
    if not args.keep:
        for sem_fp in k_sem_train + k_sem_test:
            os.remove(sem_fp)

    logi('stats.longrun.cv.fitness.average', f'Average CV: {row_average(k_fits)}')
    t_tot = time.perf_counter() - t_start
    logi('stats.longrun.walltimes', f'Total time for longruns: {t_tot}')
    global_stats['lon_time'] = global_stats.get('lon_time', 0) + t_tot
    global_stats.setdefault('lon_times', []).append(t_tot)

    # Compute average semantic for the cross-validation set
    print('Average of k-folded semantics!', k_sem_train, k_sem_test)

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='Run tests with CV model selection')
    parser.add_argument('--all', type=bool, default=False, help='Use all models without selection')
    parser.add_argument('--none', type=bool, default=False, help='Use no models without selection')
    parser.add_argument('--keep', type=bool, default=False, help='Keep semantic files after computing averages')
    parser.add_argument('--k_fold', '-k', type=int, default=10, help='Number of folds')
    parser.add_argument('--runs', '-r', type=int, default=30, help='Number of runs')
    parser.add_argument('--config', '-C', type=str, default=None, help='Configuration file to use')
    parser.add_argument('--bindir', '-B', type=str, default='..', help='Directory containing binaries')
    parser.add_argument('--modeldir', '-M', type=str, default='..', help='Directory containing models')
    parser.add_argument('--shortg', '-s', type=int, default=100, help='Number of generations for short runs')
    parser.add_argument('--longg', '-l', type=int, default=1000, help='Number of generations for long runs')
    parser.add_argument('--algorithm', '-A', type=str, default='go-gsgp-cpu', help='Path of algo to use')
    parser.add_argument('datafile', type=open, help='Dataset file to load')
    #parser.add_argument('modeldir', type=str, help='Models directory')
    parser.add_argument('outdir', type=str, help='Output directory')
    args = parser.parse_args()

    # Models we are applying
    # models = ['../models/nl_lr_sem.py', '../models/nl_mlp_sem.py', '../models/nl_svr_sem.py']
    models = []
    for f in os.listdir(args.modeldir):
        fp = os.path.join(args.modeldir, f)
        if os.path.isfile(fp):
            models.append(fp)

    if len(models) > 6:
        print('There are more than 6 models: the process could be very slow.')
        print(models[:3])
        if input('Are you sure you want to continue? (y to go on) ') != 'y':
            sys.exit(0)

    models2 = list(powerset(models))

    # Save arguments
    global_stats['args'] = {k: v for k, v in vars(args).items() if type(v) in [int, float, bool]}
    # Save model names
    mod_names = [os.path.basename(m).split('.')[0] for m in models]
    global_stats['models'] = mod_names
    # Save powerset
    global_stats['models2'] = list(powerset(mod_names))
    # Number of runs to perform
    global_stats['n_runs'] = args.runs
    # Number of k-folds
    global_stats['k_fold'] = args.k_fold

    # Create root directory for all the results
    os.mkdir(args.outdir)

    logging.basicConfig(filename=os.path.join(args.outdir, 'stats.log'), level=logging.INFO)

    logger_stats = logging.getLogger('stats')
    logger_other = logging.getLogger('other')

    # Load the dataset and prepare the k-fold
    dataset = Dataset(args.datafile, args.k_fold, out_dir=args.outdir)

    # If provided, copy configuration file
    # TODO also, read values from the config file instead of using cli arguments
    if args.config is not None:
        cfg = os.path.join(args.outdir, 'configuration.ini')
        shutil.copy(args.config, cfg)
        args.config = cfg # Replace old choice
        print(f'This is the time to review your configuration file in {cfg}')
        print(f'Generation counts will be {args.shortg} (short) and {args.longg} (long)')
        input(f'Press Enter when ready to go.')

        # Save config file to stats, for reference
        with open(cfg, 'rt') as cfgfp:
            global_stats['configuration.ini'] = cfgfp.read()

    # TODO
    # per poter fare le analisi sul tempo, bisogna avere una media delle semantiche
    # dopo tutti i run. per non recuperare le semantiche al momento delle analisi dati
    # conviene fare qui la media e la produzione di un file di output medio che sia facilmente usabile nelle analisi

    for r in range(args.runs):
        # FIXME one log per run, then use log module
        # FIXME this breaks when using paths like '../somedir'
        outdir = os.path.join(args.outdir, os.path.basename(args.outdir) + str(r))
        # Create output directory
        os.mkdir(outdir)
        # Run algo
        run_sim(args, dataset, outdir)
        # Compute average semantic

    logi('stats.walltimes', f'Total selection and longrun wallclock time: {global_stats["sel_time"]} {global_stats["lon_time"]}')
    # logi('stats.counters', f'')

    # Convert counter to dictionary, for easier serialization
    global_stats['best_models'] = {str(k): v for k, v in global_stats['best_models'].items()}
    logi('stats.selection.models.frequency', f'{global_stats["best_models"]}')

    with open(os.path.join(args.outdir, 'stats.json'), 'wt') as statfile:
        json.dump(global_stats, statfile)
        #pickle.dump(statfile, global_stats)

    # Average semantics
