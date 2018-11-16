import os
import json
import pickle
import argparse
import numpy as np
import pandas as pd
from decimal import Decimal
from pathlib import Path
from cycler import cycler
from matplotlib import pyplot as plt
from .runner import zopen, powerset
from scipy.stats import mannwhitneyu, ks_2samp
from itertools import count, combinations
from statsmodels.stats.diagnostic import lilliefors


# Use cache files if available?
pickle_cache = True

def robust_mad(a, c=.6745, axis=0):
    a = np.asarray(a)
    center = np.median(a, axis=axis)
    try:
        return np.median((np.abs(a-center))/c, axis=axis)
    except TypeError:
        c = Decimal(c)
        return np.median((np.abs(a-center))/c, axis=axis)


def use_grayscale_print_style():
    plt.style.use('grayscale')
    plt.rcParams['figure.dpi'] = 200
    plt.rcParams['lines.linewidth'] = 2.0
    plt.rcParams['font.size'] = 12

    # Create cycler object. Use any styling from above you please
    monochrome = (cycler('color', ['k']
                         ) * cycler('linestyle', ['-', '--', ':', '-.']))
    # fill_cycle = (cycler('hatch', ['///', '--', '...','\///', 'xxx', '\\\\'],
    #                     ) * cycler('color', 'w'))

    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.linewidth'] = 0.5

    plt.rcParams['axes.prop_cycle'] = monochrome
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.bottom'] = False
    plt.rcParams['axes.spines.left'] = False


def shp(obj):
    """Get the shape of an object (for debugging)."""
    if obj is None:
        return 'None'
    if isinstance(obj, tuple):
        return f'({len(obj)}|' + ','.join(shp(a) for a in obj) + ')'
    if isinstance(obj, list):
        return f'[{len(obj)}|' + ','.join(shp(a) for a in obj) + ']'
    try:
        return str(obj.shape)
    except:
        return '0'


def scatter(title, labels, series):
    '''scatter({'foo': ([x0, x1, x2], [y0, y1, y2]), 'bar': (X, Y)}).
    Optionally with standard deviations.'''
    plt.figure()
    if title is not None:
        plt.title(title)
    names = []
    for k, v in series.items():
        x, y, s = v
        if x is None:
            x = np.arange(y.shape[0])
        plt.plot(x, y)
        plt.fill_between(x, y - s, y + s, alpha=0.5)
        names.append(k)
    plt.legend(names)
    plt.ylim(ymin=0)
    if labels is not None:
        plt.xlabel(labels[0])
        plt.ylabel(labels[1])


def parse_duration(s):
    """Parse golang duration string.

    Parse a string of a Golang duration into
    a float representing seconds.
    """
    import re
    s = s.strip()

    # Tuples pattern-multiplier(s)
    patterns = [
        ('(0)s', 1,),
        ('(\\d+)ns', 1e-9,),
        ('(\\d+\\.\\d+)ns', 1e-9,),
        ('(\\d+)µs', 1e-6,),
        ('(\\d+\\.\\d+)µs', 1e-6,),
        ('(\\d+)ms', 1e-3,),
        ('(\\d+\\.\\d+)ms', 1e-3,),
        ('(\\d+)s', 1,),
        ('(\\d+\\.\\d+)s', 1,),
        ('(\\d+)m(\\d+)s', 60, 1),
        ('(\\d+)m(\\d+\\.\\d+)s', 60, 1),
        ('(\\d+)h(\\d+)s', 3600, 1),
        ('(\\d+)h(\\d+\\.\\d+)s', 3600, 1),
        ('(\\d+)h(\\d+)m(\\d+)s', 3600, 60, 1),
        ('(\\d+)h(\\d+)m(\\d+\\.\\d+)s', 3600, 60, 1),
    ]
    for p, *x in patterns:
        m = re.match(p+'$', s, flags=re.ASCII)
        if m is not None:
            tot = 0
            for g, multiplier in enumerate(x, 1):
                tot += float(m.group(g)) * multiplier
            return tot
    return float(s) * 0.001  # Try matching a pure number of mulliseconds


def load_cv_data(path):
    """Load CV data of a single run."""
    fit_train = []
    fit_test = []
    timing = []

    for p in count(0):
        ftp = os.path.join(path, f'fit_train_{p}.txt')
        # Break if there is no file
        if not os.path.isfile(ftp):
            break
        ft = pd.read_table(ftp, names=['fit'], header=None)
        fit_train.append(ft.values.flatten())

        ftp = os.path.join(path, f'fit_test_{p}.txt')
        ft = pd.read_table(ftp, names=['fit'], header=None)
        fit_test.append(ft.values.flatten())

        tp = os.path.join(path, f'timing{p}.txt')
        with open(tp) as tpf:
            timing.append([parse_duration(l.strip()) for l in tpf])

    # Cross validation mean
    fit_train = np.array(fit_train).mean(axis=0)
    fit_test = np.array(fit_test).mean(axis=0)
    timing = np.array(timing).mean(axis=0)

#    fit_train = np.minimum(fit_train, 1000)
#    fit_test = np.minimum(fit_test, 1000)

    return fit_train, fit_test, timing


def load_runs(prefix_path, sub_name, sub_prefix='{prefix}{r}'):
    """Load runs data, returns dictionary with both raw and processed data.

    Raw data are tables with one row per run.
    """
    # Get simulation name
    prefix = os.path.basename(prefix_path)
    # Lists for data
    cpu_train, cpu_test, cpu_timing = [], [], []
    # Iterate till there are runs
    for r in count(0):
        sp = sub_prefix.format(prefix=prefix, r=r)  # e.g. selection3
        # Path of the sub
        run_path = os.path.join(prefix_path, sp, sub_name)
        print('Loading runs path', run_path)
        if not os.path.isdir(run_path):
            print('Not found, aborting at run', r)
            break
        # Load data
        train, test, timing = load_cv_data(run_path)
        cpu_train.append(train)
        cpu_test.append(test)
        cpu_timing.append(timing)

    print("Averaging", len(cpu_train), "results")
    cpu_train = np.array(cpu_train)
    cpu_test = np.array(cpu_test)
    cpu_timing = np.array(cpu_timing)

    return {
        'raw_train': cpu_train,
        'raw_test': cpu_test,
        'raw_timing': cpu_timing,
        'train': (cpu_train.mean(axis=0),
                  cpu_train.std(axis=0),
                  np.median(cpu_train, axis=0),
                  robust_mad(cpu_train, axis=0)),
        'test': (cpu_test.mean(axis=0),
                 cpu_test.std(axis=0),
                 np.median(cpu_test, axis=0),
                 robust_mad(cpu_test, axis=0)),
        'timing': (cpu_timing.mean(axis=0),
                   cpu_timing.std(axis=0),
                   np.median(cpu_timing, axis=0),
                   robust_mad(cpu_timing, axis=0)),
    }


def load_nested_runs(prefix_path, sub_name, inner_sub_name):
    """Load nested runs data.

    some_path/prefix/prefixN/sub_name/sub_nameM/inner_sub_name
    """
    pass


def uses_selection(stats, name):
    """Return True if the specified test uses selection."""
    c1 = not stats[name]['args'].get('none', False)
    c2 = not stats[name]['args'].get('all', False)
    return c1 and c2


def load_semantic_avg(name, run, dataset):
    """Return the average semantic from the specified longrun."""
    return np.loadtxt(f'{name}/{name}{run}/longrun/sem_{dataset}_avg.txt')


def load_contributions(stats, run, contrib_files):
    """Loads contributions of all models for one run."""
    # Caricare la combinazione vincente di modelli per questo run (bm)
    nm = len(stats['models'])  # Number of models
    models2_ind = list(powerset(range(nm)))  # Combinations
    bmc = stats['bm_hist'][run]  # Best models combination in this run

    # contribs = []
    full_contribs = []
    for path in contrib_files:
        with zopen(path, 'rt') as fp:
            full_contrib = []
            # contrib = []
            for line in fp:
                # Contribution at one time step
                contr = [Decimal(v) for v in line.split(',')]
                # contrib.append(contr)
                # Each data we read has length eq to number of best models +1
                assert len(contr) == len(models2_ind[bmc]) + 1
                # Map contribution to the entire set of models
                full_contr = [0] * (nm + 1)  # Contribution counters
                full_contr[0] = contr[0]  # Genetic Programming is first
                # models2_ind[i] is a tuple containing the ID of used models
                for i, n in zip(models2_ind[bmc], contr[1:]):
                    full_contr[i+1] = n
                full_contrib.append(full_contr)  # One line
            # contribs.append(contrib)
            full_contribs.append(full_contrib)  # One file

    return full_contribs


def load_semantic_files(sem_files):
    """Given a list of paths, return loaded semantics, as nested lists.

    Returned list has length len(sem_files), one item per semantic.
    Each semantic has length R, where is the number of points in time (rows).
    Each row has length S, the number of values in the semantic (dataset).
    """
    sem = []
    for path in sem_files:
        print('Loading semantic file', path)
        with zopen(path, 'rt') as fp:
            sem.append([[float(v) for v in l.split(',')] for l in fp])
    return sem


def load_sem_distance_avg(name, run, k_folds, dataset):
    """Return the first-last semantic distance for the specified run.

    Distances are computed as L² norm of differences between first and last
    semantic vectors for each fold, then averaged along folds to yield a
    a real value for each time step in the data series.
    """
    # Load all semantic data
    # Prepare list of files containing semantic data
    sem_files = (f'{name}/{name}{run}/longrun/sem_{dataset}_{k}.txt.gz'
                 for k in range(k_folds))
    # Load semantic data with shape:
    # sem.shape = (k_folds, time_steps, dataset_size)
    sem = np.array(load_semantic_files(sem_files))
    assert k_folds == sem.shape[0]
    # To compute average distances, first compute distances for every run
    # Distance for time t is sum((sem[:,t,:] - sem[:,0,:])**2)**0.5
    t0 = np.repeat(sem[:, 0, :], sem.shape[1], axis=0).reshape(sem.shape)
    sd = (sem - t0) ** 2  # Squared differences
    # Sum along semantic axis and take square root
    # yielding shape ac.shape == (k_folds, time_steps)
    ac = np.sqrt(sd.sum(axis=2))
    # Compute average along rows (k_folds)
    return ac.mean(axis=0)


def load_avg_sem_distance_data(name, n_runs, k_folds, dataset):
    """Loads average semantic distance for specified dataset in a list."""
    return np.array([load_sem_distance_avg(name, i, k_folds, dataset)
                     for i in range(n_runs)])


# def load_avg_sem_distance_avg(name, n_runs, dataset):
#     """Computes average semantic distance for specified dataset."""
#     tot = load_sem_distance_avg(name, 0, dataset)
#     for i in range(1, n_runs):
#         tot += load_sem_distance_avg(name, i, dataset)
#     return tot / n_runs


# def semantic_distance(s1, s2):
#     """Compute L2-norm between two semantics."""
#     return sum((s1 - s2) ** 2)


# def load_semantic(name, run, fold, dataset):
#     """Return the semantic from the specified fold in a longrun."""
#     sem = []
#     path = f'{name}/{name}{run}/longrun/sem_{dataset}_{fold}.txt'
#     with open(path, 'rt') as fp:
#         for l in fp:
#             sem.append(np.array([float(v) for v in l.split(',')]))
#     return sem


# def load_sem_distance(name, run, fold, dataset):
#     """Load the k-fold semantic and compute distance among
#     first semantic and all the others, returning it."""
#     sem = load_semantic(name, run, fold, dataset)
#     return [semantic_distance(sem[0], s) for s in sem]


# def load_folded_sem_distance(name, run, n_folds, dataset):
#     """Return the k-fold average of semantic distance."""
#     # Load semantic of
#     dist = np.array(load_sem_distance(name, run, 0, dataset))
#     # Compute average distances
#     for i in range(1, n_folds):
#         dist += np.array(load_sem_distance(name, run, i, dataset))
#     return dist / n_folds


# def load_avg_sem_distance(name, n_runs, n_folds, dataset):
#     """Return the total average of semantic distance."""
#     dist = sum(load_folded_sem_distance(name, r, n_folds, dataset)
#                for r in range(n_runs))
#     return dist / n_runs


def save_img(path):
    """Pack layout, save it and close the image."""
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def show_img(unused):
    plt.tight_layout()
    plt.show()
    plt.close()


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Read results and generate plots and statistics")
    parser.add_argument('-d', '--dir', nargs=2, action='append',
                        metavar=('name', 'directory'),
                        help='name-directory pair to load. The first will be'
                        'used as reference sample when performing U-tests')
    parser.add_argument('-m', '--median', action='store_true',
                        help='Use median instead of mean to average data and MAD as dispersion')
    parser.add_argument('-t', '--use-title', action='store_true',
                        help='Add a title to plots')
    parser.add_argument('-c', '--use-color', action='store_true',
                        help='Enable colorful plots (for screens)')
    parser.add_argument('-w', '--extra-wide', action='store_true',
                        help='Enable extra-wide plots')
    parser.add_argument('-g', '--gui', action='store_true',
                        help='Show plots in a GUI instead of saving to file')
    parser.add_argument('-p', '--p-value', type=float, default=0.01,
                        help='Change p-value used in statistical tests')
    parser.add_argument('-b', '--bins', type=int, default=30,
                        help='Number of bins when plotting histograms')
    parser.add_argument('prefix', help='Prefix for output files')
    return parser.parse_args()


def load_stats(out_dirs):
    """Load global statistics for each directory."""
    stats = {}
    for name in out_dirs:
        with open(os.path.join(name, 'stats.json')) as statsfile:
            stats[name] = json.load(statsfile)
            stats[name]['models2'] = list(powerset(stats[name]['models']))
    return stats


def load_all_data(stats, out_dirs):
    """Load all the data from a structured directory."""
    # Load data
    all_data = {}
    for name in out_dirs:
        pfile = f'{name}_all_data.pkl'
        if pickle_cache and os.path.exists(pfile):
            print('Found existing file', pfile, 'loading it')
            # Load previously saved data
            with open(pfile, 'rb') as fp:
                all_data[name] = pickle.load(fp)
        else:
            print('Pickled file', pfile, 'not found, creating')
            runs = stats[name]['args']['runs']
            k_folds = stats[name]['args']['k_folds']
            all_data[name] = {}
            all_data[name]['longrun'] = load_runs(name, 'longrun')
            # Ensure we loaded data for the specified number of runs
            assert all_data[name]['longrun']['raw_train'].shape[0] == runs
            if uses_selection(stats, name):
                sel_data = []
                print('Using selection on models', stats[name]['models2'])
                for m, mods in enumerate(stats[name]['models2']):
                    # For each model, load all run data
                    mod_data = []
                    print('Loading model data for model set', mods, 'expected runs', runs)
                    for r in range(runs):
                        # Get base path of r-th run
                        # somerun/somerun0/selection/selection0/shortrun
                        run_path = Path(name) / f'{name}{r}' / 'selection' / f'selection{m}'
                        if not run_path.exists():
                            print('NOT FOUND SKIP')
                            continue
                        #run_path = os.path.join(name,
                        #                        f'{name}{r}',
                        #                        'selection',
                        #                        f'selection{m}')
                        print('RUN PATH', run_path)
                        # Using '{prefix}_{r}' to produce
                        # selectionM/selectionM_K
                        data = load_runs(run_path, 'shortrun', '{prefix}_{r}')
                        mod_data.append(data)
                        # In every run, we perform k-fold CV, and for each fold
                        # we perform a nested run with j-fold CV
                        print('Data raw train shape', data['raw_train'].shape, 'kfolds', k_folds)
                        assert data['raw_train'].shape[0] == k_folds
                    sel_data.append(mod_data)
                all_data[name]['selection'] = sel_data
                # print('Selection data', sel_data)
            # Save data to file for convenience
            with open(pfile, 'wb') as fp:
                pickle.dump(all_data[name], fp)
    # Return loaded data
    return all_data


def load_all_contribs(stats, out_dirs):
    """Return a dictionary with contribution data for each out dir.

    raw_contribs data has shape (n_runs, k_folds, time_steps, models_used)."""

    all_contribs = {}
    for name in out_dirs:
        pfile = f'{name}_contrib_data.pkl'
        if pickle_cache and os.path.exists(pfile):
            print('Found existing file', pfile, 'loading it')
            with open(pfile, 'rb') as fp:
                try:
                    # Uncomment this when working with old data
                    #while True:
                    #    subj = pickle.load(fp)
                    #    data = pickle.load(fp)
                    #    all_contribs[subj] = data
                    # New data version, right!
                    while True:
                        name, key, data = pickle.load(fp)
                        all_contribs.setdefault(name, {})[key] = data
                except EOFError:
                    pass
        else:
            n_runs = stats[name]['args']['runs']
            k_folds = stats[name]['args']['k_folds']
            # Load contribution data
            contrib_data = []
            for run in range(n_runs):
                # Load a contributions for each fold into a list
                contr_files = (f'{name}/{name}{run}/longrun/contribs_{k}.txt.gz'
                               for k in range(k_folds))
                contrib_data.append(load_contributions(stats[name],
                                                       run,
                                                       contr_files))
            # Save raw data for all the runs
            # Shape is (n_runs, k_folds, time_steps, models_used)
            cd = np.array(contrib_data, dtype='object')
            all_contribs[name] = {'raw_contribs': cd}
            # Average across k-folds
            all_contribs[name]['contribs'] = np.mean(cd, axis=1)
            with open(pfile, 'wb') as fp:
                for key in all_contribs[name]:
                    pickle.dump((name, key, all_contribs[name][key]), fp)
                # pickle.dump(all_contribs, fp)
    return all_contribs


def compute_and_print_lilliefors(all_data_raw, label, p_value):
    # Get last fitness train values
    samples = all_data_raw[:, -1]
    # Run lilliefors normality test
    stat, pval = lilliefors(samples)
    if pval < p_value:
        print(f'{label} IS NOT normally distributed (p={pval})')
    else:
        print(f'{label} IS normally distributed (p={pval})')
    return samples


def compute_and_print_tests(data_h0, data_h1, label, p_value):
    stat, pval = mannwhitneyu(data_h0, data_h1, alternative='two-sided')
    print(f'MWW U-test {label} U: {stat}, p-value: {pval}')
    if pval < p_value:
        print(f'  p-value lt {p_value}, DIFFERENT distributions')
    else:
        print(f'  p-value gt {p_value}, SAME distributions')

    stat, pval = ks_2samp(data_h0, data_h1)
    print(f'Kolmogorov-Smirnov test {label} KS: {stat}, p-value: {pval}')
    if pval < p_value:
        print(f'  p-value lt {p_value}, DIFFERENT distributions')
    else:
        print(f'  p-value gt {p_value}, SAME distributions')


def main():
    """Read data and produce statistics."""
    # old_cycler = plt.rcParams['axes.prop_cycle']
    args = parse_arguments()
    better_names, out_dirs = zip(*args.dir)

    # Strip trailing / from out_dirs
    out_dirs = [d.rstrip('/') for d in out_dirs]

    prefix = args.prefix
    use_mean = not args.median

    # Use a title on the plots
    ut = args.use_title

    # Decent names for files (I used lower, but can be changed)
    better_names_files = [n.lower() for n in better_names]

    # Load style
    if not args.use_color:
        use_grayscale_print_style()
    else:
        # A decent style for screens
        plt.style.use('ggplot')

    if args.extra_wide:
        plt.rcParams['figure.figsize'] = (15, 5)
    else:
        plt.rcParams['figure.figsize'] = (6, 3)

    if args.gui:
        # plt.ion()  # Interactive mode on, to show multiple images at once
        render = show_img
    else:
        render = save_img

    def indices(data):
        # Indices to compute depending on user choice
        if use_mean:
            central = np.mean(data, axis=0)
            std = np.std(data, axis=0)
        else:
            central = np.median(data, axis=0)
            std = robust_mad(data, axis=0)
        return central, std

    # Build names mapping
    bn = dict(zip(out_dirs, better_names))
    bnf = dict(zip(out_dirs, better_names_files))

    # Load training data from specified directories
    print('Loading train and test data')
    stats = load_stats(out_dirs)
    all_data = load_all_data(stats, out_dirs)

    # Load semantic distances TODO move to function
    print('Loading semantic data')
    sem_evo_trains, sem_evo_tests = {}, {}
    for name in out_dirs:
        pfile = f'{name}_sem_data.pkl'
        if pickle_cache and os.path.exists(pfile):
            print('Found existing file', pfile, 'loading it')
            with open(pfile, 'rb') as fp:
                sem_evo_dat_train, sem_evo_dat_test = pickle.load(fp)
        else:
            n_runs = stats[name]['args']['runs']
            k_folds = stats[name]['args']['k_folds']
            print('Pickled file', pfile, 'not found, creating')
            sem_evo_dat_train = load_avg_sem_distance_data(
                    name, n_runs, k_folds, 'train')
            sem_evo_dat_test = load_avg_sem_distance_data(
                    name, n_runs, k_folds, 'test')
            with open(pfile, 'wb') as fp:
                pickle.dump((sem_evo_dat_train, sem_evo_dat_test), fp)
        print('Computing averages')  # , shp(sem_evo_dat_train))

        # TODO this is not using the indices() function
        if use_mean:
            sem_evo_trains[name] = {'m': np.mean(sem_evo_dat_train, axis=0)}
            sem_evo_tests[name] = {'m': np.mean(sem_evo_dat_test, axis=0)}
            # Standard deviation
            sem_evo_trains[name]['s'] = np.std(sem_evo_dat_train, axis=0)
            sem_evo_tests[name]['s'] = np.std(sem_evo_dat_test, axis=0)
        else:
            sem_evo_trains[name] = {'m': np.median(sem_evo_dat_train, axis=0)}
            sem_evo_tests[name] = {'m': np.median(sem_evo_dat_test, axis=0)}
            # Median Absolute Dispersion
            sem_evo_trains[name]['s'] = robust_mad(sem_evo_dat_train, axis=0)
            sem_evo_tests[name]['s'] = robust_mad(sem_evo_dat_test, axis=0)

    print('Loading contribution data')
    all_contribs = load_all_contribs(stats, out_dirs)
    # Print contributions, both barchart and scatter (vs generations)
    for name in out_dirs: # all_contribs:
        mod_names = ['gp'] + stats[name]['models']
        # Average along runs
        cont_avg, cont_std = indices(all_contribs[name]['contribs'])
        plt.figure()
        plt.plot(cont_avg / cont_avg.sum(axis=1).reshape((-1, 1)))
        plt.title('Contributions for each model while evolving' * ut)
        plt.legend(mod_names)
        plt.xlabel('Generation')
        plt.ylabel('Contribution count')
        # plt.yscale('log')
        print(f'Producing file {prefix}{bnf[name]}_contrib_vs_gen.png')
        render(f'{prefix}{bnf[name]}_contrib_vs_gen.png')

        # Get x-positions for bar chart
        pos = range(cont_avg.shape[1])
        # Consider only last row now
        cont_avg = cont_avg[-1, :]
        cont_std = cont_std[-1, :]
        # Convert percentage (also std)
        cont_avg_p = 100 * cont_avg / sum(cont_avg)
        cont_std_p = 100 * cont_std / sum(cont_avg)
        # Bar plot with contributions
        plt.figure()
        ax = plt.subplot()
        #plt.bar(pos, cont_avg)
        plt.bar(pos, cont_avg_p)
        #plt.bar(pos, cont_avg_p, yerr=cont_std_p)
        plt.title('Contributions at last generation' * ut)
        plt.xlabel('Model')
        plt.ylabel('Contribution %')
        ax.set_xticks(pos)
        ax.set_xticklabels(mod_names)
        render(f'{prefix}{bnf[name]}_contrib_hist.png')

    # Time series plots
    scatter('Train fitness' * ut, ('Generation', 'Fitness'), {
        bn[name]: (None, *indices(all_data[name]['longrun']['raw_train']))
        for name in all_data
    })
    render(f'{prefix}fitness_vs_gen_train.png')

    scatter('Test fitness' * ut, ('Generation', 'Fitness'), {
        bn[name]: (None, *indices(all_data[name]['longrun']['raw_test']))
        for name in all_data
    })
    render(f'{prefix}fitness_vs_gen_test.png')

    scatter('Runtime' * ut, ('Generation', 'Time [s]'), {
        bn[name]: (None, *indices(all_data[name]['longrun']['raw_timing']))
        for name in all_data
    })
    render(f'{prefix}runtime_vs_gen.png')

    scatter('Train Runtime-vs-Fitness (average time per run)' * ut,
            ('Time [s]', 'Fitness'),
            {bn[name]: (all_data[name]['longrun']['timing'][0],
                        *indices(all_data[name]['longrun']['raw_train']))
             for name in all_data})
    render(f'{prefix}fitness_vs_runtime_train.png')

    scatter('Test Runtime-vs-fitness (average time per run)' * ut,
            ('Time [s]', 'Fitness'),
            {bn[name]: (all_data[name]['longrun']['timing'][0],
                        *indices(all_data[name]['longrun']['raw_test']))
             for name in all_data})
    render(f'{prefix}fitness_vs_runtime_test.png')

    # Plot distribution of train fitness
    plt.figure()
    plt.suptitle('Distribution of last train fitness' * ut)
    for i, name in enumerate(all_data):
        ax = plt.subplot(1, len(all_data), i+1)
        data = all_data[name]['longrun']['raw_train']
        # print('Histogram for data with shape', data.shape)
        plt.hist(data[:, -1], args.bins)
        ax.set_title(bn[name])  # Better name as title
    # plt.subplots_adjust(top=0.85)  # Avoid overlapping
    render(f'{prefix}fitness_distribution_train.png')

    plt.figure()
    plt.suptitle('Distribution of last test fitness' * ut)
    for i, name in enumerate(all_data):
        ax = plt.subplot(1, len(all_data), i+1)
        data = all_data[name]['longrun']['raw_test']
        # print('Histogram for data with shape', data.shape)
        plt.hist(data[:, -1], args.bins)
        ax.set_title(bn[name])  # Better name as title
    # plt.subplots_adjust(top=0.85)  # Avoid overlapping
    render(f'{prefix}fitness_distribution_test.png')

    last_train_fit = {}
    last_test_fit = {}
    for name in all_data:
        last_train_fit[name] = compute_and_print_lilliefors(
            all_data[name]['longrun']['raw_train'],
            f'Train {name}',
            args.p_value,
        )
        last_test_fit[name] = compute_and_print_lilliefors(
            all_data[name]['longrun']['raw_test'],
            f'Test {name}',
            args.p_value,
        )
        # Print average values for each dataset
        print(f'Statistics for {name}:')
        print(f'  Train mean:', np.mean(last_train_fit[name]))
        print(f'  Train median:', np.median(last_train_fit[name]))
        print(f'  Train std:', np.std(last_train_fit[name]))
        print(f'  Train mad:', robust_mad(last_train_fit[name]))
        print(f'  Test mean:', np.mean(last_test_fit[name]))
        print(f'  Test median:', np.median(last_test_fit[name]))
        print(f'  Test std:', np.std(last_test_fit[name]))
        print(f'  Test mad:', robust_mad(last_test_fit[name]))
        # Get last fitness train values
        #samples = all_data[name]['longrun']['raw_train'][:, -1]
        #last_train_fit[name] = samples
        # Run lilliefors normality test
        #stat, pval = lilliefors(samples)
        #if pval < args.p_value:
        #    print(f'Train fitness {name} IS NOT normally distributed (p={pval})')
        #else:
        #    print(f'Train fitness {name} IS normally distributed (p={pval})')
        ## Same for test
        #samples = all_data[name]['longrun']['raw_test'][:, -1]
        #last_test_fit[name] = samples
        #stat, pval = lilliefors(samples)
        #if pval < args.p_value:
        #    print(f'Test fitness {name} IS NOT normally distributed (p={pval})')
        #else:
        #    print(f'Test fitness {name} IS normally distributed (p={pval})')

    # Perform U-tests
    for h0, h1 in sorted(combinations(range(len(out_dirs)), 2)):
        h0, h1 = out_dirs[h0], out_dirs[h1]
        compute_and_print_tests(
            last_train_fit[h0],
            last_train_fit[h1],
            f'Train {h0} - {h1}',
            args.p_value,
        )
        compute_and_print_tests(
            last_test_fit[h0],
            last_test_fit[h1],
            f'Test {h0} - {h1}',
            args.p_value,
        )

    #for name in out_dirs[1:]:
    #    ref = out_dirs[0]
    #    stat, pval = mannwhitneyu(
    #                    last_test_fit[ref],
    #                    last_test_fit[name],
    #                    alternative='two-sided')
    #    print(f'MWW U-test {ref} - {name} U: {stat}, p-value: {pval}')
    #    if pval < args.p_value:
    #        print(f'  p-value lt {args.p_value}, DIFFERENT distributions')
    #    else:
    #        print(f'  p-value gt {args.p_value}, SAME distributions')

    # Plot fitness in wall-clock time
    plt.figure()
    ax = plt.subplot()
    plt.title('Train Runtime-vs-Fitness (wall clock time)' * ut)
    names = []
    for name in all_data:
        names.append(bn[name])
        # In the plot, use average time of a single run
        n_runs = stats[name]['args']['runs']
        k_folds = stats[name]['args']['k_folds']
        j_folds = stats[name]['args']['j_folds']
        # sel_time is the total time for all runs and k_folds
        seltime = stats[name]['sel_time'] / (n_runs * k_folds)
        lontime = stats[name]['lon_time'] / (n_runs * k_folds)
        fitdata = indices(all_data[name]['longrun']['raw_train'])
        runtime = np.linspace(seltime, seltime+lontime, len(fitdata[0]))
        plt.plot(runtime, fitdata[0])
        plt.fill_between(runtime,
                         fitdata[0] - fitdata[1],
                         fitdata[0] + fitdata[1],
                         alpha=0.5)
        if uses_selection(stats, name):
            ax.annotate('End of selection', ha="center",
                        xy=(seltime, fitdata[0][0] * 1.1))
    plt.legend(names)
    plt.ylim(ymin=0)
    plt.xlabel('WC Time [s]')
    plt.ylabel('Fitness')
    render(f'{prefix}fitness_vs_wct_train.png')

    plt.figure()
    ax = plt.subplot()
    plt.title('Test Runtime-vs-Fitness (wall clock time)' * ut)
    names = []
    for name in all_data:
        names.append(bn[name])
        # In the plot, use average time of a single run
        n_runs = stats[name]['args']['runs']
        k_folds = stats[name]['args']['k_folds']
        j_folds = stats[name]['args']['j_folds']
        # sel_time is the total time for all runs and k_folds
        seltime = stats[name]['sel_time'] / (n_runs * k_folds)
        lontime = stats[name]['lon_time'] / (n_runs * k_folds)
        fitdata = indices(all_data[name]['longrun']['raw_test'])
        runtime = np.linspace(seltime, seltime+lontime, len(fitdata[0]))
        plt.plot(runtime, fitdata[0])
        plt.fill_between(runtime,
                         fitdata[0] - fitdata[1],
                         fitdata[0] + fitdata[1],
                         alpha=0.5)
        if uses_selection(stats, name):
            ax.annotate('End of selection', ha="center",
                        xy=(seltime, fitdata[0][0] * 1.1))
    plt.legend(names)
    plt.ylim(ymin=0)
    plt.xlabel('WC Time [s]')
    plt.ylabel('Fitness')
    render(f'{prefix}fitness_vs_wtc_test.png')

    for name in all_data:
        scatter(f'{bn[name]} Train-vs-Test Fitness (Runtime)' * ut,
                ('Time [s]', 'Fitness'),
                {f'{bn[name]} {ds}': (all_data[name]['longrun']['timing'][0],
                                      *indices(all_data[name]['longrun'][ds]))
                 for ds in ['raw_train', 'raw_test']})
        render(f'{prefix}{bnf[name]}_train_vs_test_fitness_runtime.png')

    # Pie charts with running times
    for name in stats:
        n_runs = stats[name]['args']['runs']
        k_folds = stats[name]['args']['k_folds']
        j_folds = stats[name]['args']['j_folds']
        runtimes = stats[name]['sel_time'], stats[name]['lon_time']
        labels = [f'selection ({j_folds}-folds)\n{runtimes[0]/n_runs:.1f}s',
                  f'evolution ({k_folds}-folds)\n{runtimes[1]/n_runs:.1f}s']

        fig, ax = plt.subplots()
        patches = ax.pie([r for r in runtimes],
                         labels=labels,
                         autopct='%1.1f%%',
                         startangle=45)
        for i, p in enumerate(patches[0]):
            p.set_color(['0.8', '0.5'][i])
            p.set_edgecolor('k')
            # p.set_hatch('///' * i)
        # Plot as a circle
        ax.axis('equal')
        title = f'{bn[name]} average running time (selection -vs- evolution)'
        ax.set_title(title * ut)
        render(f'{prefix}{bnf[name]}_average_running_time.png')

    # Selection frequencies
    for name in stats:
        # Best model frequency
        bm = stats[name]['best_models']
        # Models combinations
        mc = stats[name]['models2']
        # Number of combinations
        nm = len(mc)

        # Remove possible suffix _sem from model names
        mc = [[s.rsplit('_sem', 1)[0] for s in m] for m in mc]

        # Prepare data for plotting
        # If powerset was used, show all combinations as bins
        # otherwise, show only winning combinations
        if stats[name]['args']['powerset']:
            x = list(range(nm))
            l = x
            h = [bm.get(str(i), 0) for i in range(nm)]
        else:
            x = list(range(len(bm.keys())))
            l = [int(i) for i in sorted(bm.keys())]
            h = [bm.get(i) for i in sorted(bm.keys())]
        # Convert to relative values
        tot = sum(h)
        h_rel = [100 * v / tot for v in h]

        # Prepare bar chart
        fig, ax = plt.subplots()
        ax.bar(x, h_rel)

        # Write models in column
        labels = ['\n'.join(mc[i]) if mc[i] else '(none)' for i in l]
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        plt.xlabel('Models combinations')
        plt.ylabel('Selection freq. (%)')

        # Set title
        fig.suptitle(
            f'{name} best models combination absolute frequency' * ut)
        # print('Best models freq', name, bm)
        render(f'{prefix}{bnf[name]}_selection_frequency.png')

    # Plot semantic distances
    for name in out_dirs:
        fig, ax = plt.subplots(1, 1)
        ax.plot(sem_evo_trains[name]['m'])
        ax.plot(sem_evo_tests[name]['m'])
        ax.fill_between(range(sem_evo_trains[name]['m'].shape[0]),
                        sem_evo_trains[name]['m'] - sem_evo_trains[name]['s'],
                        sem_evo_trains[name]['m'] + sem_evo_trains[name]['s'],
                        alpha=0.5)
        # TODO mettere anche qui mediane e deviazioni standard
        title = f'Evolution of semantic distances in time for {name}'
        ax.set_title(title * ut)
        plt.xlabel('Generation')
        plt.ylabel('Semantic distance')  # (L²)')
        plt.legend(['Train', 'Test'])
        render(f'{prefix}{bnf[name]}_sem_dist_evolution.png')


if __name__ == '__main__':
    main()
