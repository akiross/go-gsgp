import os
import sys
import json
import pickle
import argparse
import numpy as np
import pandas as pd
from cycler import cycler
from itertools import count
from matplotlib import pyplot as plt

# plt.style.use('ggplot')
# plt.rcParams['figure.figsize'] = (15, 5)


def use_grayscale_print_style():
    plt.style.use('grayscale')
    plt.rcParams['figure.figsize'] = (6, 3)
    plt.rcParams['figure.dpi'] = 200
    plt.rcParams['lines.linewidth'] = 2.0
    plt.rcParams['font.size'] = 12

    # Create cycler object. Use any styling from above you please
    monochrome = (cycler('color', ['k']
                         ) * cycler('linestyle', ['-', '--', ':', '=.']))
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
    """Load runs data."""
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
                  np.median(cpu_train, axis=0)),
        'test': (cpu_test.mean(axis=0),
                 cpu_test.std(axis=0),
                 np.median(cpu_test, axis=0)),
        'timing': (cpu_timing.mean(axis=0),
                   cpu_timing.std(axis=0),
                   np.median(cpu_timing, axis=0)),
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


def load_sem_distance_avg(name, run, dataset):
    """Return the first-last semantic distance for the specified run."""
    sem = load_semantic_avg(name, run, dataset)
    sd = ((sem - sem[0]) ** 2).mean(axis=1) ** 0.5  # Sum along rows
    return sd  # / len(sd)


def load_avg_sem_distance_data(name, n_runs, dataset):
    """Loads average semantic distance for specified dataset in a list."""
    return np.array([load_sem_distance_avg(name, i, dataset)
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


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Read results and generate plots and statistics")
    parser.add_argument('-d', '--dir', nargs=2, action='append',
            metavar=('name', 'directory'),
            help='name-directory pair to load')
    parser.add_argument('-m', '--median', action='store_true',
            help='Use median instead of mean to average data')
    parser.add_argument('prefix', help='Prefix for output files')
    return parser.parse_args()


def main():
    """Read data and produce statistics."""
    # old_cycler = plt.rcParams['axes.prop_cycle']

    args = parse_arguments()
    better_names, out_dirs = zip(*args.dir)
    prefix = args.prefix
    #print('Using prefix', pod)
    # Where data files are stored, these must be CLEAN directory names in PWD
    # e.g. foo/ -> invalid, ..foo -> invalid, ./foo/ -> invalid, foo -> valid
    #out_dirs = [pod + 'none', pod + 'mod', pod + 'all']
    #else:
    #    print('Using directories', pod)
    #    out_dirs = pod

    use_mean = not args.median

    # Skip titles
    use_title = False

    # Decent names for files (I used lower, but can be changed)
    better_names_files = [n.lower() for n in better_names]

    # Probably nothing to change below this line

    # Load style
    use_grayscale_print_style()

    # Build names mapping
    bn = dict(zip(out_dirs, better_names))
    bnf = dict(zip(out_dirs, better_names_files))

    # Load global statistics
    stats = {}
    for name in out_dirs:
        with open(os.path.join(name, 'stats.json')) as statsfile:
            stats[name] = json.load(statsfile)

    # Load training data from specified directories
    print('Loading train and test data')
    all_data = {}
    for name in out_dirs:
        pfile = f'{name}_all_data.pkl'
        if os.path.exists(pfile):
            print('Found existing file', pfile, 'loading it')
            # Load previously saved data
            with open(pfile, 'rb') as fp:
                all_data[name] = pickle.load(fp)
        else:
            print('Pickled file', pfile, 'not found, creating')
            runs = stats[name]['args']['runs']
            k_folds = stats[name]['args']['k_folds']
            print('Ottenuto dalle stats il numero di runs', runs, 'e il k_folds', k_folds)
            all_data[name] = {}
            all_data[name]['longrun'] = load_runs(name, 'longrun')
            # Ensure we loaded data for the specified number of runs
            assert all_data[name]['longrun']['raw_train'].shape[0] == runs
            if uses_selection(stats, name):
                # some_path/prefix/prefixR/selection/selectionM/selectionM_K/inner_sub_name
                sel_data = []
                for m, mods in enumerate(stats[name]['models2']):
                    # For each model, load all run data
                    mod_data = []
                    for r in range(runs):
                        # Get base path of r-th run
                        run_path = os.path.join(name, f'{name}{r}', 'selection')
                        # lollete_sel/lollete_sel0/selection/selection0/shortrun
                        run_path = os.path.join(name, f'{name}{r}', 'selection', f'selection{m}')
                        # lollete_sel/lollete_sel0/selection/selection3/selection3_1/inner_sub_name
                        # Using '{prefix}_{r}' to produce selectionM/selectionM_K
                        data = load_runs(run_path, 'shortrun', '{prefix}_{r}')
                        mod_data.append(data)
                        # In every run, we perform k-fold CV, and for each fold
                        # we perform a nested run with j-fold CV
                        assert data['raw_train'].shape[0] == k_folds
                    sel_data.append(mod_data)
                all_data[name]['selection'] = sel_data
                print('Selection data', sel_data)
            # Save data to file for convenience
            with open(pfile, 'wb') as fp:
                pickle.dump(all_data[name], fp)

    # Load semantic distances
    print('Loading semantic data')
    sem_evo_trains, sem_evo_tests = {}, {}
    for name in out_dirs:
        pfile = f'{name}_sem_data.pkl'
        if os.path.exists(pfile):
            print('Found existing file', pfile, 'loading it')
            with open(pfile, 'rb') as fp:
                sem_evo_dat_train, sem_evo_dat_test = pickle.load(fp)
        else:
            print('Pickled file', pfile, 'not found, creating')
            sem_evo_dat_train = load_avg_sem_distance_data(
                    name, stats[name]['args']['runs'], 'train')
            sem_evo_dat_test = load_avg_sem_distance_data(
                    name, stats[name]['args']['runs'], 'test')
            with open(pfile, 'wb') as fp:
                pickle.dump((sem_evo_dat_train, sem_evo_dat_test), fp)
        print('Computing averages') # , shp(sem_evo_dat_train))

        if use_mean:
            sem_evo_trains[name] = {'m': np.mean(sem_evo_dat_train, axis=0)}
            sem_evo_tests[name] = {'m': np.mean(sem_evo_dat_test, axis=0)}
        else:
            sem_evo_trains[name] = {'m': np.median(sem_evo_dat_train, axis=0)}
            sem_evo_tests[name] = {'m': np.median(sem_evo_dat_test, axis=0)}
        # Standard deviation
        sem_evo_trains[name]['s'] = np.std(sem_evo_dat_train, axis=0)
        sem_evo_tests[name]['s'] = np.std(sem_evo_dat_test, axis=0)

    def indices(data):
        if use_mean:
            central = np.mean(data, axis=0)
        else:
            central = np.median(data, axis=0)
        std = np.std(data, axis=0)
        return central, std

    # Time series plots
    scatter('Train fitness' * use_title, ('Generation', 'Fitness'), {
        bn[name]: (None, *indices(all_data[name]['longrun']['raw_train']))
        for name in all_data
    })
    save_img(f'{prefix}fitness_vs_gen_train.png')

    scatter('Test fitness' * use_title, ('Generation', 'Fitness'), {
        bn[name]: (None, *indices(all_data[name]['longrun']['raw_test']))
        for name in all_data
    })
    save_img(f'{prefix}fitness_vs_gen_test.png')

    scatter('Runtime' * use_title, ('Generation', 'Time [s]'), {
        bn[name]: (None, *indices(all_data[name]['longrun']['raw_timing']))
        for name in all_data
    })
    save_img(f'{prefix}runtime_vs_gen.png')

    scatter('Train Runtime-vs-Fitness (average time per run)' * use_title,
            ('Time [s]', 'Fitness'),
            {bn[name]: (np.mean(all_data[name]['longrun']['timing'], axis=0),
                        *indices(all_data[name]['longrun']['raw_train']))
             for name in all_data})
    save_img(f'{prefix}fitness_vs_runtime_train.png')

    scatter('Test Runtime-vs-fitness (average time per run)' * use_title,
            ('Time [s]', 'Fitness'),
            {bn[name]: (np.mean(all_data[name]['longrun']['timing'], axis=0),
                        *indices(all_data[name]['longrun']['raw_test']))
             for name in all_data})
    save_img(f'{prefix}fitness_vs_runtime_test.png')

    plt.figure()
    ax = plt.subplot()
    plt.title('Train Runtime-vs-Fitness (wall clock time)' * use_title)
    names = []
    for name in all_data:
        names.append(bn[name])
        seltime = stats[name]['sel_time'] / stats[name]['n_runs']
        lontime = stats[name]['lon_time'] / stats[name]['n_runs']
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
    save_img(f'{prefix}fitness_vs_wct_train.png')

    plt.figure()
    ax = plt.subplot()
    plt.title('Test Runtime-vs-Fitness (wall clock time)' * use_title)
    names = []
    for name in all_data:
        names.append(bn[name])
        seltime = stats[name]['sel_time'] / stats[name]['n_runs']
        lontime = stats[name]['lon_time'] / stats[name]['n_runs']
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
    save_img(f'{prefix}fitness_vs_wtc_test.png')

    for name in all_data:
        scatter(f'{bn[name]} Train-vs-Test Fitness (Runtime)' * use_title,
                ('Time [s]', 'Fitness'),
                {f'{bn[name]} {ds}': (np.mean(all_data[name]['longrun']['timing'], axis=0),
                                      *indices(all_data[name]['longrun'][ds]))
                 for ds in ['raw_train', 'raw_test']})
        save_img(f'{prefix}{bnf[name]}_train_vs_test_fitness_runtime.png')

    # Pie charts with running times
    for name in stats:
        plt.figure()
        runtimes = stats[name]['sel_time'], stats[name]['lon_time']
        labels = [f'selection\n{runtimes[0]/stats[name]["n_runs"]:.1f}s',
                  f'evolution\n{runtimes[1]/stats[name]["n_runs"]:.1f}s']

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
        ax.set_title(title * use_title)
        save_img(f'{prefix}{bnf[name]}_average_running_time.png')

    # Selection frequencies
    for name in stats:
        plt.figure()
        # Best model frequency
        bm = stats[name]['best_models']
        # Models combinations
        mc = stats[name]['models2']
        # Number of combinations
        nm = len(mc)

        # Remove possible suffix _sem from model names
        mc = [[s.rsplit('_sem', 1)[0] for s in m] for m in mc]

        # Prepare data for plotting
        x = list(range(nm))
        h = [bm.get(str(i), 0) for i in range(nm)]

        # Prepare bar chart
        fig, ax = plt.subplots()
        ax.bar(x, h)

        # Write models in column
        labels = ['\n'.join(s[3:] for s in m) if m else '(none)' for m in mc]
        ax.set_xticks(range(nm))
        ax.set_xticklabels(labels)

        # Set title
        fig.suptitle(
            f'{name} best models combination absolute frequency' * use_title)
        # print('Best models freq', name, bm)
        save_img(f'{prefix}{bnf[name]}_selection_frequency.png')

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
        ax.set_title(title * use_title)
        plt.xlabel('Generations')
        plt.ylabel('Semantic distance')  # (L²)')
        plt.legend(['Train', 'Test'])
        save_img(f'{prefix}{bnf[name]}_sem_dist_evolution.png')


if __name__ == '__main__':
    main()
