# Autorun

Autorun contains code to easily perform tests and result analysis.

Given a dataset, `runner.py` will use to to perform a given number of K-fold
cross-validated runs, optional with a model selection phaze obtained by a
nested J-fold cross-validated runs.

The `reporter.py` will take output directories and produce plots for them. I
designed it basically to compare 3 type of runs:

 - with model selection,
 - with no models selected (classic GSGP),
 - with all models selected.

CLI for `reporter.py` is not very useful as that script is intended primarily
to produce plots for journals.

## Installing

Install the package and call it as a python module:

	pip3 install [--user] --upgrade-strategy only-if-needed -U .

In this way the module is accessible from anywhere using

    python3 -m autorun.runner
    python3 -m autorun.reporter

## Running

An example execution is the following:

    python3 -m autorun.runner --noask -C config.txt -r 30 -B ./bin/ \
                              -k 5 -j 4 \ 
                              -s 50 -l 2000 \
                              -M ./models/ \
                              dataset.txt \
                              results_sel
    python3 -m autorun.runner --noask -C config.txt -r 30 -B ./bin/ \
                              -k 5 -j 4 \ 
                              -l 2000 \
                              -M ./models/ --all \
                              dataset.txt \
                              results_all
    python3 -m autorun.runner --noask -C config.txt -r 30 -B ./bin/ \
                              -k 5 -j 4 \
                              -l 2000 \
                              -M ./models/ --none \
                              datasets.txt \
                              results_none

These commands perform 30 runs of go-gsgp using `config.txt` as configuration
file and `dataset.txt` as dataset. Each run uses 5-folds cross-validation, and
in the first case, a selection phase is performed using the models in `models`
directory. Each selection run uses 4-fold nested cross validation, this means
that each of the 5 train partitions of `dataset.txt` is used as dataset for the
nested 4-folds, for a sub-total of 5*4=20 execution. This is repeated for each
of the 30 runs, for a total of 20*30=600 executions.
Each selection execution is performed for `-s 50` generations, while - once the
selection process completed - a long run of `-l 2000` generations is performed.

Note that in the second and third command, no selection phase is performed as
the `--all` and `--none` options are passed (and the `-s` flag was omitted).

## Analysis

When done, the `reporter.py` script can be run like the following:

    python3 -m autorun.reporter -d None results_none \
                                -d Select results_sel \
	                            -d All results_all \
						        some_prefix_

The module can be installed with pip for live exploration of data, as well as
batch processing by using

    python3 -m autorun.reporter -d None results_none (etc)

Please note that order matters: the first result will be use as null hypothesis
when performing U-tests (other data are tested against the first).

This will read the data in the specified directory and produce the results
using `some_prefix_` prepended to files. This will also pickle the data onto
separated files for later analysis (in case you change something in the plots,
you don't want to re-read all the data from scratch).

The `-w` flag can be used for extra-wide plots, `-c` for colorful ggplots and
`-t` for having a title over the plots.
