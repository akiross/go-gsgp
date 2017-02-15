# go-gsgp
Geometric Semantic Genetic Programming ported to Go

The original code (by Mauro Castelli) is available at http://gsgp.sf.net

This version is compatible with the version 1.0 and features:

 - shorter, safer Go code;
 - better reading of configuration files;
 - better handing of command line arguments.

# Usage

    go get github.com/akiross/go-gsgp
	$GOPATH/bin/go-gsgp -train_file train_dataset -test_file test_dataset

To change parameters, edit the `configuration.ini` file.

The train and test files have the following format:

    n_VARS
	m_EXAMPLES
	V11 V12 V3 ... V1n T
	V21 V22 V3 ... V2n T
	...
	Vm1 Vm2 V3 ... Vmn T

Where, the first line contains the number `n` of variables, the second line
contains the number `m` of cases in the dataset. Then, follow `m` lines of
`n+1` space-separated columns, where the last column is the target value.

# Custom initialization

It is possible to initialize the population by running external commands.
This is done by providing the `-models` flag and then specifying the commands
to run as final arguments of the command. The commands will be executed and
their outputs captured to build individuals.

For example, given an `enode_individual` command:

    $ encode_individual x0+4*x1
	(+ x0 (* x1 4))

the following command

	$ go-gsgp -models 'encode_individual x0+4*x1' 'encode_individual 3*x1'

will run go-gsgp using the configuration.ini (if present) file and the first
two individuals will be initialized using the outputs from the commands

	encode_individual x0+4*x1
	encode_individual 3*x1

For correct initialization, the custom commands shall return a s-expression
on a single line describing the tree to be used as individual.
Examples of valid s-expressions:

	3.14                      // a real number
	x4                        // the 5th input variable (x0 is the first)
	(+ 3.14 x4)               // representing the sum x4+3.14
	(- (+ 3.14 0.0015) x4)    // representing 3.1415 - x4

Valid functionals are currently +, -, *, / (protected) and sqrt (1-ary).
