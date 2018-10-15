# go-gsgp
Geometric Semantic Genetic Programming ported to Go

The original code (by Mauro Castelli) is available at http://gsgp.sf.net

This version is compatible with the version 1.0 and features:

 - shorter, safer Go code;
 - better reading of configuration files;
 - better handing of command line arguments.

# Building

To generate protobuf code, get protobuf on your host, then

    go get github.com/golang/protobuf/protoc-gen-go

To include git version number, compile using:

    go build -ldflags "-X main.gitCommit=`git describe --long --dirty --tags --always`"

The commit will show up when running with `-version` flag.

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

# Initialization via semantic feeding

Instead of randomly generating all the initial individuals, and computing
their semantic, it is possible to feed some pre-computed semantics via files.

To do so, provide a list of files as positional arguments to the program:

    $ go-gsgp [options] semantic1 semantic2 ... semanticN

The files consist in a semantic vector with one value per line. The file shall
contain the semantic values for the training set followed by the test values.

