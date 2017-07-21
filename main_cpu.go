// gsgp-go implements Geometric Semantic Genetic Programming
//
// Original C++ code from Mauro Castelli http://gsgp.sf.net
//
// Go port and subsequent changes from Alessandro Re
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

// +build gc,!gccgo

package main

import (
	"bufio"
	"flag"
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"math"
	"math/rand"
	"os"
	"runtime"
	"runtime/pprof"
	"strconv"
	"strings"
	"sync"
	"time"
)

type cInt int
type cFloat64 float64

func exp64(v cFloat64) cFloat64 {
	r := math.Exp(float64(v))
	// Check for overflows
	if (r == 0 && v > 0) || math.IsInf(r, 1) {
		return cFloat64(math.MaxFloat64)
	}
	// Check for underflow
	if r == 1 && v != 0 {
		return 0
	}
	return cFloat64(r)
}

// Instance represent a single training/test instance in memory
type Instance struct {
	vars    []cFloat64 // Values of the input (independent) variables
	y_value cFloat64   // Target value
}

// Config stores the parameters of a configuration.ini file
type Config struct {
	population_size        *int     // Number of candidate solutions
	max_number_generations *int     // Number of generations of the GP algorithm
	init_type              *int     // Initialization method: 0 -> grow, 1 -> full, 2 -> ramped h&h
	p_crossover            *float64 // Crossover rate
	p_mutation             *float64 // Mutation rate
	max_depth_creation     *int     // Maximum depth of a newly created individual
	tournament_size        *int     // Size of the tournament selection
	zero_depth             *bool    // Are single-node individuals acceptable in initial population?
	mutation_step          *float64 // Step size for the geometric semantic mutation
	num_random_constants   *int     // Number of constants to be inserted in terminal set
	min_random_constant    *float64 // Minimum possible value for a random constant
	max_random_constant    *float64 // Maximum possible value for a random constant
	minimization_problem   *bool    // True if we are minimizing, false if maximizing
	path_in, path_test     *string  // Paths for input data files
	rng_seed               *int64   // Seed for random numbers
	of_train, of_test      *string  // Paths for output fitness files
	of_timing              *string  // Path for file with timings
	error_measure          *string  // Error measure to use for fitness
	n_workers              *int     // Number of workers to use (goroutines)
}

// Symbol represents a symbol of the set T (terminal symbols) or F (functional symbols).
type Symbol struct {
	isFunc bool     // Functional or terminal
	arity  cInt     // Number of arguments accepted by a symbol. Terminals have arity -1 when constants and the index of the variable otherwise
	id     cInt     // Unique identifier for this symbol
	name   string   // Symbolic name
	value  cFloat64 // Current value of terminal symbol
}

// Node is used to represent a node of the tree.
type Node struct {
	root     *Symbol // Symbol for the node
	parent   *Node   // Parent of the node, if any (can be nil)
	children []*Node // Child nodes, can be empty
}

// Population is used to represent a GP population.
type Population struct {
	individuals []*Node // Individuals' root node
	num_ind     cInt    // Number of individuals in the population
}

// The Semantic of one individual is a vector as long as the dataset where each
// component is the value obtaining by applying the individual to the datum.
type Semantic []cFloat64

var (
	gitCommit string // This will be filled at link-time
	// Create flag/configuration variables with default values (in case config file is missing)
	config_file = flag.String("config", "configuration.ini", "Path of the configuration file")
	// Config is initially filled with default values, before init() is executed
	config = Config{
		population_size:        flag.Int("population_size", 200, "Number of candidate solutions"),
		max_number_generations: flag.Int("max_number_generations", 300, "Number of generations of the GP algorithm"),
		init_type:              flag.Int("init_type", 2, "Initialization method: 0 -> grow, 1 -> full, 2 -> ramped h&h"),
		p_crossover:            flag.Float64("p_crossover", 0.6, "Crossover rate"),
		p_mutation:             flag.Float64("p_mutation", 0.3, "Mutation rate"),
		max_depth_creation:     flag.Int("max_depth_creation", 6, "Maximum depth of a newly created individual"),
		tournament_size:        flag.Int("tournament_size", 4, "Size of the tournament selection"),
		zero_depth:             flag.Bool("zero_depth", false, "Are single-node individuals acceptable in initial population?"),
		mutation_step:          flag.Float64("mutation_step", 1, "Step size for the geometric semantic mutation"),
		num_random_constants:   flag.Int("num_random_constants", 0, "Number of constants to be inserted in terminal set"),
		min_random_constant:    flag.Float64("min_random_constant", -100, "Minimum possible value for a random constant"),
		max_random_constant:    flag.Float64("max_random_constant", 100, "Maximum possible value for a random constant"),
		minimization_problem:   flag.Bool("minimization_problem", true, "True if we are minimizing, false if maximizing"),
		path_in:                flag.String("train_file", "", "Path for the train file"),
		path_test:              flag.String("test_file", "", "Path for the test file"),
		rng_seed:               flag.Int64("seed", time.Now().UnixNano(), "Specify a seed for the RNG (uses time by default)"),
		of_train:               flag.String("out_file_train_fitness", "fitnesstrain.txt", "Path for the output file with train fitness data"),
		of_test:                flag.String("out_file_test_fitness", "fitnesstest.txt", "Path for the output file with test fitness data"),
		of_timing:              flag.String("out_file_exec_timing", "execution_time.txt", "Path for the output file containing timings"),
		error_measure:          flag.String("error_measure", "MSE", "Error measures to use for fitness (MSE, MAE or MRE)"),
		n_workers:              flag.Int("workers", runtime.NumCPU(), "Number of workers (goroutines) to use"),
	}
	cpuprofile  = flag.String("cpuprofile", "", "Write CPU profile to file")
	memprofile  = flag.String("memprofile", "", "Write memory profile to file")
	showVersion = flag.Bool("version", false, "Show version")

	NUM_FUNCTIONAL_SYMBOLS cInt // Number of functional symbols
	NUM_VARIABLE_SYMBOLS   cInt // Number of terminal symbols for variables
	NUM_CONSTANT_SYMBOLS   cInt // Number of terminal symbols for constants

	// Terminal and functional symbols
	// This slice is filled only by create_T_F() and add_symbol() (which is used by read_sem() on initialization)
	// len(symbols) == NUM_FUNCTIONAL_SYMBOLS+NUM_VARIABLE_SYMBOLS+NUM_CONSTANT_SYMBOLS
	// In this slice, first you find NUM_FUNCTIONAL_SYMBOLS symbols, then
	// NUM_VARIABLE_SYMBOLS symbols, finally NUM_CONSTANT_SYMBOLS symbols
	symbols = make([]*Symbol, 0)

	set       []Instance // Store training and test instances
	nrow      int        // Number of rows (instances) in training dataset
	nvar      int        // Number of variables (columns excluding target) in training dataset
	nrow_test int        // Number of rows (instances) in test dataset
	nvar_test int        // Number of input variables (columns excluding target) in test dataset FIXME unused

	fit          []cFloat64 // Training fitness values at generation g
	fit_test     []cFloat64 // Test fitness values at generation g
	fit_new      []cFloat64 // Training fitness values at current generation g+1
	fit_test_new []cFloat64 // Test fitness values at current generation g+1

	sem_train_cases     []Semantic // Semantics of the population, computed on training set, at generation g
	sem_train_cases_new []Semantic // Semantics of the population, computed on training set, at current generation g+1
	sem_test_cases      []Semantic // Semantics of the population, computed on test set, at generation g
	sem_test_cases_new  []Semantic // Semantics of the population, computed on test set, at current generation g+1

	index_best cInt // Index of the best individual (where? sem_*?)

	semchan chan Semantic // Channel to move semantics fromm device to host
	cmdchan chan int      // Channel where commands are sent

	dist_func func(cFloat64, cFloat64) cFloat64 // Distance function to use for fitness
)

// Define a sink type that works like /dev/null, but can be closed
type sink int

func (s sink) Close() error                { return nil }
func (s sink) Write(p []byte) (int, error) { return len(p), nil }

func init() {
	// Look for the configuration file flag
	for i := range os.Args[1:] {
		s := os.Args[i]
		if len(s) < 7 {
			continue // Skip flags that are not long enough
		}
		// Flag could be "-config file", "-config=file", "--config file" or "--config=file"
		if s[:7] == "-config" {
			if len(s) == 7 {
				// Configuration file is next argument
				*config_file = os.Args[i+1]
				break
			} else if s[7] == '=' {
				*config_file = s[7:]
				break
			} else {
				fmt.Errorf("Cannot parse config flag, use -config or --config followed by file path")
				os.Exit(1)
			}
		} else if s[:8] == "--config" {
			if len(s) == 8 {
				*config_file = os.Args[i+1]
				break
			} else if s[8] == '=' {
				*config_file = s[8:]
				break
			} else {
				fmt.Errorf("Cannot parse config flag, use -config or --config followed by file path")
				os.Exit(1)
			}
		}
	}
	// Reading the config here allows to use a different config file path, as init is executed after variables initialization
	// Read variables: if present in the config, they will override the defaults
	if _, err := os.Stat(*config_file); os.IsNotExist(err) {
		log.Println("Configuration file", *config_file, "does not exists, using defaults")
	} else {
		read_config_file(*config_file)
	}
}

func square_diff(a, b cFloat64) cFloat64  { return (a - b) * (a - b) }
func abs_diff(a, b cFloat64) cFloat64     { return cFloat64(math.Abs(float64(a - b))) }
func rel_abs_diff(a, b cFloat64) cFloat64 { return cFloat64(math.Abs(float64(a-b))) / a }

func atoi(s string) int {
	v, err := strconv.Atoi(s)
	if err != nil {
		panic(err)
	}
	return v
}

func atof(s string) float64 {
	v, err := strconv.ParseFloat(s, 64)
	if err != nil {
		panic(err)
	}
	return v
}

// read_config_file returns a filled Config struct with values read in the specified file
func read_config_file(path string) {
	file, err := os.Open(path)
	if err != nil {
		panic(err)
	}
	defer file.Close()
	input := bufio.NewScanner(file)
	for input.Scan() {
		fields := strings.Split(input.Text(), "=")
		fields[0], fields[1] = strings.TrimSpace(fields[0]), strings.TrimSpace(fields[1])
		// Skip comments
		if strings.HasPrefix(fields[0], "#") {
			continue
		}
		// Parse options
		switch strings.ToLower(fields[0]) {
		case "population_size":
			*config.population_size = atoi(fields[1])
		case "max_number_generations":
			*config.max_number_generations = atoi(fields[1])
		case "init_type":
			*config.init_type = atoi(fields[1])
		case "p_crossover":
			*config.p_crossover = atof(fields[1])
		case "p_mutation":
			*config.p_mutation = atof(fields[1])
		case "max_depth_creation":
			*config.max_depth_creation = atoi(fields[1])
		case "tournament_size":
			*config.tournament_size = atoi(fields[1])
		case "zero_depth":
			*config.zero_depth = atoi(fields[1]) == 1
		case "mutation_step":
			*config.mutation_step = atof(fields[1])
		case "num_random_constants":
			*config.num_random_constants = atoi(fields[1])
		case "min_random_constant":
			*config.min_random_constant = atof(fields[1])
		case "max_random_constant":
			*config.max_random_constant = atof(fields[1])
		case "minimization_problem":
			*config.minimization_problem = atoi(fields[1]) == 1
		case "train_file":
			*config.path_in = fields[1]
		case "test_file":
			*config.path_test = fields[1]
		case "out_file_train_fitness":
			*config.of_train = fields[1]
		case "out_file_test_fitness":
			*config.of_test = fields[1]
		case "out_file_exec_timing":
			*config.of_timing = fields[1]
		case "error_measure":
			*config.error_measure = fields[1]
		default:
			println("Read unknown parameter: ", fields[0])
		}
		if *config.p_crossover < 0 || *config.p_mutation < 0 || *config.p_crossover+*config.p_mutation > 1 {
			panic("Crossover rate and mutation rate must be greater or equal to 0 and their sum must be smaller or equal to 1.")
		}
	}
}

// Reads the data from the training file and from the test file.
func read_input_data(train_file, test_file string) {
	// Open files for reading
	in_f, err := os.Open(train_file)
	if err != nil {
		panic(err)
	}
	defer in_f.Close()
	in_test_f, err := os.Open(test_file)
	if err != nil {
		panic(err)
	}
	defer in_test_f.Close()
	// Build scanners to read one space-separated word at time
	in := bufio.NewScanner(in_f)
	in.Split(bufio.ScanWords)
	in_test := bufio.NewScanner(in_test_f)
	in_test.Split(bufio.ScanWords)
	// Read first two tokens of each file
	nvar = atoi(next_token(in))           // Number of variables
	nvar_test = atoi(next_token(in_test)) // FIXME is this necessary? it is not used
	if nvar != nvar_test {
		panic("Train and Test datasets must have the same number of variables")
	}
	nrow = atoi(next_token(in)) // Number of rows
	nrow_test = atoi(next_token(in_test))
	set = make([]Instance, nrow+nrow_test)
	for i := 0; i < nrow; i++ {
		set[i].vars = make([]cFloat64, nvar)
		for j := 0; j < nvar; j++ {
			set[i].vars[j] = cFloat64(atof(next_token(in)))
		}
		set[i].y_value = cFloat64(atof(next_token(in)))
	}
	for i := nrow; i < nrow+nrow_test; i++ {
		set[i].vars = make([]cFloat64, nvar)
		for j := 0; j < nvar; j++ {
			set[i].vars[j] = cFloat64(atof(next_token(in_test)))
		}
		set[i].y_value = cFloat64(atof(next_token(in_test)))
	}
}

// create_T_F creates the terminal and functional sets
// Names in created symbols shall not include the characters '(' or ')'
// because they are used when reading and writing a tree to string
func create_T_F() {
	NUM_VARIABLE_SYMBOLS = cInt(nvar)
	// Create functional symbols
	fs := []struct {
		name  string
		arity cInt
	}{
		// When changing these, remember to change the kernel accordingly
		{"+", 2},
		{"-", 2},
		{"*", 2},
		{"/", 2},
		//{"sqrt", 1},
	}
	NUM_FUNCTIONAL_SYMBOLS = cInt(len(fs))
	for i, s := range fs {
		symbols = append(symbols, &Symbol{true, s.arity, cInt(i), s.name, 0})
	}
	// Create terminal symbols for variables
	for i := NUM_FUNCTIONAL_SYMBOLS; i < NUM_VARIABLE_SYMBOLS+NUM_FUNCTIONAL_SYMBOLS; i++ {
		str := fmt.Sprintf("x%d", i-NUM_FUNCTIONAL_SYMBOLS)
		symbols = append(symbols, &Symbol{false, i - NUM_FUNCTIONAL_SYMBOLS, i, str, 0})
	}
	// Create terminal symbols for constants
	for i := NUM_VARIABLE_SYMBOLS + NUM_FUNCTIONAL_SYMBOLS; i < NUM_VARIABLE_SYMBOLS+NUM_FUNCTIONAL_SYMBOLS+NUM_CONSTANT_SYMBOLS; i++ {
		a := cFloat64(*config.min_random_constant + rand.Float64()*(*config.max_random_constant-*config.min_random_constant))
		str := fmt.Sprintf("%f", a)
		symbols = append(symbols, &Symbol{false, -1, i, str, a})
	}
}

// choose_function randomly selects a functional symbol and returns its ID
func choose_function() cInt {
	return cInt(rand.Intn(int(NUM_FUNCTIONAL_SYMBOLS)))
}

// choose_terminal randomly selects a terminal symbol.
// With probability 0.7 a variable is selected, while random constants have a probability of 0.3 to be selected.
// To change these probabilities just change their values in the function.
// It returns the ID of the chosen terminal symbol
func choose_terminal() cInt {
	if NUM_CONSTANT_SYMBOLS == 0 {
		return NUM_FUNCTIONAL_SYMBOLS + cInt(rand.Intn(int(NUM_VARIABLE_SYMBOLS)))
	}
	if rand.Float64() < 0.7 {
		return NUM_FUNCTIONAL_SYMBOLS + cInt(rand.Intn(int(NUM_VARIABLE_SYMBOLS)))
	}
	return NUM_FUNCTIONAL_SYMBOLS + NUM_VARIABLE_SYMBOLS + cInt(rand.Intn(int(NUM_CONSTANT_SYMBOLS)))
}

// create_grow_pop creates a population using the grow method
func create_grow_pop(p *Population) {
	for p.num_ind < cInt(*config.population_size) {
		node := create_grow_tree(0, nil, cInt(*config.max_depth_creation))
		p.individuals[p.num_ind] = node
		p.num_ind++
	}
}

// Creates a population of full trees (each tree has a depth equal to the maximum length possible)
func create_full_pop(p *Population) {
	for p.num_ind < cInt(*config.population_size) {
		node := create_full_tree(0, nil, cInt(*config.max_depth_creation))
		p.individuals[p.num_ind] = node
		p.num_ind++
	}
}

// Creates a population with the ramped half and half algorithm.
func create_ramped_pop(p *Population) {
	var (
		population_size    = cInt(*config.population_size)
		max_depth_creation = cInt(*config.max_depth_creation)
		sub_pop            cInt
		r                  cInt
		min_depth          cInt
	)

	if !*config.zero_depth {
		sub_pop = (population_size - p.num_ind) / max_depth_creation
		r = (population_size - p.num_ind) % max_depth_creation
		min_depth = 1
	} else {
		sub_pop = (population_size - p.num_ind) / (max_depth_creation + 1)
		r = (population_size - p.num_ind) % (max_depth_creation + 1)
		min_depth = 0
	}
	for j := max_depth_creation; j >= min_depth; j-- {
		if j < max_depth_creation {
			for k := cInt(0); k < cInt(math.Ceil(float64(sub_pop)*0.5)); k++ {
				node := create_full_tree(0, nil, j)
				p.individuals[p.num_ind] = node
				p.num_ind++
			}
			for k := cInt(0); k < cInt(math.Floor(float64(sub_pop)*0.5)); k++ {
				node := create_grow_tree(0, nil, j)
				p.individuals[p.num_ind] = node
				p.num_ind++
			}
		} else {
			for k := cInt(0); k < cInt(math.Ceil(float64(sub_pop+r)*0.5)); k++ {
				node := create_full_tree(0, nil, j)
				p.individuals[p.num_ind] = node
				p.num_ind++
			}
			for k := cInt(0); k < cInt(math.Floor(float64(sub_pop+r)*0.5)); k++ {
				node := create_grow_tree(0, nil, j)
				p.individuals[p.num_ind] = node
				p.num_ind++
			}
		}
	}
}

// Create a new Population. It is possible to pass "seeds", which are
// s-expressions to be parsed as starting individuals. If too many seeds
// are provided (greater than config.population_size), it will panic.
func NewPopulation(seeds ...string) *Population {
	if len(seeds) > *config.population_size {
		panic("Too many seeds")
	}
	p := &Population{
		individuals: make([]*Node, *config.population_size),
		num_ind:     cInt(len(seeds)),
	}
	for i := range seeds {
		p.individuals[i] = read_sem(seeds[i])
	}
	return p
}

// Fills the population using the method specified by the parameter
func initialize_population(p *Population, method cInt) {
	switch method {
	case 0:
		create_grow_pop(p)
	case 1:
		create_full_pop(p)
	default:
		create_ramped_pop(p)
	}
}

// Creates a random tree with depth in the range [0;max_depth] and returning its root Node
func create_grow_tree(depth cInt, parent *Node, max_depth cInt) *Node {
	if depth == 0 && !*config.zero_depth {
		sym := symbols[choose_function()]
		el := &Node{
			root:     sym,
			parent:   nil,
			children: make([]*Node, sym.arity),
		}
		for i := cInt(0); i < sym.arity; i++ {
			el.children[i] = create_grow_tree(depth+1, el, max_depth)
		}
		return el
	}
	if depth == max_depth {
		return &Node{
			root:     symbols[choose_terminal()],
			parent:   parent,
			children: nil,
		}
	}
	if rand.Intn(2) == 0 {
		sym := symbols[choose_function()]
		el := &Node{
			root:     sym,
			parent:   parent,
			children: make([]*Node, sym.arity),
		}
		for i := cInt(0); i < sym.arity; i++ {
			el.children[i] = create_grow_tree(depth+1, el, max_depth)
		}
		return el
	} else {
		term := choose_terminal()
		return &Node{
			root:     symbols[term],
			parent:   parent,
			children: nil,
		}
	}
}

func create_grow_tree_arrays(depth, max_depth cInt, base_index cInt) []cInt {
	if depth == 0 && !*config.zero_depth {
		// No zero-depth inviduals allowed: start with a functional
		op := choose_function()                   // Get ID of the selected functional
		tree := make([]cInt, symbols[op].arity+1) // Create space for ID and children pointers
		tree[0] = cInt(op)                        // Save functional ID in first location
		// Create children trees
		for c := cInt(1); c <= symbols[op].arity; c++ {
			tree[c] = cInt(len(tree)) + base_index // Save child position in next location
			child := create_grow_tree_arrays(depth+1, max_depth, tree[c])
			tree = append(tree, child...)
		}
		return tree
	}
	if depth == max_depth {
		return []cInt{cInt(choose_terminal())}
	}
	if rand.Intn(2) == 0 {
		return []cInt{cInt(choose_terminal())}
	} else {
		op := choose_function()
		tree := make([]cInt, symbols[op].arity+1)
		tree[0] = cInt(op)
		for c := cInt(1); c <= symbols[op].arity; c++ {
			tree[c] = cInt(len(tree)) + base_index
			child := create_grow_tree_arrays(depth+1, max_depth, tree[c])
			tree = append(tree, child...)
		}
		return tree
	}
}

// Creates a tree with depth equal to the ones specified by the parameter max_depth
func create_full_tree(depth cInt, parent *Node, max_depth cInt) *Node {
	if depth == 0 && depth < max_depth {
		sym := symbols[choose_function()]
		el := &Node{
			root:     sym,
			parent:   nil,
			children: make([]*Node, sym.arity),
		}
		for i := cInt(0); i < sym.arity; i++ {
			el.children[i] = create_full_tree(depth+1, el, max_depth)
		}
		return el
	}
	if depth == max_depth {
		return &Node{
			root:     symbols[choose_terminal()],
			parent:   parent,
			children: nil,
		}
	}
	sym := symbols[choose_function()]
	el := &Node{
		root:     sym,
		parent:   parent,
		children: make([]*Node, sym.arity),
	}
	for i := cInt(0); i < sym.arity; i++ {
		el.children[i] = create_full_tree(depth+1, el, max_depth)
	}
	return el
}

// Convert a Node-based tree to a array-based tree
func tree_to_array(root *Node) []cInt {
	var rec_build func(n *Node, base cInt) []cInt
	rec_build = func(n *Node, base cInt) []cInt {
		if n.root.isFunc {
			t := make([]cInt, n.root.arity+1)
			t[0] = cInt(n.root.id)
			for c := range n.children {
				t[c+1] = cInt(len(t)) + base
				ct := rec_build(n.children[c], t[c+1])
				t = append(t, ct...)
			}
			return t
		} else {
			return []cInt{cInt(n.root.id)}
		}
	}

	return rec_build(root, 0)
}

// Convert string with numeric constant into a symbol and add it to list
func add_symbol(name string) *Symbol {
	val, err := strconv.ParseFloat(name, 64)
	if err != nil {
		return nil // Not a float, must be a wrong variable or functional
	}
	// Conversion was successful, must be a constant
	sym := &Symbol{false, -1, NUM_CONSTANT_SYMBOLS, name, cFloat64(val)}
	symbols = append(symbols, sym)
	// Increase symbol count
	NUM_CONSTANT_SYMBOLS++
	return sym
}

// Reads the file and returns a node that represents a semantic
func read_sem(path string) *Node {
	file, err := os.Open(path)
	if err != nil {
		panic(err)
	}
	defer file.Close()
	// Output node, has no symbol because it's a dummy node
	node := &Node{nil, nil, nil}
	// There should be one line for each train and test case
	input := bufio.NewScanner(file)
	var i int
	for i = 0; input.Scan() && i < nrow+nrow_test; i++ {
		s := input.Text()
		sym := add_symbol(s)
		node.children = append(node.children, &Node{sym, nil, nil})
	}
	if i != nrow+nrow_test {
		panic("Not enough values when reading semantic file")
	}
	return node
}

// Implements a protected division. If the denominator is equal to 0 the function returns 1 as a result of the division;
func protected_division(num, den cFloat64) cFloat64 {
	if den == 0 {
		return 1
	}
	return num / den
}

// This function retrieves the value of a terminal symbol given
// the i-th instance as input.
func terminal_value(i cInt, sym *Symbol) cFloat64 {
	if sym.id >= NUM_FUNCTIONAL_SYMBOLS && sym.id < NUM_FUNCTIONAL_SYMBOLS+NUM_VARIABLE_SYMBOLS {
		// Variables take their value from the input data
		return set[i].vars[sym.id-NUM_FUNCTIONAL_SYMBOLS]
	} else {
		// The value of a constant can be used directly
		return sym.value
	}
}

func eval_arrays(tree []cInt, start cInt, i cInt) cFloat64 {
	switch {
	case symbols[tree[start]].name == "+":
		return eval_arrays(tree, tree[start+1], i) + eval_arrays(tree, tree[start+2], i)
	case symbols[tree[start]].name == "-":
		return eval_arrays(tree, tree[start+1], i) - eval_arrays(tree, tree[start+2], i)
	case symbols[tree[start]].name == "*":
		return eval_arrays(tree, tree[start+1], i) * eval_arrays(tree, tree[start+2], i)
	case symbols[tree[start]].name == "/":
		return protected_division(eval_arrays(tree, tree[start+1], i), eval_arrays(tree, tree[start+2], i))
	case symbols[tree[start]].name == "sqrt":
		v := eval_arrays(tree, tree[start+1], i)
		if v < 0 {
			return cFloat64(math.Sqrt(float64(-v)))
		} else {
			return cFloat64(math.Sqrt(float64(v)))
		}
	default:
		return terminal_value(i, symbols[tree[start]]) // Root points to a terminal
	}
}

// Calculates the fitness of all the individuals and determines the best individual in the population
// Evaluate is called once, after individuals have been initialized for the first time.
// This function fills fit using semantic_evaluate
func evaluate(p *Population) {
	for i := 0; i < *config.population_size; i++ {
		arr := tree_to_array(p.individuals[i])

		var a, b cFloat64
		sem_train_cases[i] = semantic_evaluate_array(arr, cInt(nrow), 0)
		fit[i], a, b = fitness_of_semantic_train(sem_train_cases[i], cInt(nrow), 0)

		sem_test_cases[i] = semantic_evaluate_array(arr, cInt(nrow_test), cInt(nrow))
		fit_test[i] = fitness_of_semantic_test(sem_test_cases[i], cInt(nrow_test), cInt(nrow), a, b)
	}
}

func semantic_evaluate_array(tree []cInt, sem_size, sem_offs cInt) Semantic {
	val := make(Semantic, sem_size) // Array with semantic to be computed

	if *config.n_workers > 1 {
		n_workers := cInt(*config.n_workers)
		block := (sem_size + n_workers - 1) / n_workers

		var wg sync.WaitGroup

		wg.Add(int(n_workers))
		for w := cInt(0); w < n_workers; w++ {
			go func(start, end cInt) {
				// Check limit
				if end > sem_size {
					end = sem_size
				}
				// Perform evaluation loop
				for i := sem_offs + start; i < sem_offs+end; i++ {
					val[i-sem_offs] = eval_arrays(tree, 0, i)
				}
				wg.Done()
			}(block*w, block*(w+1))
		}
		wg.Wait()
	} else {
		for i := sem_offs; i < sem_size+sem_offs; i++ {
			val[i-sem_offs] = eval_arrays(tree, 0, i)
		}
	}
	return val
}

// Implements a tournament selection procedure
func tournament_selection() cInt {
	// Select first participant
	best_index := rand.Intn(*config.population_size)
	for i := 1; i < *config.tournament_size; i++ {
		next := rand.Intn(*config.population_size)
		if better(fit[next], fit[best_index]) {
			best_index = next
		}
	}
	return cInt(best_index)
}

// Copies an individual of the population at generation g-1 to the current population (generation g)
// Any individual (any position) can be selected to be copied in position i
func reproduction(i cInt) {
	old_i := i
	// Elitism: if i is the best individual, reproduce it
	if i != index_best {
		// If it's not the best, select one at random to reproduce
		i = tournament_selection()
	}

	// Copy fitness and semantics of the selected individual
	copy(sem_train_cases_new[old_i], sem_train_cases[i])
	copy(sem_test_cases_new[old_i], sem_test_cases[i])

	fit_new[old_i] = fit[i]
	fit_test_new[old_i] = fit_test[i]
}

// Performs a geometric semantic crossover
func geometric_semantic_crossover(i cInt) {
	if i != index_best {
		// Create random tree
		rt := create_grow_tree_arrays(0, cInt(*config.max_depth_creation), 0)
		// Replace the individual with the crossover of two parents
		p1 := tournament_selection()
		p2 := tournament_selection()

		var ls_a, ls_b cFloat64
		// Generate a random tree and compute its semantic (train and test)
		sem_rt := semantic_evaluate_array(rt, cInt(nrow), 0)
		sem_rt_test := semantic_evaluate_array(rt, cInt(nrow_test), cInt(nrow))

		// Compute the geometric semantic (train)
		for j := 0; j < nrow; j++ {
			sigmoid := 1 / (1 + exp64(-sem_rt[j]))
			sem_train_cases_new[i][j] = sem_train_cases[p1][j]*sigmoid + sem_train_cases[p2][j]*(1-sigmoid)
		}
		fit_new[i], ls_a, ls_b = fitness_of_semantic_train(sem_train_cases_new[i], cInt(nrow), 0)
		// Compute the geometric semantic (test)
		for j := 0; j < nrow_test; j++ {
			sigmoid := 1 / (1 + exp64(-sem_rt_test[j]))
			sem_test_cases_new[i][j] = sem_test_cases[p1][j]*sigmoid + sem_test_cases[p2][j]*(1-sigmoid)
		}
		fit_test_new[i] = fitness_of_semantic_test(sem_test_cases_new[i], cInt(nrow_test), cInt(nrow), ls_a, ls_b)
	} else {
		// The best individual will not be changed
		copy(sem_train_cases_new[i], sem_train_cases[i])
		copy(sem_test_cases_new[i], sem_test_cases[i])

		fit_new[i] = fit[i]
		fit_test_new[i] = fit_test[i]
	}
}

// Performs a geometric semantic mutation
func geometric_semantic_mutation(i cInt) {
	if i != index_best {
		mut_step := cFloat64(rand.Float64())
		// Create two random trees and copy it to unified memory
		rt1 := create_grow_tree_arrays(0, cInt(*config.max_depth_creation), 0)
		rt2 := create_grow_tree_arrays(0, cInt(*config.max_depth_creation), 0)

		var ls_a, ls_b cFloat64
		// Replace the individual with a mutated version
		sem_rt1 := semantic_evaluate_array(rt1, cInt(nrow), 0)
		sem_rt1_test := semantic_evaluate_array(rt1, cInt(nrow_test), cInt(nrow))

		sem_rt2 := semantic_evaluate_array(rt2, cInt(nrow), 0)
		sem_rt2_test := semantic_evaluate_array(rt2, cInt(nrow_test), cInt(nrow))

		for j := 0; j < nrow; j++ {
			sigmoid1 := 1 / (1 + exp64(-sem_rt1[j]))
			sigmoid2 := 1 / (1 + exp64(-sem_rt2[j]))
			sem_train_cases_new[i][j] += mut_step * (sigmoid1 - sigmoid2)
		}
		fit_new[i], ls_a, ls_b = fitness_of_semantic_train(sem_train_cases_new[i], cInt(nrow), 0)

		for j := 0; j < nrow_test; j++ {
			sigmoid1 := 1 / (1 + exp64(-sem_rt1_test[j]))
			sigmoid2 := 1 / (1 + exp64(-sem_rt2_test[j]))
			sem_test_cases_new[i][j] += mut_step * (sigmoid1 - sigmoid2)
		}
		fit_test_new[i] = fitness_of_semantic_test(sem_test_cases_new[i], cInt(nrow_test), cInt(nrow), ls_a, ls_b)
	}
	// Mutation happens after reproduction: elite are reproduced but are not mutated
}

// Given a semantic, compute the fitness of a subset of that semantic as the
// Mean Squared Difference between the semantic and the dataset.
// From the dataset, only sem_size elements, starting from sem_offs, will be considered in the computation
func fitness_of_semantic_train(sem Semantic, sem_size, sem_offs cInt) (d, a, b cFloat64) {
	if *config.n_workers > 1 {
		n_workers := cInt(*config.n_workers)
		block := (sem_size + n_workers - 1) / n_workers

		var wg sync.WaitGroup

		var (
			sum_out = make([]cFloat64, n_workers)
			sum_tar = make([]cFloat64, n_workers)
			sum_oxo = make([]cFloat64, n_workers)
			sum_oxt = make([]cFloat64, n_workers)
		)

		wg.Add(int(n_workers))
		for w := cInt(0); w < n_workers; w++ {
			go func(id, start, end cInt) {
				// Check limit
				if end > sem_size {
					end = sem_size
				}
				// Perform evaluation
				for i := sem_offs + start; i < sem_offs+end; i++ {
					t := set[i].y_value
					y := sem[i-sem_offs]
					sum_out[id] += y
					sum_tar[id] += t
					sum_oxo[id] += y * y
					sum_oxt[id] += y * t
				}
				wg.Done()
			}(w, block*w, block*(w+1))
		}
		wg.Wait()

		tot_out := sum_out[0]
		tot_tar := sum_tar[0]
		tot_oxo := sum_oxo[0]
		tot_oxt := sum_oxt[0]
		for i := cInt(1); i < n_workers; i++ {
			tot_out += sum_out[i]
			tot_tar += sum_tar[i]
			tot_oxo += sum_oxo[i]
			tot_oxt += sum_oxt[i]
		}

		avg_out := tot_out / cFloat64(sem_size)
		avg_tar := tot_tar / cFloat64(sem_size)

		num := tot_oxt - tot_tar*avg_out - tot_out*avg_tar + cFloat64(sem_size)*avg_out*avg_tar
		den := tot_oxo - 2.0*tot_out*avg_out + cFloat64(sem_size)*avg_out*avg_out

		b = num / den
		a = avg_tar - b*avg_out

		par_d := make([]cFloat64, n_workers)
		wg.Add(int(n_workers))
		for w := cInt(0); w < n_workers; w++ {
			go func(id, start, end cInt) {
				// Check limit
				if end > sem_size {
					end = sem_size
				}
				// Perform evaluation
				for i := sem_offs + start; i < sem_offs+end; i++ {
					par_d[id] += dist_func(set[i].y_value, a+b*sem[i-sem_offs])
				}
				wg.Done()
			}(w, block*w, block*(w+1))
		}
		wg.Wait()
		d = par_d[0]
		for i := cInt(1); i < n_workers; i++ {
			d += par_d[i]
		}
		d = d / cFloat64(sem_size)
	} else {
		var avg_out, avg_tar cFloat64
		for i := sem_offs; i < sem_size+sem_offs; i++ {
			avg_out += sem[i-sem_offs]
			avg_tar += set[i].y_value
		}
		avg_out /= cFloat64(sem_size)
		avg_tar /= cFloat64(sem_size)

		var num, den cFloat64
		for i := sem_offs; i < sem_offs+sem_size; i++ {
			odiff := sem[i-sem_offs] - avg_out
			num += (set[i].y_value - avg_tar) * odiff
			den += odiff * odiff
		}
		b = num / den
		a = avg_tar - b*avg_out

		for i := sem_offs; i < sem_offs+sem_size; i++ {
			d += dist_func(set[i].y_value, a+b*sem[i-sem_offs])
		}
		d = d / cFloat64(sem_size)
	}
	return d, a, b
}

func fitness_of_semantic_test(sem Semantic, sem_size, sem_offs cInt, a, b cFloat64) cFloat64 {
	var d cFloat64

	if *config.n_workers > 1 {
		n_workers := cInt(*config.n_workers)
		block := (sem_size + n_workers - 1) / n_workers

		var wg sync.WaitGroup
		par_d := make([]cFloat64, n_workers)
		wg.Add(int(n_workers))
		for w := cInt(0); w < n_workers; w++ {
			go func(id, start, end cInt) {
				// Check limit
				if end > sem_size {
					end = sem_size
				}
				// Perform evaluation
				for i := sem_offs + start; i < sem_offs+end; i++ {
					par_d[id] += dist_func(set[i].y_value, a+b*sem[i-sem_offs])
				}
				wg.Done()
			}(w, block*w, block*(w+1))
		}
		wg.Wait()
		d = par_d[0]
		for i := cInt(1); i < n_workers; i++ {
			d += par_d[i]
		}
		return d / cFloat64(sem_size)
	} else {
		for i := sem_offs; i < sem_offs+sem_size; i++ {
			d += dist_func(set[i].y_value, a+b*sem[i-sem_offs])
		}
		return d / cFloat64(sem_size)
	}
}

// Finds the best individual in the population
func best_individual() cInt {
	var best_index cInt
	for i := 1; i < len(fit); i++ {
		if better(fit[i], fit[best_index]) {
			best_index = cInt(i)
		}
	}
	return best_index
}

// Updates the tables used to store fitness values and semantics of the individual. It is used at the end of each iteration of the algorithm
func update_tables() {
	fit, fit_new = fit_new, fit
	fit_test, fit_test_new = fit_test_new, fit_test
	sem_train_cases, sem_train_cases_new = sem_train_cases_new, sem_train_cases
	sem_test_cases, sem_test_cases_new = sem_test_cases_new, sem_test_cases
}

// Return the next text token in the provided scanner
func next_token(in *bufio.Scanner) string {
	in.Scan()
	return in.Text()
}

// Compares the fitness of two solutions.
func better(f1, f2 cFloat64) bool {
	if *config.minimization_problem {
		return f1 < f2
	} else {
		return f1 > f2
	}
}

// Calculates the number of nodes of a solution.
func node_count(el *Node) cInt {
	var counter cInt = 1
	if el.children != nil {
		for i := cInt(0); i < el.root.arity; i++ {
			counter += node_count(el.children[i])
		}
	}
	return counter
}

// Create file or panic if an error occurs
// If path is empty, will return a sink
func create_or_panic(path string) io.WriteCloser {
	if path == "" {
		return sink(0)
	}

	f, err := os.Create(path)
	if err != nil {
		panic(err)
	}
	return f
}

func load_file_and_replace(path string, repl map[string]interface{}) string {
	cont, err := ioutil.ReadFile(path)
	if err != nil {
		panic(err)
	}
	target := string(cont)
	for k := range repl {
		target = strings.Replace(target, k, fmt.Sprint(repl[k]), -1)
	}
	return target
}

// Allocate memory for fitness and semantic value for each individual
func init_tables() {
	fit = make([]cFloat64, *config.population_size)
	fit_test = make([]cFloat64, *config.population_size)
	fit_new = make([]cFloat64, *config.population_size)
	fit_test_new = make([]cFloat64, *config.population_size)

	sem_train_cases = make([]Semantic, *config.population_size)
	sem_train_cases_new = make([]Semantic, *config.population_size)
	sem_test_cases = make([]Semantic, *config.population_size)
	sem_test_cases_new = make([]Semantic, *config.population_size)

	for i := 0; i < *config.population_size; i++ {
		sem_train_cases[i] = make(Semantic, nrow)
		sem_train_cases_new[i] = make(Semantic, nrow)
		sem_test_cases[i] = make(Semantic, nrow_test)
		sem_test_cases_new[i] = make(Semantic, nrow_test)
	}
}

func main() {
	// Parse CLI arguments: if they are set, they will override defaults and config file
	flag.Parse()
	// If required, show version and exit
	if *showVersion {
		if gitCommit != "" {
			fmt.Println(gitCommit)
		} else {
			fmt.Println("Compiled without version info")
		}
		return
	}
	// After config is read and flags are parsed
	NUM_CONSTANT_SYMBOLS = cInt(*config.num_random_constants)

	if *config.path_in == "" {
		fmt.Println("Please specify the train dataset using the train_file option")
		return
	}
	if *config.path_test == "" {
		fmt.Println("Please specify the test dataset using the test_file option")
		return
	}

	switch strings.ToUpper(*config.error_measure) {
	case "MAE":
		dist_func = abs_diff
	case "MRE":
		dist_func = rel_abs_diff
	case "MSE":
		dist_func = square_diff
	default:
		panic("Unknown error measure: " + *config.error_measure)
	}

	if *cpuprofile != "" {
		f, err := os.Create(*cpuprofile)
		if err != nil {
			panic(err)
		}
		pprof.StartCPUProfile(f)
		defer pprof.StopCPUProfile()
	}

	// If any extra argument is specified, it is considered as a seed file
	var sem_seed []string
	if flag.NArg() > 0 {
		for _, path := range flag.Args() {
			// TODO verify if file exist
			sem_seed = append(sem_seed, path)
		}
	}

	// Create some files for output data
	executiontime := create_or_panic(*config.of_timing)
	defer executiontime.Close()
	fitness_train := create_or_panic(*config.of_train)
	defer fitness_train.Close()
	fitness_test := create_or_panic(*config.of_test)
	defer fitness_test.Close()

	// Seed RNG
	rand.Seed(*config.rng_seed)
	// Read training and testing datasets (populate nvar, nrow and set)
	read_input_data(*config.path_in, *config.path_test)
	// Create tables with terminals and functionals
	create_T_F()

	// Tracking time
	var start time.Time
	start = time.Now()

	// Create population and feed
	p := NewPopulation(sem_seed...)
	// Prepare tables (memory allocation)
	init_tables()
	initialize_population(p, cInt(*config.init_type))
	// Evaluate each individual in the population, filling fitnesses and finding best individual
	evaluate(p)
	index_best = best_individual()
	fmt.Fprintln(fitness_train, fit[index_best])
	fmt.Fprintln(fitness_test, fit_test[index_best])

	fmt.Fprintln(executiontime, time.Since(start))

	// main GP cycle
	for num_gen := 0; num_gen < *config.max_number_generations; num_gen++ {
		log.Println("Generation", num_gen+1)
		for k := 0; k < *config.population_size; k++ {
			rand_num := rand.Float64()
			switch {
			case rand_num < *config.p_crossover:
				geometric_semantic_crossover(cInt(k))
			case rand_num < *config.p_crossover+*config.p_mutation:
				reproduction(cInt(k))
				geometric_semantic_mutation(cInt(k))
			default:
				reproduction(cInt(k))
			}
		}

		update_tables()

		index_best = best_individual()

		fmt.Fprintln(fitness_train, fit[index_best])
		fmt.Fprintln(fitness_test, fit_test[index_best])

		fmt.Fprintln(executiontime, time.Since(start))
	}
	log.Println("Total elapsed time since start:", time.Since(start))

	if *memprofile != "" {
		f, err := os.Create(*memprofile)
		if err != nil {
			panic(err)
		}
		pprof.WriteHeapProfile(f)
		f.Close()
	}
}
