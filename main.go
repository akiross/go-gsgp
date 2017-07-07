/* gsgp-go implements Geometric Semantic Genetic Programming


	Original C++ code from Mauro Castelli http://gsgp.sf.net

	Go port and subsequent changes from Alessandro Re

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.

    You should have received a copy of the GNU Affero General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/
package main

// #cgo CFLAGS: -I/usr/local/cuda/include
// #cgo LDFLAGS: -L/usr/local/cuda/lib64 -L/usr/lib64/nvidia-bumblebee -lcuda
// #include <stdlib.h>
import "C"

import (
	"bufio"
	"flag"
	"fmt"
	cuda "github.com/akiross/go-cudart"
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
	"time"
	"unsafe"
)

type cInt C.int
type cFloat64 C.double

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
	use_goroutines         *bool    // Flag that enables concurrent computation
	use_cuda               *bool    // Flag that enables CUDA computation
	of_train, of_test      *string  // Paths for output fitness files
	of_timing              *string  // Path for file with timings
	error_measure          *string  // Error measure to use for fitness
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
		use_goroutines:         flag.Bool("use_goroutines", false, "Enable goroutines in evaluation of fitness"),
		use_cuda:               flag.Bool("use_cuda", false, "Enable CUDA in generation of random trees"),
		of_train:               flag.String("out_file_train_fitness", "fitnesstrain.txt", "Path for the output file with train fitness data"),
		of_test:                flag.String("out_file_test_fitness", "fitnesstest.txt", "Path for the output file with test fitness data"),
		of_timing:              flag.String("out_file_exec_timing", "execution_time.txt", "Path for the output file containing timings"),
		error_measure:          flag.String("error_measure", "MSE", "Error measures to use for fitness (MSE, MAE or MRE)"),
	}
	cpuprofile  = flag.String("cpuprofile", "", "Write CPU profile to file")
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

	ctx *cuda.Context

	cu_tpb     int = 256
	cu_bgp_ds  int
	cu_bpg_pop int

	cu_prog         *cuda.Program
	cu_mod          *cuda.Module
	kern_eval_array *cuda.Function
	kern_crossover  *cuda.Function
	kern_fitness    *cuda.Function
	kern_mutation   *cuda.Function

	cu_set                 *cuda.Buffer
	cu_sym_val             *cuda.Buffer
	cu_fit                 *cuda.Buffer
	cu_fit_test            *cuda.Buffer
	cu_fit_new             *cuda.Buffer
	cu_fit_test_new        *cuda.Buffer
	cu_sem_train_cases     []*cuda.Buffer
	cu_sem_train_cases_new []*cuda.Buffer
	cu_sem_test_cases      []*cuda.Buffer
	cu_sem_test_cases_new  []*cuda.Buffer

	// Temporary memory
	cu_tmp_tree_arr1                  *cuda.Buffer
	cu_tmp_tree_arr2                  *cuda.Buffer
	cu_tmp_sem_train, cu_tmp_sem_test *cuda.Buffer
	cu_tmp_sem_tot1                   *cuda.Buffer
	cu_tmp_sem_tot2                   *cuda.Buffer

	cu_tmp_d1, cu_tmp_d2        *cuda.Buffer
	tmp_sem_train, tmp_sem_test []cFloat64
)

const (
	CMD_REPRODUCE = iota
	CMD_MUTATE
	CMD_CROSSOVER
)

// Define a sink type that works like /dev/null
type sink int

func (s sink) Close() error                { return nil }
func (s sink) Write(p []byte) (int, error) { return len(p), nil }

// Simple tool for benchmarking a section of code
type Benchmark struct {
	start time.Time
	durat time.Duration
	count int
}

var benchmarks map[string]*Benchmark

func StartBenchmark(name string) *Benchmark {
	if benchmarks == nil {
		benchmarks = make(map[string]*Benchmark)
	}
	var bmk Benchmark
	benchmarks[name] = &bmk
	defer bmk.Start()
	return &bmk
}

func (b *Benchmark) Start() {
	b.start = time.Now()
}

func (b *Benchmark) Measure() {
	b.durat += time.Since(b.start)
	b.count++
}

func (b *Benchmark) Average() time.Duration {
	if b.count == 0 {
		return -1
	}
	return time.Duration(int64(b.durat) / int64(b.count))
}

func BenchResult(name string) time.Duration {
	val, ok := benchmarks[name]
	if ok {
		return val.Average()
	} else {
		return -1
	}
}

func init() {
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
				ct := rec_build(n.children[c], t[c+1]) //base+n.root.arity+1)
				t = append(t, ct...)
			}
			return t
		} else {
			return []cInt{cInt(n.root.id)}
		}
	}

	return rec_build(root, 0)
}

// Return a string representing a tree (S-expr)
func write_tree(el *Node) string {
	if el.root.isFunc {
		out := fmt.Sprintf("(%v[%v] ", el.root.name, el.root.id) //"(" + el.root.name + "[" + "] "
		for i := cInt(0); i < el.root.arity-1; i++ {
			out += write_tree(el.children[i]) + " "
		}
		return out + write_tree(el.children[el.root.arity-1]) + ")"
	} else {
		return fmt.Sprintf("%v[%v]", el.root.name, el.root.id) // el.root.name // This should be the variable name or the constant value
	}
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

// Evaluates evaluates a tree on the i-th input instance
func eval(tree *Node, i cInt) cFloat64 {
	switch {
	case tree.root == nil:
		// If root is nil, this tree has been created by parse_sem
		return tree.children[i].root.value
	case tree.root.isFunc:
		switch tree.root.name {
		case "+":
			return eval(tree.children[0], i) + eval(tree.children[1], i)
		case "-":
			return eval(tree.children[0], i) - eval(tree.children[1], i)
		case "*":
			return eval(tree.children[0], i) * eval(tree.children[1], i)
		case "/":
			return protected_division(eval(tree.children[0], i), eval(tree.children[1], i))
		case "sqrt":
			v := eval(tree.children[0], i)
			if v < 0 {
				return cFloat64(math.Sqrt(float64(-v)))
			} else {
				return cFloat64(math.Sqrt(float64(v)))
			}
		//case "^":
		//	return math.Pow(eval(tree.children[0], i), eval(tree.children[1], i))
		default:
			panic("Undefined symbol: '" + tree.root.name + "'")
		}
	default:
		return terminal_value(i, tree.root) // Root points to a terminal
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
	f, s := semantic_evaluate(p.individuals[0], cInt(nrow), 0)
	fit[0] = f
	copy(sem_train_cases[0], s)

	f, s = semantic_evaluate(p.individuals[0], cInt(nrow_test), cInt(nrow))
	fit_test[0] = f
	copy(sem_test_cases[0], s)

	for i := 1; i < *config.population_size; i++ {
		f, s = semantic_evaluate(p.individuals[i], cInt(nrow), 0)
		fit[i] = f
		copy(sem_train_cases[i], s)

		f, s = semantic_evaluate(p.individuals[i], cInt(nrow_test), cInt(nrow))
		fit_test[i] = f
		copy(sem_test_cases[i], s)
	}
}

// Calculates semantic and training fitness of an individual (representing as a tree)
func semantic_evaluate(el *Node, sem_size, sem_offs cInt) (cFloat64, Semantic) {
	var d cFloat64
	val := make(Semantic, sem_size)
	if !*config.use_goroutines {
		for i := sem_offs; i < sem_offs+sem_size; i++ {
			res := eval(el, i)
			val[i-sem_offs] = res
			d += dist_func(set[i].y_value, res)
		}
	} else {
		// Communication channel
		ch := make(chan cFloat64)
		// Create some workers to work on chunks of rows
		nw := cInt(runtime.NumCPU())
		for w := cInt(0); w < nw; w++ {
			go func(id cInt) {
				var wd cFloat64
				// Each worker works on a separated share
				for i := sem_offs + id; i < sem_offs+sem_size; i += nw {
					res := eval(el, i)
					val[i-sem_offs] = res
					wd += dist_func(set[i].y_value, res)
				}
				ch <- wd // Send partial results
			}(w)
		}
		for w := cInt(0); w < nw; w++ {
			d += <-ch
		}
	}
	d = d / cFloat64(nrow_test)
	return d, val
}

func semantic_evaluate_array(tree *[]cInt, sem_size, sem_offs cInt) (cFloat64, Semantic) {
	var d cFloat64
	val := make(Semantic, sem_size)
	if !*config.use_goroutines {
		for i := sem_offs; i < sem_offs+sem_size; i++ {
			res := eval_arrays(*tree, 0, i)
			val[i-sem_offs] = res
			d += dist_func(set[i].y_value, res)
		}
	}
	d = d / cFloat64(nrow_test)
	return d, val
}

/*
func random_tree_semantics() (Semantic, Semantic) {
	bmk := StartBenchmark("random_tree")
	defer bmk.Measure()

	// Use same array for riproducible results
	rt := create_grow_tree_arrays(0, cInt(*config.max_depth_creation), 0)

	if !*config.use_cuda {
		// Generate a random tree and compute its semantic (train and test)
		//	rt := create_grow_tree(0, nil, cInt(*config.max_depth_creation))
		//_, sem_train := semantic_evaluate(rt, cInt(nrow), 0)
		//_, sem_test := semantic_evaluate(rt, cInt(nrow_test), cInt(nrow))
		_, sem_train := semantic_evaluate_array(&rt, cInt(nrow), 0)
		_, sem_test := semantic_evaluate_array(&rt, cInt(nrow_test), cInt(nrow))

		return sem_train, sem_test
	} else {
		// Create a tree and copy it to unified memory
		//		rt := create_grow_tree_arrays(0, cInt(*config.max_depth_creation), 0)
		cu_tmp_tree_arr1.FromHostN(unsafe.Pointer(&rt[0]), C.sizeof_int*len(rt))

		// Evaluate the tree on GPU
		kern_eval_array.Launch1D(
			cu_bgp_ds,        // Number of blocks for dataset
			cu_tpb,           // Number of threads per block is fixed
			0,                // No shared memory needed
			cu_sym_val,       // Value for symbols
			cu_set,           // Dataset
			cu_tmp_tree_arr1, // Tree to evaluate
			cu_tmp_sem_tot1)  // Output, semantic for whole dataset

		// Copy back the results and return
		sem_tot := make([]cFloat64, nrow+nrow_test)
		cu_tmp_sem_tot1.FromDevice(unsafe.Pointer(&sem_tot[0]))
		sem_train, sem_test := sem_tot[:nrow], sem_tot[nrow:]
		return sem_train, sem_test
	}
}
*/

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
	fit_new[old_i] = fit[i]
	copy(sem_test_cases_new[old_i], sem_test_cases[i])
	fit_test_new[old_i] = fit_test[i]
}

// Performs a geometric semantic crossover
func geometric_semantic_crossover(i cInt) {
	bmk := StartBenchmark("crossover")
	defer bmk.Measure()

	if i != index_best {
		// Create random tree
		rt := create_grow_tree_arrays(0, cInt(*config.max_depth_creation), 0)
		// Replace the individual with the crossover of two parents
		p1 := tournament_selection()
		p2 := tournament_selection()

		if !*config.use_cuda {
			// Generate a random tree and compute its semantic (train and test)
			//sem_rt, sem_rt_test := random_tree_semantics()
			_, sem_rt := semantic_evaluate_array(&rt, cInt(nrow), 0)
			_, sem_rt_test := semantic_evaluate_array(&rt, cInt(nrow_test), cInt(nrow))

			// Compute the geometric semantic (train)
			for j := 0; j < nrow; j++ {
				sigmoid := 1 / (1 + exp64(-sem_rt[j]))
				sem_train_cases_new[i][j] = sem_train_cases[p1][j]*sigmoid + sem_train_cases[p2][j]*(1-sigmoid)
			}
			fit_new[i] = fitness_of_semantic(sem_train_cases_new[i], cInt(nrow), 0)
			// Compute the geometric semantic (test)
			for j := 0; j < nrow_test; j++ {
				sigmoid := 1 / (1 + exp64(-sem_rt_test[j]))
				sem_test_cases_new[i][j] = sem_test_cases[p1][j]*sigmoid + sem_test_cases[p2][j]*(1-sigmoid)
			}
			fit_test_new[i] = fitness_of_semantic(sem_test_cases_new[i], cInt(nrow_test), cInt(nrow))
		} else {
			// Create a tree and copy it to unified memory
			cu_tmp_tree_arr1.FromHostN(unsafe.Pointer(&rt[0]), C.sizeof_int*len(rt))

			// Evaluate the tree on GPU
			kern_eval_array.Launch1D(
				cu_bgp_ds,        // Number of blocks for dataset
				cu_tpb,           // Number of threads per block is fixed
				0,                // No shared memory needed
				cu_sym_val,       // Value for symbols
				cu_set,           // Dataset
				cu_tmp_tree_arr1, // Tree to evaluate
				cu_tmp_sem_tot1)  // Output, semantic for whole dataset

			// Run kernel that perform crossover
			kern_crossover.Launch1D(
				cu_bgp_ds,
				cu_tpb,
				0,
				cu_tmp_sem_tot1,           // Semantic to use
				cu_sem_train_cases[p1],    // Old train semantic of first parent
				cu_sem_test_cases[p1],     // Old test semantic of first parent
				cu_sem_train_cases[p2],    // Old train semantic of second parent
				cu_sem_test_cases[p2],     // Old test semantic of second parent
				cu_sem_train_cases_new[i], // Destination for new train sem
				cu_sem_test_cases_new[i],  // Destination for new test sem
			)

			// Evaluate fitness
			kern_fitness.Launch1D(1, cu_tpb, 0, cu_set, cu_sem_train_cases_new[i], cu_sem_test_cases_new[i], cu_tmp_d1, cu_tmp_d2)

			ctx.Synchronize()

			fit_new[i] = cFloat64(cu_tmp_d1.ToFloat64())
			fit_test_new[i] = cFloat64(cu_tmp_d2.ToFloat64())
		}
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
	bmk := StartBenchmark("mutation")
	defer bmk.Measure()

	if i != index_best {
		mut_step := cFloat64(rand.Float64())
		// Create two random trees and copy it to unified memory
		rt1 := create_grow_tree_arrays(0, cInt(*config.max_depth_creation), 0)
		rt2 := create_grow_tree_arrays(0, cInt(*config.max_depth_creation), 0)

		if !*config.use_cuda {
			// Replace the individual with a mutated version
			//		sem_rt1, sem_rt1_test := random_tree_semantics()
			//		sem_rt2, sem_rt2_test := random_tree_semantics()

			_, sem_rt1 := semantic_evaluate_array(&rt1, cInt(nrow), 0)
			_, sem_rt1_test := semantic_evaluate_array(&rt1, cInt(nrow_test), cInt(nrow))

			_, sem_rt2 := semantic_evaluate_array(&rt2, cInt(nrow), 0)
			_, sem_rt2_test := semantic_evaluate_array(&rt2, cInt(nrow_test), cInt(nrow))

			for j := 0; j < nrow; j++ {
				sigmoid1 := 1 / (1 + exp64(-sem_rt1[j]))
				sigmoid2 := 1 / (1 + exp64(-sem_rt2[j]))
				sem_train_cases_new[i][j] += mut_step * (sigmoid1 - sigmoid2)
			}
			fit_new[i] = fitness_of_semantic(sem_train_cases_new[i], cInt(nrow), 0)

			for j := 0; j < nrow_test; j++ {
				sigmoid1 := 1 / (1 + exp64(-sem_rt1_test[j]))
				sigmoid2 := 1 / (1 + exp64(-sem_rt2_test[j]))
				sem_test_cases_new[i][j] += mut_step * (sigmoid1 - sigmoid2)
			}
			fit_test_new[i] = fitness_of_semantic(sem_test_cases_new[i], cInt(nrow_test), cInt(nrow))
		} else {
			// Copy trees to GPU
			cu_tmp_tree_arr1.FromHostN(unsafe.Pointer(&rt1[0]), C.sizeof_int*len(rt1))
			cu_tmp_tree_arr2.FromHostN(unsafe.Pointer(&rt2[0]), C.sizeof_int*len(rt2))
			// Evaluate the tree on GPU
			kern_eval_array.Launch1D(cu_bgp_ds, cu_tpb, 0, cu_sym_val, cu_set, cu_tmp_tree_arr1, cu_tmp_sem_tot1)
			kern_eval_array.Launch1D(cu_bgp_ds, cu_tpb, 0, cu_sym_val, cu_set, cu_tmp_tree_arr2, cu_tmp_sem_tot2)

			// Copy mut step to GPU
			cu_tmp_d1.FromFloat64(float64(mut_step))
			// Run kernel that perform mutation
			kern_mutation.Launch1D(cu_bgp_ds, cu_tpb, 0, cu_tmp_sem_tot1, cu_tmp_sem_tot2, cu_tmp_d1, cu_sem_train_cases_new[i], cu_sem_test_cases_new[i])

			// Evaluate fitness
			kern_fitness.Launch1D(1, cu_tpb, 0, cu_set, cu_sem_train_cases_new[i], cu_sem_test_cases_new[i], cu_tmp_d1, cu_tmp_d2)
			// Copy back results
			ctx.Synchronize()
			fit_new[i] = cFloat64(cu_tmp_d1.ToFloat64())
			fit_test_new[i] = cFloat64(cu_tmp_d2.ToFloat64())
		}
	}
	// Mutation happens after reproduction: elite are reproduced but are not mutated
}

// Given a semantic, compute the fitness of a subset of that semantic as the
// Mean Squared Difference between the semantic and the dataset.
// Only sem_size elements, starting from sem_offs, will be considered in the computation
func fitness_of_semantic(sem Semantic, sem_size, sem_offs cInt) cFloat64 {
	var d cFloat64
	for j := sem_offs; j < sem_offs+sem_size; j++ {
		d += dist_func(set[j].y_value, sem[j-sem_offs])
	}
	return d / cFloat64(sem_size)
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
	// Swap cuda buffers if needed
	if *config.use_cuda {
		cu_fit, cu_fit_new = cu_fit_new, cu_fit
		cu_fit_test, cu_fit_test_new = cu_fit_test_new, cu_fit_test
		cu_sem_train_cases, cu_sem_train_cases_new = cu_sem_train_cases_new, cu_sem_train_cases
		cu_sem_test_cases, cu_sem_test_cases_new = cu_sem_test_cases_new, cu_sem_test_cases
	}
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
		//f = ioutil.Discard
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

// Allocates a slice of given size onto Unified Memory and returns
// the memory as well as the cuda buffer handler
// If not using CUDA, a regular allocation will happen and buffer
// will be nil
func alloc_cFloat64(size int) (*cuda.Buffer, []cFloat64) {
	if *config.use_cuda {
		buf, ptr := cuda.AllocManaged(C.sizeof_double * size)
		data := (*[2 << 30]cFloat64)(ptr)
		return buf, data[:size]
	} else {
		return nil, make([]cFloat64, size)
	}
}

func alloc_cInt(size int) (*cuda.Buffer, []cInt) {
	if *config.use_cuda {
		buf, ptr := cuda.AllocManaged(C.sizeof_int * size)
		data := (*[2 << 30]cInt)(ptr)
		return buf, data[:size]
	} else {
		return nil, make([]cInt, size)
	}
}

func alloc_and_copy_cInt(v []cInt) (*cuda.Buffer, []cInt) {
	// Allocate data and copy
	buf, sli := alloc_cInt(len(v))
	for i := 0; i < len(v); i++ {
		sli[i] = v[i]
	}
	return buf, sli
}

// Allocate memory for fitness and semantic value for each individual
func init_tables() {
	cu_fit, fit = alloc_cFloat64(*config.population_size)                   // make([]cFloat64, *config.population_size)
	cu_fit_test, fit_test = alloc_cFloat64(*config.population_size)         // make([]cFloat64, *config.population_size)
	cu_fit_new, fit_new = alloc_cFloat64(*config.population_size)           // make([]cFloat64, *config.population_size)
	cu_fit_test_new, fit_test_new = alloc_cFloat64(*config.population_size) // make([]cFloat64, *config.population_size)

	cu_sem_train_cases = make([]*cuda.Buffer, *config.population_size)
	cu_sem_train_cases_new = make([]*cuda.Buffer, *config.population_size)
	cu_sem_test_cases = make([]*cuda.Buffer, *config.population_size)
	cu_sem_test_cases_new = make([]*cuda.Buffer, *config.population_size)

	sem_train_cases = make([]Semantic, *config.population_size)
	sem_train_cases_new = make([]Semantic, *config.population_size)
	sem_test_cases = make([]Semantic, *config.population_size)
	sem_test_cases_new = make([]Semantic, *config.population_size)

	for i := 0; i < *config.population_size; i++ {
		cu_sem_train_cases[i], sem_train_cases[i] = alloc_cFloat64(nrow)            // make(Semantic, nrow)
		cu_sem_train_cases_new[i], sem_train_cases_new[i] = alloc_cFloat64(nrow)    // make(Semantic, nrow)
		cu_sem_test_cases[i], sem_test_cases[i] = alloc_cFloat64(nrow_test)         // make(Semantic, nrow_test)
		cu_sem_test_cases_new[i], sem_test_cases_new[i] = alloc_cFloat64(nrow_test) // make(Semantic, nrow_test)
	}
}

func bpg(nThreads int) int {
	return (nThreads + cu_tpb - 1) / cu_tpb
}

func main() {
	// CUDA context is bound to a specific thread, therefore it is necessary to lock this
	// goroutine to the current thread
	runtime.LockOSThread()
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

	if *config.use_goroutines {
		log.Println("Using goroutines with", runtime.NumCPU(), "CPUs")
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

	if *config.use_cuda {
		// Initialize CUDA environment
		cuda.Init()
		// Get CUDA device
		devs := cuda.GetDevices()
		maj, min := cuda.GetNVRTCVersion()
		// Be verbose on GPU being used
		log.Println("CUDA Driver Version:", cuda.GetVersion())
		log.Println("NVRTC Version:", maj, min)
		log.Println("CUDA Num devices:", cuda.GetDevicesCount())
		log.Println("Compute devices")
		for i, d := range devs {
			log.Printf("Device %d: %s %v bytes of memory\n", i, d.Name, d.TotalMem)
			mbx, mby, mbz := d.GetMaxBlockDim()
			log.Println("Max block size:", mbx, mby, mbz)
			mgx, mgy, mgz := d.GetMaxGridDim()
			log.Println("Max grid size:", mgx, mgy, mgz)
		}
		// Create context and make it current
		ctx = cuda.Create(devs[0], 0)
		defer ctx.Destroy() // When done

		log.Println("Context API version:", ctx.GetApiVersion())
		ctx.Synchronize() // Check for errors
		log.Println("CUDA initialized successfully")

		// Allocate room for dataset and copy
		//	var tmp_set []cFloat64
		//	cu_set, tmp_set = alloc_cFloat64((nrow + nrow_test) * (nvar + 1))

		// Dataset is never modified, so there is no advantage in using Unified Memory
		var tmp_set []cFloat64 = make([]cFloat64, (nrow+nrow_test)*(nvar+1))
		cu_set = cuda.NewBuffer(int(C.sizeof_double * (nrow + nrow_test) * (nvar + 1))) // Storage for dataset
		// Copy datasets, including target which is used to compute fitness
		for i := 0; i < nrow+nrow_test; i++ {
			for j := 0; j < nvar; j++ {
				tmp_set[i*(nvar+1)+j] = cFloat64(set[i].vars[j])
			}
			tmp_set[i*(nvar+1)+nvar] = cFloat64(set[i].y_value)
		}
		cu_set.FromHost(unsafe.Pointer(&tmp_set[0]))

		// Allocate room for symbols and copy
		//	var tmp_sym_val []cFloat64
		//	cu_sym_val, tmp_sym_val = alloc_cFloat64(len(symbols))

		// Symbols are never modified, so there is no advantage in using Unified Memory
		var tmp_sym_val = make([]cFloat64, len(symbols))
		cu_sym_val = cuda.NewBuffer(C.sizeof_double * len(symbols))
		// Copy symbols to temporary memory
		for i := range symbols {
			if !symbols[i].isFunc {
				tmp_sym_val[i] = cFloat64(symbols[i].value)
			} else {
				tmp_sym_val[i] = -1 // Functionals have no value
			}
		}
		cu_sym_val.FromHost(unsafe.Pointer(&tmp_sym_val[0]))

		// Setup some temporary memory
		/*
			cu_tmp_sem_train, tmp_sem_train = alloc_cFloat64(nrow)
			cu_tmp_sem_test, tmp_sem_test = alloc_cFloat64(nrow_test)
		*/
		cu_tmp_sem_train = cuda.NewBuffer(C.sizeof_double * nrow)
		cu_tmp_sem_test = cuda.NewBuffer(C.sizeof_double * nrow_test)
		cu_tmp_sem_tot1 = cuda.NewBuffer(C.sizeof_double * (nrow + nrow_test))
		cu_tmp_sem_tot2 = cuda.NewBuffer(C.sizeof_double * (nrow + nrow_test))
		cu_tmp_tree_arr1 = cuda.NewBuffer(C.sizeof_int * (2 << uint(*config.max_depth_creation+1))) // Storage for generated trees
		cu_tmp_tree_arr2 = cuda.NewBuffer(C.sizeof_int * (2 << uint(*config.max_depth_creation+1))) // Storage for generated trees
		cu_tmp_d1 = cuda.NewBuffer(C.sizeof_double)
		cu_tmp_d2 = cuda.NewBuffer(C.sizeof_double)

		// Number of blocks required to have at least 1 thread for each row in dataset
		cu_bgp_ds = bpg(nrow + nrow_test)
		cu_bpg_pop = bpg(*config.population_size)

		// Load CUDA code and replace some variables
		kernel_src := load_file_and_replace("./kernels.cu", map[string]interface{}{
			"NUM_FUNCTIONAL_SYMBOLS": NUM_FUNCTIONAL_SYMBOLS,
			"NUM_VARIABLE_SYMBOLS":   NUM_VARIABLE_SYMBOLS,
			"NUM_CONSTANT_SYMBOLS":   NUM_CONSTANT_SYMBOLS,
			"NROWS_TRAIN":            nrow,
			"NROWS_TEST":             nrow_test,
			"NROWS_TOT":              nrow + nrow_test,
			"NUM_THREADS":            cu_tpb,
		})

		// Prepare kernels to eval and reduce trees
		cu_mod = cuda.CreateModule()
		cu_prog = cuda.CreateProgram(cuda.Source{kernel_src, "semantic_eval_arrays"})
		cu_prog.Compile()
		cu_mod.LoadData(cu_prog)

		kern_eval_array = cu_mod.GetFunction("semantic_eval_arrays")
		kern_crossover = cu_mod.GetFunction("sem_crossover")
		kern_fitness = cu_mod.GetFunction("sem_fitness")
		kern_mutation = cu_mod.GetFunction("sem_mutation")
	}

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
	log.Println("Benchmark:", BenchResult("random_tree"), BenchResult("crossover"), BenchResult("mutation"))
}
