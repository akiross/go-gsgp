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

const use_goroutines_for_fitness = true

// Instance represent a single training/test instance in memory
type Instance struct {
	vars    []float64 // Values of the input (independent) variables
	y_value float64   // Target value
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
	path_in, path_test     *string
	rng_seed               *int64
}

// Symbol represents a symbol of the set T (terminal symbols) or F (functional symbols).
type Symbol struct {
	isFunc bool    // Functional or terminal
	arity  int     // Number of arguments accepted by a symbol. Terminals have arity -1 when constants and the index of the variable otherwise
	id     int     // Unique identifier for this symbol
	name   string  // Symbolic name
	value  float64 // Current value of terminal symbol
}

// Node is used to represent a node of the tree.
type Node struct {
	root     *Symbol // Symbol for the node
	parent   *Node   // Parent of the node, if any (can be nil)
	children []*Node // Child nodes, can be empty
}

// Population is used to represent a GP population.
type Population struct {
	individuals []*Node   // Individuals' root node
	num_ind     int       // Number of individuals in the population
	index_best  int       // Index of the best individual after evaluate is run
	fitness     []float64 // Fitness values for each individual
}

// The Semantic of one individual is a vector as long as the dataset where each
// component is the value obtaining by applying the individual to the datum.
type Semantic []float64

var (
	// Create flag/configuration variables with default values (in case config file is missing)
	config_file = flag.String("config", "configuration.ini", "Path of the configuration file")
	config      = Config{
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
	}
	cpuprofile = flag.String("cpuprofile", "", "Write CPU profile to file")

	NUM_FUNCTIONAL_SYMBOLS int // Number of functional symbols
	NUM_VARIABLE_SYMBOLS   int // Number of terminal symbols for variables
	NUM_CONSTANT_SYMBOLS   int // Number of terminal symbols for constants

	// Terminal and functional symbols
	// This slice is filled only by create_T_F() and search_terminal_or_add()
	// len(symbols) == NUM_FUNCTIONAL_SYMBOLS+NUM_VARIABLE_SYMBOLS+NUM_CONSTANT_SYMBOLS
	// In this slice, first you find NUM_FUNCTIONAL_SYMBOLS symbols, then
	// NUM_VARIABLE_SYMBOLS symbols, finally NUM_CONSTANT_SYMBOLS symbols
	symbols = make([]*Symbol, 0)

	set       []Instance // Store training and test instances
	nrow      int        // Number of rows (instances) in training dataset
	nvar      int        // Number of variables (columns excluding target) in training dataset
	nrow_test int        // Number of rows (instances) in test dataset
	nvar_test int        // Number of input variables (columns excluding target) in test dataset

	fit          = make([]float64, 0) // Training fitness values at generation g
	fit_test     = make([]float64, 0) // Test fitness values at generation g
	fit_new      = make([]float64, 0) // Training fitness values at current generation g+1
	fit_new_test = make([]float64, 0) // Test fitness values at current generation g+1

	sem_train_cases     = make([]Semantic, 0) // Semantics of the population, computed on training set, at generation g
	sem_train_cases_new = make([]Semantic, 0) // Semantics of the population, computed on training set, at current generation g+1
	sem_test_cases      = make([]Semantic, 0) // Semantics of the population, computed on test set, at generation g
	sem_test_cases_new  = make([]Semantic, 0) // Semantics of the population, computed on test set, at current generation g+1

	index_best int // Index of the best individual (where? sem_*?)
)

func init() {
	// lettura qui così ci permette di usare un path diverso per il config file
	// Read variables: if present in the config, they will override the defaults
	read_config_file(*config_file)
}

func square_diff(a, b float64) float64 { return (a - b) * (a - b) }

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
			NUM_CONSTANT_SYMBOLS = *config.num_random_constants
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
	nvar = atoi(next_token(in)) // Number of variables
	nvar_test = atoi(next_token(in_test))
	nrow = atoi(next_token(in)) // Number of rows
	nrow_test = atoi(next_token(in_test))
	set = make([]Instance, nrow+nrow_test)
	for i := 0; i < nrow; i++ {
		set[i].vars = make([]float64, nvar)
		for j := 0; j < nvar; j++ {
			set[i].vars[j] = atof(next_token(in))
		}
		set[i].y_value = atof(next_token(in))
	}
	for i := nrow; i < nrow+nrow_test; i++ {
		set[i].vars = make([]float64, nvar)
		for j := 0; j < nvar; j++ {
			set[i].vars[j] = atof(next_token(in_test))
		}
		set[i].y_value = atof(next_token(in_test))
	}
}

// create_T_F creates the terminal and functional sets
// Names in created symbols shall not include the characters '(' or ')'
// because they are used when reading and writing a tree to string
func create_T_F() {
	NUM_VARIABLE_SYMBOLS = nvar
	// Create functional symbols
	fs := []struct {
		name  string
		arity int
	}{
		{"+", 2},
		{"-", 2},
		{"*", 2},
		{"/", 2},
		{"sqrt", 1},
	}
	NUM_FUNCTIONAL_SYMBOLS = len(fs)
	for i, s := range fs {
		symbols = append(symbols, &Symbol{true, s.arity, i, s.name, 0})
	}
	// Create terminal symbols for variables
	for i := NUM_FUNCTIONAL_SYMBOLS; i < NUM_VARIABLE_SYMBOLS+NUM_FUNCTIONAL_SYMBOLS; i++ {
		str := fmt.Sprintf("x%d", i-NUM_FUNCTIONAL_SYMBOLS)
		symbols = append(symbols, &Symbol{false, i - NUM_FUNCTIONAL_SYMBOLS, i, str, 0})
	}
	// Create terminal symbols for constants
	for i := NUM_VARIABLE_SYMBOLS + NUM_FUNCTIONAL_SYMBOLS; i < NUM_VARIABLE_SYMBOLS+NUM_FUNCTIONAL_SYMBOLS+NUM_CONSTANT_SYMBOLS; i++ {
		a := *config.min_random_constant + rand.Float64()*(*config.max_random_constant-*config.min_random_constant)
		str := fmt.Sprintf("%f", a)
		symbols = append(symbols, &Symbol{false, -1, i, str, a})
	}
}

// choose_function randomly selects a functional symbol
func choose_function() int {
	return rand.Intn(NUM_FUNCTIONAL_SYMBOLS)
}

// choose_terminal randomly selects a terminal symbol.
// With probability 0.7 a variable is selected, while random constants have a probability of 0.3 to be selected.
// To change these probabilities just change their values in the function.
// It returns the ID of the chosen terminal symbol
func choose_terminal() int {
	if NUM_CONSTANT_SYMBOLS == 0 {
		return NUM_FUNCTIONAL_SYMBOLS + rand.Intn(NUM_VARIABLE_SYMBOLS)
	}
	if rand.Float64() < 0.7 {
		return NUM_FUNCTIONAL_SYMBOLS + rand.Intn(NUM_VARIABLE_SYMBOLS)
	}
	return NUM_FUNCTIONAL_SYMBOLS + NUM_VARIABLE_SYMBOLS + rand.Intn(NUM_CONSTANT_SYMBOLS)
}

// create_grow_pop creates a population using the grow method
func create_grow_pop(p *Population) {
	for p.num_ind < *config.population_size {
		node := create_grow_tree(0, nil, *config.max_depth_creation)
		p.individuals[p.num_ind] = node
		p.num_ind++
	}
}

// Creates a population of full trees (each tree has a depth equal to the maximum length possible)
func create_full_pop(p *Population) {
	for p.num_ind < *config.population_size {
		node := create_full_tree(0, nil, *config.max_depth_creation)
		p.individuals[p.num_ind] = node
		p.num_ind++
	}
}

// Creates a population with the ramped half and half algorithm.
func create_ramped_pop(p *Population) {
	var sub_pop, r, min_depth int
	if !*config.zero_depth {
		sub_pop = (*config.population_size - p.num_ind) / *config.max_depth_creation
		r = (*config.population_size - p.num_ind) % *config.max_depth_creation
		min_depth = 1
	} else {
		sub_pop = (*config.population_size - p.num_ind) / (*config.max_depth_creation + 1)
		r = (*config.population_size - p.num_ind) % (*config.max_depth_creation + 1)
		min_depth = 0
	}
	for j := *config.max_depth_creation; j >= min_depth; j-- {
		if j < *config.max_depth_creation {
			for k := 0; k < int(math.Ceil(float64(sub_pop)*0.5)); k++ {
				node := create_full_tree(0, nil, j)
				p.individuals[p.num_ind] = node
				p.num_ind++
			}
			for k := 0; k < int(math.Floor(float64(sub_pop)*0.5)); k++ {
				node := create_grow_tree(0, nil, j)
				p.individuals[p.num_ind] = node
				p.num_ind++
			}
		} else {
			for k := 0; k < int(math.Ceil(float64(sub_pop+r)*0.5)); k++ {
				node := create_full_tree(0, nil, j)
				p.individuals[p.num_ind] = node
				p.num_ind++
			}
			for k := 0; k < int(math.Floor(float64(sub_pop+r)*0.5)); k++ {
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
		num_ind:     len(seeds),
		fitness:     make([]float64, *config.population_size),
	}
	for i := range seeds {
		p.individuals[i] = read_sem(seeds[i])
	}
	return p
}

// Fills the population using the method specified by the parameter
func initialize_population(p *Population, method int) {
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
func create_grow_tree(depth int, parent *Node, max_depth int) *Node {
	if depth == 0 && !*config.zero_depth {
		sym := symbols[choose_function()]
		el := &Node{
			root:     sym,
			parent:   nil,
			children: make([]*Node, sym.arity),
		}
		for i := 0; i < sym.arity; i++ {
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
	if depth > 0 && depth < max_depth || depth == 0 && *config.zero_depth {
		if rand.Intn(2) == 0 {
			sym := symbols[choose_function()]
			el := &Node{
				root:     sym,
				parent:   parent,
				children: make([]*Node, sym.arity),
			}
			for i := 0; i < sym.arity; i++ {
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
	panic("Unreachable code was reached in grow!")
}

// Creates a tree with depth equal to the ones specified by the parameter max_depth
func create_full_tree(depth int, parent *Node, max_depth int) *Node {
	if depth == 0 && depth < max_depth {
		sym := symbols[choose_function()]
		el := &Node{
			root:     sym,
			parent:   nil,
			children: make([]*Node, sym.arity),
		}
		for i := 0; i < sym.arity; i++ {
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
	for i := 0; i < sym.arity; i++ {
		el.children[i] = create_full_tree(depth+1, el, max_depth)
	}
	return el
}

// Return a string representing a tree (S-expr)
func write_tree(el *Node) string {
	if el.root.isFunc {
		out := "(" + el.root.name + " "
		for i := 0; i < el.root.arity-1; i++ {
			out += write_tree(el.children[i]) + " "
		}
		return out + write_tree(el.children[el.root.arity-1]) + ")"
	} else {
		return el.root.name // This should be the variable name or the constant value
	}
}

// Convert string with numeric constant into a symbol and add it to list
func add_symbol(name string) *Symbol {
	val, err := strconv.ParseFloat(name, 64)
	if err != nil {
		return nil // Not a float, must be a wrong variable or functional
	}
	// Conversion was successful, must be a constant
	sym := &Symbol{false, -1, NUM_CONSTANT_SYMBOLS, name, val}
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
func protected_division(num, den float64) float64 {
	if den == 0 {
		return 1
	}
	return num / den
}

// This function retrieves the value of a terminal symbol given
// the i-th instance as input.
func terminal_value(i int, sym *Symbol) float64 {
	if sym.id >= NUM_FUNCTIONAL_SYMBOLS && sym.id < NUM_FUNCTIONAL_SYMBOLS+NUM_VARIABLE_SYMBOLS {
		// Variables take their value from the input data
		return set[i].vars[sym.id-NUM_FUNCTIONAL_SYMBOLS]
	} else {
		// The value of a constant can be used directly
		return sym.value
	}
}

// Evaluates evaluates a tree on the i-th input instance
func eval(tree *Node, i int) float64 {
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
				return math.Sqrt(-v)
			} else {
				return math.Sqrt(v)
			}
		case "^":
			return math.Pow(eval(tree.children[0], i), eval(tree.children[1], i))
		default:
			panic("Undefined symbol: '" + tree.root.name + "'")
		}
	default:
		return terminal_value(i, tree.root) // Root points to a terminal
	}
}

// Calculates the fitness of all the individuals and determines the best individual in the population
// Evaluate is called once, after individuals have been initialized for the first time.
// This function fills p.fitness and p.index_best using semantic_evaluate
func evaluate(p *Population) {
	p.fitness[0] = semantic_evaluate(p.individuals[0])
	p.index_best = 0
	fit = append(fit, p.fitness[0])
	fit_test = append(fit_test, semantic_evaluate_test(p.individuals[0]))
	for i := 1; i < *config.population_size; i++ {
		p.fitness[i] = semantic_evaluate(p.individuals[i])
		fit = append(fit, p.fitness[i])
		fit_test = append(fit_test, semantic_evaluate_test(p.individuals[i]))
		if better(p.fitness[i], p.fitness[p.index_best]) {
			p.index_best = i
		}
	}
}

// Calculates the training fitness of an individual (representing as a tree).
// This function will append to sem_train_cases the semantic of the evaluated node.
func semantic_evaluate(el *Node) float64 {
	var d float64
	val := make(Semantic, nrow)
	if !use_goroutines_for_fitness {
		for i := 0; i < nrow; i++ {
			res := eval(el, i) // Evaluate the element on the i-th instance
			val[i] = res
			d += square_diff(res, set[i].y_value)
		}
	} else {
		// Communication channel
		ch := make(chan float64)
		// Create some workers to work on chunks of rows
		nw := runtime.NumCPU()
		for w := 0; w < nw; w++ {
			go func(id int) {
				var wd float64
				// Each worker uses a partition of the dataset
				for i := id; i < nrow; i += nw {
					res := eval(el, i)
					val[i] = res
					wd += square_diff(res, set[i].y_value)
				}
				ch <- wd // Send partial results
			}(w)
		}
		for w := 0; w < nw; w++ {
			d += <-ch
		}
	}
	sem_train_cases = append(sem_train_cases, val)
	d = d / float64(nrow)
	return d
}

// Calculates the test fitness of an individual (representing as a tree).
// This function will append to sem_test_cases the semantic of the evaluated node.
func semantic_evaluate_test(el *Node) float64 {
	var d float64
	val := make(Semantic, nrow_test)
	if !use_goroutines_for_fitness {
		for i := nrow; i < nrow+nrow_test; i++ {
			res := eval(el, i)
			val[i-nrow] = res
			d += square_diff(res, set[i].y_value)
		}
	} else {
		// Communication channel
		ch := make(chan float64)
		// Create some workers to work on chunks of rows
		nw := runtime.NumCPU()
		for w := 0; w < nw; w++ {
			go func(id int) {
				var wd float64
				// Each worker works on a separated share
				for i := nrow + id; i < nrow+nrow_test; i += nw {
					res := eval(el, i)
					val[i-nrow] = res
					wd += square_diff(res, set[i].y_value)
				}
				ch <- wd // Send partial results
			}(w)
		}
		for w := 0; w < nw; w++ {
			d += <-ch
		}
	}
	sem_test_cases = append(sem_test_cases, val)
	d = d / float64(nrow_test)
	return d
}

// Calculates the semantics (considering training instances) of a randomly generated tree. The tree is used to perform the semantic geometric crossover or the geometric semantic mutation
func semantic_evaluate_random(el *Node) Semantic {
	sem := make(Semantic, nrow)
	if !use_goroutines_for_fitness {
		for i := 0; i < nrow; i++ {
			sem[i] = eval(el, i)
		}
	} else {
		sc := make(chan bool) // Sync channel
		// Create some workers to work on chunks of rows
		nw := runtime.NumCPU()
		for w := 0; w < nw; w++ {
			go func(id int) {
				// Each worker uses a partition of the dataset
				for i := id; i < nrow; i += nw {
					sem[i] = eval(el, i)
				}
				sc <- true
			}(w)
		}
		for w := 0; w < nw; w++ {
			<-sc
		}
	}
	return sem
}

// Calculates the semantics (considering test instances) of a randomly generated tree. The tree is used to perform the semantic geometric crossover or the geometric semantic mutation
func semantic_evaluate_random_test(el *Node) Semantic {
	sem := make(Semantic, nrow_test)
	if !use_goroutines_for_fitness {
		for i := nrow; i < nrow+nrow_test; i++ {
			sem[i-nrow] = eval(el, i)
		}
	} else {
		sc := make(chan bool) // Sync channel
		// Create some workers to work on chunks of rows
		nw := runtime.NumCPU()
		for w := 0; w < nw; w++ {
			go func(id int) {
				// Each worker uses a partition of the dataset
				for i := id + nrow; i < nrow+nrow_test; i += nw {
					sem[i-nrow] = eval(el, i)
				}
				sc <- true
			}(w)
		}
		for w := 0; w < nw; w++ {
			<-sc
		}
	}
	return sem
}

// Implements a tournament selection procedure
func tournament_selection() int {
	// Select first participant
	best_index := rand.Intn(*config.population_size)
	// Pick best individual
	for i := 1; i < *config.tournament_size; i++ {
		next := rand.Intn(*config.population_size)
		if better(fit[next], fit[best_index]) {
			best_index = next
		}
	}
	return best_index
}

// Copies an individual of the population at generation g-1 to the current population(generation g)
func reproduction(i int) {
	// Elitism: if i is the best individual, reproduce it
	if i != index_best {
		// If it's not the best, select one at random to reproduce
		i = tournament_selection()
	}
	// Copy fitness and semantics of the selected individual
	sem_train_cases_new = append(sem_train_cases_new, sem_train_cases[i])
	fit_new = append(fit_new, fit[i])
	sem_test_cases_new = append(sem_test_cases_new, sem_test_cases[i])
	fit_new_test = append(fit_new_test, fit_test[i])
}

// Performs a geometric semantic crossover
func geometric_semantic_crossover(i int) {
	if i != index_best {
		// Replace the individual with the crossover of two parents
		p1 := tournament_selection()
		p2 := tournament_selection()
		// Generate a random tree and compute its semantic (train and test)
		rt := create_grow_tree(0, nil, *config.max_depth_creation)
		sem_rt := semantic_evaluate_random(rt)
		sem_rt_test := semantic_evaluate_random_test(rt)
		// Compute the geometric semantic (train)
		val := make(Semantic, nrow)
		val_test := make(Semantic, nrow_test)
		for j := 0; j < nrow; j++ {
			sigmoid := 1 / (1 + math.Exp(-sem_rt[j]))
			val[j] = sem_train_cases[p1][j]*sigmoid + sem_train_cases[p2][j]*(1-sigmoid)
		}
		sem_train_cases_new = append(sem_train_cases_new, val)
		update_training_fitness(val, true)
		// Compute the geometric semantic (test)
		for j := 0; j < nrow_test; j++ {
			sigmoid := 1 / (1 + math.Exp(-sem_rt_test[j]))
			val_test[j] = sem_test_cases[p1][j]*sigmoid + sem_test_cases[p2][j]*(1-sigmoid)
		}
		sem_test_cases_new = append(sem_test_cases_new, val_test)
		update_test_fitness(val_test, true)
	} else {
		// The best individual will not be changed
		sem_train_cases_new = append(sem_train_cases_new, sem_train_cases[i])
		fit_new = append(fit_new, fit[i])
		sem_test_cases_new = append(sem_test_cases_new, sem_test_cases[i])
		fit_new_test = append(fit_new_test, fit_test[i])
	}
}

// Performs a geometric semantic mutation
func geometric_semantic_mutation(i int) {
	if i != index_best {
		// Replace the individual with a mutated version
		rt1 := create_grow_tree(0, nil, *config.max_depth_creation)
		rt2 := create_grow_tree(0, nil, *config.max_depth_creation)

		sem_rt1 := semantic_evaluate_random(rt1)
		sem_rt1_test := semantic_evaluate_random_test(rt1)
		sem_rt2 := semantic_evaluate_random(rt2)
		sem_rt2_test := semantic_evaluate_random_test(rt2)

		mut_step := rand.Float64()

		for j := 0; j < nrow; j++ {
			sigmoid1 := 1 / (1 + math.Exp(-sem_rt1[j]))
			sigmoid2 := 1 / (1 + math.Exp(-sem_rt2[j]))
			sem_train_cases_new[i][j] = sem_train_cases_new[i][j] + mut_step*(sigmoid1-sigmoid2)
		}
		update_training_fitness(sem_train_cases_new[i], false)

		for j := 0; j < nrow_test; j++ {
			sigmoid1 := 1 / (1 + math.Exp(-sem_rt1_test[j]))
			sigmoid2 := 1 / (1 + math.Exp(-sem_rt2_test[j]))
			sem_test_cases_new[i][j] = sem_test_cases_new[i][j] + mut_step*(sigmoid1-sigmoid2)
		}
		update_test_fitness(sem_test_cases_new[i], false)
	} else {
		// The best individual will not be changed
		sem_train_cases_new = append(sem_train_cases_new, sem_train_cases[i])
		fit_new = append(fit_new, fit[i])
		sem_test_cases_new = append(sem_test_cases_new, sem_test_cases[i])
		fit_new_test = append(fit_new_test, fit_test[i])
	}
}

// Calculates the training fitness of an individual using the information stored in its semantic vector.
// The function updates the data structure that stores the training fitness of the individuals
func update_training_fitness(semantic_values Semantic, crossover bool) {
	var d float64
	for j := 0; j < nrow; j++ {
		d += square_diff(semantic_values[j], set[j].y_value)
	}
	if crossover {
		fit_new = append(fit_new, d/float64(nrow))
	} else {
		fit_new[len(fit_new)-1] = d / float64(nrow)
	}
}

// Calculates the test fitness of an individual using the information stored in its semantic vector.
// The function updates the data structure that stores the test fitness of the individuals
func update_test_fitness(semantic_values Semantic, crossover bool) {
	var d float64
	for j := nrow; j < nrow+nrow_test; j++ {
		d += square_diff(semantic_values[j-nrow], set[j].y_value)
	}
	if crossover {
		fit_new_test = append(fit_new_test, d/float64(nrow_test))
	} else {
		fit_new_test[len(fit_new_test)-1] = d / float64(nrow_test)
	}
}

// Finds the best individual in the population
func best_individual() int {
	var best_index int
	for i := 1; i < len(fit); i++ {
		if better(fit[i], fit[best_index]) {
			best_index = i
		}
	}
	return best_index
}

// Updates the tables used to store fitness values and semantics of the individual. It is used at the end of each iteration of the algorithm
func update_tables() {
	fit = fit_new
	fit_new = make([]float64, 0)
	sem_train_cases = sem_train_cases_new
	sem_train_cases_new = make([]Semantic, 0)
	fit_test = fit_new_test
	fit_new_test = make([]float64, 0)
	sem_test_cases = sem_test_cases_new
	sem_test_cases_new = make([]Semantic, 0)
}

func next_token(in *bufio.Scanner) string {
	in.Scan()
	return in.Text()
}

// Compares the fitness of two solutions.
func better(f1, f2 float64) bool {
	if *config.minimization_problem {
		return f1 < f2
	} else {
		return f1 > f2
	}
}

// Calculates the number of nodes of a solution.
func node_count(el *Node) int {
	counter := 1
	if el.children != nil {
		for i := 0; i < el.root.arity; i++ {
			counter += node_count(el.children[i])
		}
	}
	return counter
}

func create_or_panic(path string) *os.File {
	f, err := os.Create(path)
	if err != nil {
		panic(err)
	}
	return f
}

func main() {
	runtime.LockOSThread() // For CUDA context

	// Parse CLI arguments: if they are set, they override config file
	flag.Parse()

	if *config.path_in == "" {
		fmt.Println("Please specify the train dataset using the train_file option")
		return
	}
	if *config.path_test == "" {
		fmt.Println("Please specify the test dataset using the test_file option")
		return
	}

	fmt.Println("NumCPU:", runtime.NumCPU())

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

	// Initialize CUDA environment
	cuda.Init()
	devs := cuda.GetDevices()
	if true {
		maj, min := cuda.GetNVRTCVersion()
		fmt.Println("CUDA Driver Version:", cuda.GetVersion())
		fmt.Println("NVRTC Version:", maj, min)
		fmt.Println("CUDA Num devices:", cuda.GetDevicesCount())
		fmt.Println("Compute devices")
		for i, d := range devs {
			fmt.Printf("Device %d: %s %v bytes of memory\n", i, d.Name, d.TotalMem)
			mbx, mby, mbz := d.GetMaxBlockDim()
			fmt.Println("Max block size:", mbx, mby, mbz)
			mgx, mgy, mgz := d.GetMaxGridDim()
			fmt.Println("Max grid size:", mgx, mgy, mgz)
		}
	}

	// CUDA context is bound to a specific thread, therefore it is necessary to lock this
	// goroutine to the current thread
	runtime.LockOSThread()
	// Create context and make it current
	ctx := cuda.Create(devs[0], 0)
	runtime.SetFinalizer(ctx, func(interface{}) { fmt.Println("FINALIZER FOR CTX called!") })
	//defer ctx.Destroy() // When done
	//ctx.SetCurrent()
	fmt.Println("Context API version:", ctx.GetApiVersion())

	prog := cuda.CreateProgram(cuda.Source{`
extern "C" __global__
void somma(int *a, int *b, int *c, int *len) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < *len)
		c[i] = a[i] + b[i];
}`, "somma"}, nil)
	prog.Compile(nil)

	mod := cuda.CreateModule()
	mod.LoadData(prog)
	kernel := mod.GetFunction("somma")

	fmt.Println("CUDA initialized successfully")

	executiontime := create_or_panic("execution_time.txt")
	defer executiontime.Close()

	var start time.Time
	start = time.Now()

	fitness_train := create_or_panic("fitnesstrain.txt")
	defer fitness_train.Close()
	fitness_test := create_or_panic("fitnesstest.txt")
	defer fitness_test.Close()
	// Seed RNG
	rand.Seed(*config.rng_seed)
	read_input_data(*config.path_in, *config.path_test)
	create_T_F()

	fmt.Println("CUDA ctx sync...")
	ctx.Synchronize()
	fmt.Println("Synched")

	// Allocate memory for storing semantic data
	cuda_data_size := C.size_t(nrow + nrow_test)
	hlen := (*[1]C.int)(C.malloc(C.sizeof_int)) // Number of elements
	defer C.free(unsafe.Pointer(&hlen[0]))
	ha := (*[1 << 30]C.int)(C.malloc(C.sizeof_int * cuda_data_size))
	defer C.free(unsafe.Pointer(&ha[0]))
	hb := (*[1 << 30]C.int)(C.malloc(C.sizeof_int * cuda_data_size))
	defer C.free(unsafe.Pointer(&hb[0]))
	hc := (*[1 << 30]C.int)(C.malloc(C.sizeof_int * cuda_data_size))
	defer C.free(unsafe.Pointer(&hc[0]))

	for i := 0; i < int(cuda_data_size); i++ {
		ha[i] = C.int(i + 1)
		hb[i] = C.int(10000 - i*i)
		hc[i] = -1
	}

	dlen := cuda.NewBuffer(C.sizeof_int)
	da := cuda.NewBuffer(int(C.sizeof_int * cuda_data_size))
	db := cuda.NewBuffer(int(C.sizeof_int * cuda_data_size))
	dc := cuda.NewBuffer(int(C.sizeof_int * cuda_data_size))

	dlen.FromHost(unsafe.Pointer(&hlen[0]))
	da.FromHost(unsafe.Pointer(&ha[0]))
	db.FromHost(unsafe.Pointer(&hb[0]))

	tpb := 256 // Get this from attr
	bpg := (nrow + nrow_test + tpb - 1) / tpb
	kernel.Launch1D(bpg, tpb, 0, da, db, dc, dlen)

	dc.FromDevice(unsafe.Pointer(&hc[0]))

	fmt.Println("CUDA compute done")

	// Create population and feed
	p := NewPopulation(sem_seed...)
	initialize_population(p, *config.init_type)
	evaluate(p)
	fmt.Fprintln(fitness_train, semantic_evaluate(p.individuals[p.index_best]))
	fmt.Fprintln(fitness_test, semantic_evaluate_test(p.individuals[p.index_best]))
	index_best = best_individual()

	elapsedTime := time.Since(start) / time.Millisecond
	fmt.Fprintln(executiontime, elapsedTime)

	// main GP cycle
	for num_gen := 0; num_gen < *config.max_number_generations; num_gen++ {
		var gen_start = time.Now()

		fmt.Println("Generation", num_gen+1)
		for k := 0; k < *config.population_size; k++ {
			rand_num := rand.Float64()
			switch {
			case rand_num < *config.p_crossover:
				geometric_semantic_crossover(k)
			case rand_num < *config.p_crossover+*config.p_mutation:
				reproduction(k)
				geometric_semantic_mutation(k)
			default:
				reproduction(k)
			}
		}

		update_tables()
		index_best = best_individual()

		fmt.Fprintln(fitness_train, fit[index_best])
		fmt.Fprintln(fitness_test, fit_test[index_best])

		elapsedTime += time.Since(gen_start) / time.Millisecond
		fmt.Fprintln(executiontime, elapsedTime)
	}

	//runtime.KeepAlive(ctx)
	ctx.Destroy() // When done
}
