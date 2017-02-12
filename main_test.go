package main

import "testing"

func testMalformed(t *testing.T) {
	runTest := func(errorMessage, malformedString string) {
		defer func() {
			err := recover()
			if err == nil {
				t.Error("read_tree did not panic for", errorMessage, "in", malformedString)
			} else if err.(string) != errorMessage {
				t.Error("read_tree panicked with wrong message: ", err.(string), "instead of", errorMessage, "for", malformedString)
			}
		}()

		read_tree(malformedString)
	}
	/* Bad things that could happen:
	- parentheses not opened (extra trailing chars are tolerated)
	- token not a valid float
	- unknown variable
	- unknown operation
	- wrong arity for operator
	*/
	runTest("Malformed expression", "(+ 3.14 (- 15 92")
	runTest("Malformed expression", "(+ 3.14 (- 15 92)")
	//runTest("Malformed expression", "(+ 3.14 (- 15 92)))")
	//runTest("Malformed expression", "(+ 3.14 15))")
	runTest("Invalid terminal", "+ 3.14 .0015")
	runTest("Invalid terminal", "hello")
	runTest("Invalid terminal", "3.IA")
	runTest("Invalid terminal", "x2")
	runTest("Unknown terminal: x2", "(+ 3.14 x2)")
	runTest("Invalid functional: %", "(+ 3.14 (% 15 92))")
	runTest("Wrong arity: expected 2 children, got 3", "(+ 3.14 15 92)")
	runTest("Wrong arity: expected 2 children, got 1", "(+ 3.14)")
}

func testRead(t *testing.T) {
	read_tree("3.14")
	read_tree("  .1415  ")
	read_tree("31.415e-1")
	read_tree("x0")
	read_tree("(+ 3.14 (- 15 92))")
	read_tree(" ( +  x0  x1 ) ")
	read_tree("(+ 0 (- x0 92))")
	// Extra trailing characters can be ignored
	read_tree("(+ 0 (- x0 92)))))")
	read_tree("(+ 0 (- x0 92))foobar")
}

func testWriteAndRead(t *testing.T) {
	// Try some randomly generated trees
	for i := 0; i < 10; i++ {
		tree := create_full_tree(0, nil, 0)
		update_terminal_symbols(i % len(set))
		repr := write_tree(tree)
		read_tree(repr)
	}

	// Try closed loop
	testStrings := []string{
		"3.14",
		".1415",
		"31.415e-1",
		"x0",
		"(+ 3.14 (- 15 92))",
		"(+ x0 x1)",
		"(+ 0 (- x0 92))",
	}
	for _, s := range testStrings {
		//println("Reading tree", s)
		tree := read_tree(s)
		//println("Processing ", s, " produced tree", tree)
		repr := write_tree(tree)
		if repr != s {
			t.Error("Error when converting expression: ", s)
		}
	}
}

func TestTreeToString(t *testing.T) {
	set = []Instance{
		{[]float64{1.1, 2.2}, 3.3},
		{[]float64{4.4, 5.5}, 9.9},
		{[]float64{1.2, 2.4}, 3.6},
		{[]float64{4.1, 2.6}, 6.7},
	}

	nvar, nvar_test = 2, 2
	nrow, nrow_test = 2, 2

	create_T_F()

	t.Run("Malformed", testMalformed)
	t.Run("Read", testRead)
	t.Run("WriteAndRead", testWriteAndRead)
}
