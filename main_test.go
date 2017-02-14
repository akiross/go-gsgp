package main

import "testing"

func testMalformed(t *testing.T) {
	runTest := func(errorMessage, malformedString string) {
		defer func() {
			err := recover()
			if err == nil {
				t.Error("read_tree did not panic for", errorMessage, "in", malformedString)
			} else {
				switch e := err.(type) {
				case string:
					if e != errorMessage {
						t.Error("read_tree panicked with wrong message: ", err.(string), "instead of", errorMessage, "for", malformedString)
					}
				case error:
					if e.Error() != errorMessage {
						t.Error("read_tree panicked with wrong message: ", e.Error(), "instead of", errorMessage, "for", malformedString)
					}
				default:
					t.Error("unknown error type", e)
				}
			}
		}()

		read_tree(malformedString)
	}
	// Bad things that could happen:
	// - parentheses not opened (extra trailing chars are tolerated)
	// - token not a valid float
	// - unknown variable
	// - unknown operation
	// - wrong arity for operator
	runTest("runtime error: index out of range", "(")
	runTest("runtime error: index out of range", "  (  ")
	runTest("Malformed expression", "(+")
	runTest("Malformed expression", "(+ 3.14")
	runTest("runtime error: index out of range", "(+ 3.14 ")
	runTest("runtime error: index out of range", "(+ 3.14 (")
	runTest("Malformed expression", "(+ 3.14 (-")
	runTest("Malformed expression", "(+ 3.14 (- 15")
	runTest("runtime error: index out of range", "(+ 3.14 (- 15 92")
	runTest("runtime error: index out of range", "(+ 3.14 (- 15 92)")
	runTest("Invalid terminal: ", "(+ 3.14)")

	//runTest("Malformed expression", "(+ 3.14 (- 15 92)))")
	//runTest("Malformed expression", "(+ 3.14 15))")
	runTest("Invalid terminal: +", "+ 3.14 .0015")
	runTest("Invalid terminal: hello", "hello")
	runTest("Invalid terminal: 3.IA", "3.IA")
	runTest("Invalid terminal: x2", "x2")
	runTest("Invalid terminal: x2", "(+ 3.14 x2)")
	runTest("Invalid functional: %", "(+ 3.14 (% 15 92))")
	runTest("Unexpected character: 9", "(+ 3.14 15 92)")
	runTest("Invalid terminal: ", "(+ 3.14)")
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
		"(+ 0 (- (* x0 x0) 92))",
		"(- (/ 10 32) (* x1 (+ x0 0.5)))",
		"(- (/ (+ 1 9) 32) (* x1 (+ x0 0.5)))",
		"(- (/ (+ (* x0 x0) (* x1 (* x1 x1))) 32) (* x0 (+ x1 0.5)))",
	}
	for _, s := range testStrings {
		println("--------------- Reading tree", s)
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
	//t.Run("Read", testRead)
	//t.Run("WriteAndRead", testWriteAndRead)
}
