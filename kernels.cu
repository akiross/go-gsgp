extern "C" __device__
double get_var(double *set, int var, int row) {
	return set[row * (NUM_VARIABLE_SYMBOLS + 1) + var];
}

extern "C" __device__
double eval_arrays(double *sym, double *set, int *tree, int start, int i) {
	int id = tree[start];
	if (id < NUM_FUNCTIONAL_SYMBOLS) {
		switch (id) {
		// These IDs are hard-coded as in create_T_F()
			case 0: {
				double v1 = eval_arrays(sym, set, tree, tree[start+1], i);
				double v2 = eval_arrays(sym, set, tree, tree[start+2], i);
				return v1 + v2;
			}
			case 1: {
				double v1 = eval_arrays(sym, set, tree, tree[start+1], i);
				double v2 = eval_arrays(sym, set, tree, tree[start+2], i);
				return v1 - v2;
			}
			case 2: {
				double v1 = eval_arrays(sym, set, tree, tree[start+1], i);
				double v2 = eval_arrays(sym, set, tree, tree[start+2], i);
				return v1 * v2;
			}
			case 3: {
				double v1 = eval_arrays(sym, set, tree, tree[start+1], i);
				double v2 = eval_arrays(sym, set, tree, tree[start+2], i);
				if (v2 == 0)
					return 1;
				else
					return v1 / v2;
			}
			default: {
				double v = eval_arrays(sym, set, tree, tree[start+1], i);
				if (v < 0)
					return sqrt(-v);
				else
					return sqrt(v);
			}
		}
	}
	if (id >= NUM_FUNCTIONAL_SYMBOLS && id < NUM_FUNCTIONAL_SYMBOLS + NUM_VARIABLE_SYMBOLS) {
		return get_var(set, id - NUM_FUNCTIONAL_SYMBOLS, i);
	}
	if (id >= NUM_FUNCTIONAL_SYMBOLS+NUM_VARIABLE_SYMBOLS) {
		return sym[id];
	}
}

extern "C" __global__
void reduce(double *in_data, double *out_data) {
	__shared__ double shm[NUM_THREADS];              // Shared memory where to accumulate data

	int tib = threadIdx.x;                           // ID of thread in its block
	int tig = blockIdx.x * blockDim.x + threadIdx.x; // ID of thread globally

	shm[tib] = in_data[tig]; // Copy data from global to local memory
	__syncthreads();

	if (threadIdx.x == 0) {
		for (int i = 1; i < blockDim.x; i++) {
			shm[0] += shm[i];
		}
	}
	if (threadIdx.x == 0) {
		out_data[blockIdx.x] = shm[0];
	}
}

extern "C" __global__
void semantic_eval_arrays(double *sym, double *set, int *tree, double *outs) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < NROWS) {
		outs[i] = eval_arrays(sym, set, tree, 0, i);
	}
}
