// Selects the value from the dataset of variable varId at given row
extern "C" __device__
double getDataset(int varId, int row, double *set) {
	return set[row * (NUM_VARIABLE_SYMBOLS + 1) + varId];
}

extern "C" __device__
double exp64(double v) {
	double r = exp(v);
	if (isinf(r)) {
		return 1.7976931348623158e+308;
	}
	return r;
}


// Given symbols, dataset and a tree, evaluates the tree for a given row in the dataset
extern "C" __device__
double eval_arrays(double *sym, double *set, int *tree, int start, int i) {
	int id = tree[start];
	if (id < NUM_FUNCTIONAL_SYMBOLS) {
		switch (id) {
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
		return getDataset(id - NUM_FUNCTIONAL_SYMBOLS, i, set);
	}
	//if (id >= NUM_FUNCTIONAL_SYMBOLS+NUM_VARIABLE_SYMBOLS) {
	return sym[id];
	//}
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

// Evaluates array for all the rows in the dataset
// number of total threads required: NROWS_TOT
extern "C" __global__
void semantic_eval_arrays(double *sym, double *set, int *tree, double *out_sem_tot) { //train, double *out_sem_test) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < NROWS_TOT) {
		double sem_val = eval_arrays(sym, set, tree, 0, i);
		out_sem_tot[i] = sem_val;
		/*
		if (i < NROWS_TRAIN)
			out_sem_train[i] = sem_val;
		else
			out_sem_test[i-NROWS_TRAIN] = sem_val;
		*/
	}
}


extern "C" __global__
void test_kernel(double *out_sem_train, double *out_sem_test) {
	int tig = blockIdx.x * blockDim.x + threadIdx.x;
	if (tig < NROWS_TOT) {
		if (tig < NROWS_TRAIN)
			out_sem_train[tig] = tig;
		else
			out_sem_test[tig-NROWS_TRAIN] = tig;
	}
}

// Executed once for NROWS_TOT
extern "C" __global__
void sem_crossover(
		double *sem_tot,
		double *p1_train_sem, double *p1_test_sem,
		double *p2_train_sem, double *p2_test_sem,
		double *out_train_sem, double *out_test_sem
	) {
	int tig = blockIdx.x * blockDim.x + threadIdx.x;

	if (tig < NROWS_TRAIN) {
		int j = tig;
		double sigm = 1.0 / (1.0 + exp64(-sem_tot[tig]));
		out_train_sem[j] = p1_train_sem[j] * sigm + p2_train_sem[j] * (1-sigm);
	} else if (tig < NROWS_TOT) {
		int j = tig - NROWS_TRAIN;
		double sigm = 1.0 / (1.0 + exp64(-sem_tot[tig]));
		out_test_sem[j] = p1_test_sem[j] * sigm + p2_test_sem[j] * (1-sigm);
	}
}

extern "C" __device__
double square_diff(double a, double b) { return (a - b) * (a - b); }

extern "C" __device__
double abs_diff(double a, double b) { return abs(a - b); }

extern "C" __device__
double rel_abs_diff(double a, double b) { return abs (a - b) / a; }

extern "C" __global__
void sem_copy(double *dest_train, double *src_train, double *dest_test, double *src_test) {
	int tig = blockIdx.x * blockDim.x + threadIdx.x;
	if (tig < NROWS_TRAIN) {
		dest_train[tig] = src_train[tig];
	} else if (tig < NROWS_TOT) {
		dest_test[tig-NROWS_TRAIN] = src_test[tig-NROWS_TRAIN];
	}
}

// Runs with only 2 threads, for this unoptimized version
extern "C" __global__
void sem_fitness_train(double *set, double *sem_train, double *out_fit_train, double *out_ls_a, double *out_ls_b) {
	int tig = blockIdx.x * blockDim.x + threadIdx.x;
	if (tig == 0) {
		double sum_out = 0; // Maybe initialize it to first value and start for loop with 1, faster?
		double sum_tar = 0;
		double sum_oxo = 0;
		double sum_oxt = 0;
		for (int i = 0; i < NROWS_TRAIN; i++) {
			double t = getDataset(NUM_VARIABLE_SYMBOLS, i, set);
			double y = sem_train[i];
			sum_out += y;
			sum_tar += t;
			sum_oxo += y * y;
			sum_oxt += y * t;
		}

		double avg_out = sum_out / NROWS_TRAIN;
		double avg_tar = sum_tar / NROWS_TRAIN;

		double num = sum_oxt - sum_tar * avg_out - sum_out * avg_tar + NROWS_TRAIN * avg_out * avg_tar; 
		double den = sum_oxo - 2.0 * sum_out * avg_out + NROWS_TRAIN * avg_out * avg_out; 

		double b = num / den;
		double a = avg_tar - b * avg_out;
		*out_ls_b = b;
		*out_ls_a = a;

		double d = 0; // Same as above
		for (int i = 0; i < NROWS_TRAIN; i++) {
			double yy = getDataset(NUM_VARIABLE_SYMBOLS, i, set);
			d += ERROR_FUNC(yy, a + b * sem_train[i]);
		}
		out_fit_train[0] = d / double(NROWS_TRAIN);
	}
}

extern "C" __global__
void sem_fitness_test(double *set, double *sem_test, double *ls_a, double *ls_b, double *out_fit_test) {
	int tig = blockIdx.x * blockDim.x + threadIdx.x;
	if (tig == 0) {
		double d = 0;
		double a = *ls_a;
		double b = *ls_b;
		for (int i = 0; i < NROWS_TEST; i++) {
			double yy = getDataset(NUM_VARIABLE_SYMBOLS, i+NROWS_TRAIN, set);
			d += ERROR_FUNC(yy, a + b * sem_test[i]);
		}
		out_fit_test[1] = d / double(NROWS_TEST);
	}
}

extern "C" __global__
void sem_fitness(double *set, double *sem_train, double *sem_test, double *out_fit) {
	int tig = blockIdx.x * blockDim.x + threadIdx.x;
	if (tig == 0) {
		double d = 0;
		for (int i = 0; i < NROWS_TRAIN; i++) {
			double yy = getDataset(NUM_VARIABLE_SYMBOLS, i, set);
			d += ERROR_FUNC(yy, sem_train[i]);
		}
		out_fit[0] = d / double(NROWS_TRAIN);
	} else if (tig == 1) {
		double d = 0;
		for (int i = 0; i < NROWS_TEST; i++) {
			double yy = getDataset(NUM_VARIABLE_SYMBOLS, i+NROWS_TRAIN, set);
			d += ERROR_FUNC(yy, sem_test[i]);
		}
		out_fit[1] = d / double(NROWS_TEST);
	}
}

extern "C" __global__
void sem_mutation(double *sem_tot1, double *sem_tot2, double *mut_step, double *out_train_sem, double *out_test_sem) {
	int tig = blockIdx.x * blockDim.x + threadIdx.x;

	if (tig < NROWS_TRAIN) {
		double sigm1 = 1.0 / (1.0 + exp64(-sem_tot1[tig]));
		double sigm2 = 1.0 / (1.0 + exp64(-sem_tot2[tig]));
		out_train_sem[tig] += *mut_step * (sigm1 - sigm2);
	} else if (tig < NROWS_TOT) {
		double sigm1 = 1.0 / (1.0 + exp64(-sem_tot1[tig]));
		double sigm2 = 1.0 / (1.0 + exp64(-sem_tot2[tig]));
		out_test_sem[tig-NROWS_TRAIN] += *mut_step * (sigm1 - sigm2);
	}
}
