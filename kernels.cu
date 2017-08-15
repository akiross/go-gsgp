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
	return sym[id];
}

// Evaluates array for all the rows in the dataset
// number of total threads required: NROWS_TOT
extern "C" __global__
void semantic_eval_arrays(double *sym, double *set, int *tree, double *out_sem_tot) { //train, double *out_sem_test) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < NROWS_TOT) {
		double sem_val = eval_arrays(sym, set, tree, 0, i);
		out_sem_tot[i] = sem_val;
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

extern "C" __global__
void sem_copy_split(double *dest_train, double *dest_test, double *src_tot) {
	int tig = blockIdx.x * blockDim.x + threadIdx.x;
	if (tig < NROWS_TRAIN) {
		dest_train[tig] = src_tot[tig];
	} else if (tig < NROWS_TOT) {
		dest_test[tig-NROWS_TRAIN] = src_tot[tig];
	}
}

extern "C" __device__
double pass_value(double v) {
	return v;
}

extern "C" __device__
double square_root(double v) {
	return sqrt(v);
}

extern "C" __global__
void sem_fitness_train_nls(double *set, double *sem_train, double *out_fit_train, double *out_ls_a, double *out_ls_b) {
	__shared__ double shm[32]; // Partial fitness sum
	int tig = blockIdx.x * blockDim.x + threadIdx.x;
	int block_size = (NROWS_TRAIN + 32 - 1) / 32;

	if (tig < 32) {
		int start = block_size * tig;
		int end = block_size * (tig + 1);
		if (end > NROWS_TRAIN)
			end = NROWS_TRAIN;
		shm[tig] = 0;
		for (int i = start; i < end; i++) {
			double yy = getDataset(NUM_VARIABLE_SYMBOLS, i, set);
			shm[tig] += ERROR_FUNC(yy, sem_train[i]);
		}
	}
	__syncthreads();

	if (tig == 0) {
		for (int i = 1; i < 32; i++) {
			shm[0] += shm[i];
		}
		out_fit_train[0] = POST_ERR_FUNC(shm[0] / double(NROWS_TRAIN));
	}
}

// Runs with 32 threads
extern "C" __global__
void sem_fitness_test_nls(double *set, double *sem_test, double *ls_a, double *ls_b, double *out_fit_test) {
	__shared__ double shm[32]; // Warp size shared memory
	int tig = blockIdx.x * blockDim.x + threadIdx.x;
	int block_size = (NROWS_TEST + 32 - 1) / 32;

	if (tig < 32) {
		int start = block_size * tig;
		int end = block_size * (tig + 1);
		if (end > NROWS_TEST)
			end = NROWS_TEST;
		shm[tig] = 0;
		for (int i = start; i < end; i++) {
			double yy = getDataset(NUM_VARIABLE_SYMBOLS, i+NROWS_TRAIN, set);
			shm[tig] += ERROR_FUNC(yy, sem_test[i]);
		}
	}
	__syncthreads();

	if (tig == 0) {
		for (int i = 1; i < 32; i++) {
			shm[0] += shm[i];
		}
		out_fit_test[1] = POST_ERR_FUNC(shm[0] / double(NROWS_TEST));
	}
}

// Runs with 32 threads
extern "C" __global__
void sem_fitness_train_ls(double *set, double *sem_train, double *out_fit_train, double *out_ls_a, double *out_ls_b) {
	// Warp size shared memory
	__shared__ double sh_out[32]; // Sum of out semantics
	__shared__ double sh_tar[32]; // Sum of target semantics
	__shared__ double sh_oxo[32]; // Sum of squared out semantics
	__shared__ double sh_oxt[32]; // Sum of out times target semantics

	int tig = blockIdx.x * blockDim.x + threadIdx.x;
	int block_size = (NROWS_TRAIN + 32 - 1) / 32;

	if (tig < 32) {
		int start = block_size * tig;
		int end = block_size * (tig + 1);
		if (end > NROWS_TRAIN)
			end = NROWS_TRAIN;

		sh_out[tig] = 0;
		sh_tar[tig] = 0;
		sh_oxo[tig] = 0;
		sh_oxt[tig] = 0;

		for (int i = start; i < end; i++) {
			double t = getDataset(NUM_VARIABLE_SYMBOLS, i, set);
			double y = sem_train[i];
			sh_out[tig] += y;
			sh_tar[tig] += t;
			sh_oxo[tig] += y * y;
			sh_oxt[tig] += y * t;
		}
	}

	// Wait for all the threads to finish
	__syncthreads();

	__shared__ double a;
	__shared__ double b;

	if (tig == 0) {
		double sum_out = sh_out[0];
		double sum_tar = sh_tar[0];
		double sum_oxo = sh_oxo[0];
		double sum_oxt = sh_oxt[0];

		for (int i = 1; i < 32; i++) {
			sum_out += sh_out[i];
			sum_tar += sh_tar[i];
			sum_oxo += sh_oxo[i];
			sum_oxt += sh_oxt[i];
		}
	
		double avg_out = sum_out / NROWS_TRAIN;
		double avg_tar = sum_tar / NROWS_TRAIN;

		double num = sum_oxt - sum_tar * avg_out - sum_out * avg_tar + NROWS_TRAIN * avg_out * avg_tar; 
		double den = sum_oxo - 2.0 * sum_out * avg_out + NROWS_TRAIN * avg_out * avg_out; 

		if (den != 0) {
			b = num / den;
		} else {
			b = 0;
		}

		a = avg_tar - b * avg_out;

		*out_ls_b = b;
		*out_ls_a = a;
	}

	__syncthreads();

	__shared__ double shm[32]; // Partial fitness sum

	if (tig < 32) {
		int start = block_size * tig;
		int end = block_size * (tig + 1);
		if (end > NROWS_TRAIN)
			end = NROWS_TRAIN;
		shm[tig] = 0;
		for (int i = start; i < end; i++) {
			double yy = getDataset(NUM_VARIABLE_SYMBOLS, i, set);
			shm[tig] += ERROR_FUNC(yy, a + b * sem_train[i]);
		}
	}
	__syncthreads();

	if (tig == 0) {
		for (int i = 1; i < 32; i++) {
			shm[0] += shm[i];
		}
		out_fit_train[0] = POST_ERR_FUNC(shm[0] / double(NROWS_TRAIN));
	}
}

// Runs with 32 threads
extern "C" __global__
void sem_fitness_test_ls(double *set, double *sem_test, double *ls_a, double *ls_b, double *out_fit_test) {
	__shared__ double shm[32]; // Warp size shared memory
	int tig = blockIdx.x * blockDim.x + threadIdx.x;
	int block_size = (NROWS_TEST + 32 - 1) / 32;

	if (tig < 32) {
		double a = *ls_a;
		double b = *ls_b;
		int start = block_size * tig;
		int end = block_size * (tig + 1);
		if (end > NROWS_TEST)
			end = NROWS_TEST;
		shm[tig] = 0;
		for (int i = start; i < end; i++) {
			double yy = getDataset(NUM_VARIABLE_SYMBOLS, i+NROWS_TRAIN, set);
			shm[tig] += ERROR_FUNC(yy, a + b * sem_test[i]);
		}
	}
	__syncthreads();

	if (tig == 0) {
		for (int i = 1; i < 32; i++) {
			shm[0] += shm[i];
		}
		out_fit_test[1] = POST_ERR_FUNC(shm[0] / double(NROWS_TEST));
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
