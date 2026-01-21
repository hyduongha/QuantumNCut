from memqoop import memcompiler
import numpy as np
import pandas as pd
import os
num_qubits = 5
num_repeats = 25


for k in range(2, 20):
	cost_with_mems = []
	cost_without_mems = []
	i_with_mems = []
	i_without_mems = []
	mem_compilation = memcompiler.MemCompilation(num_qubits)
	mem_compilation.gen_vecs(k)
	mem_compilation.compile_vecs()
		
	for i in range(num_repeats):
		print(f"num_qubits: {num_qubits}, k: {k}, i: {i}")

		v = np.random.randn(2**num_qubits)
		v /= np.linalg.norm(v)
		costs, thetass, i = mem_compilation.compile_v_with_mem(v)
		cost_with_mems.append(costs[-1])
		i_with_mems.append(i)
		
		costs, thetass, i = mem_compilation.compile_v_without_mem(v)
		cost_without_mems.append(costs[-1])
		i_without_mems.append(i)
	records = {
		"num_qubits": num_qubits,
		"k": k,
		"cost_with_mems": np.mean(cost_with_mems),
		"cost_without_mems": np.mean(cost_without_mems),
		"i_with_mems": np.mean(i_with_mems),
		"i_without_mems": np.mean(i_without_mems)
	}
	file_name = "records5.csv"
	if os.path.exists(file_name):
		pd.DataFrame([records]).to_csv(file_name, mode='a', header=False, index=False)
	else:
		pd.DataFrame([records]).to_csv(file_name, index=False)