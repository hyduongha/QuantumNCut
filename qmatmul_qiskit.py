import numpy as np
import qiskit
from qiskit_aer import Aer
from qiskit_aer.primitives import Sampler, Estimator
from qiskit_aer import AerSimulator



def qmatmul_qiskit(origin_L, origin_v):
    size = origin_v.shape[0]
    res = np.zeros(size)
    norm_L =  np.linalg.norm(origin_L, axis=1, keepdims=True)
    norm_v = np.linalg.norm(origin_v)
    L = origin_L / norm_L
    V = origin_v / norm_v
    num_qubits = int(np.log2(len(origin_v)))
    U_V = qiskit.QuantumCircuit(num_qubits)
    U_V.prepare_state(origin_v / norm_v, range(num_qubits))
    for i in range(size):
        qc = qiskit.QuantumCircuit(1 + 2 * num_qubits, 1)
        U_L = qiskit.QuantumCircuit(num_qubits)
        U_L.prepare_state(L[i, :], range(num_qubits))

        qc.h(0)
        qc.append(U_L, range(1, num_qubits + 1))
        qc.append(U_V, range(num_qubits + 1, 2*num_qubits + 1))
        for j in range(num_qubits):
            qc.cswap(0, j + 1, j + num_qubits + 1)
        qc.h(0)
        qc.measure(0, 0)
        
        simulator = AerSimulator()
        result = simulator.run(qc, shots=1024).result()
        counts = result.get_counts(qc)
        res[i] = (norm_v * norm_L[i] * (np.sqrt(counts.get('0',0) - counts.get('1', 0)))/(np.sqrt(counts.get('0',0) + counts.get('1', 0))))
    return res

# results_df = None
# num_repeats = 100
# import pandas as pd
# for size in [128, 256, 512, 1024]:
# 	print(f"Size: {size}")
# 	mses = []
# 	for _ in range(num_repeats):
# 		origin_L = np.random.uniform(1, 4, (size, size))
# 		origin_v = np.random.uniform(1, 2, size)
# 		res = qmatmul_qiskit(origin_L, origin_v)
# 		mse = (np.square(res - np.dot(origin_L, origin_v))).mean(axis=0)
# 		mses.append(mse)
# 	print(f"Mean MSE: {np.mean(mses):.4f}, Std MSE: {np.std(mses):.4f}")
# 	record = {'size': size, 'mean_mse': np.mean(mses), 'std_mse': np.std(mses)}
# 	if 'results_df' not in locals():
# 		results_df = pd.DataFrame(columns=record.keys())
# 	results_df = pd.concat([results_df, pd.DataFrame([record])], ignore_index=True)
# 	results_df.to_csv(f'benchmark_acc3.csv', index=False)