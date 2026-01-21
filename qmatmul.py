import qiskit
from qoop.core.ansatz import WchainCNOT_xyz
from qoop.compilation.qsp import QuantumStatePreparation
import numpy as np
from qiskit_aer import Aer
from qiskit_aer.primitives import Sampler, Estimator
from qiskit_aer import AerSimulator
def qmatmul_qiskit(origin_L, origin_v):

    origin_L = origin_L.astype(np.float64)
    origin_v = origin_v.astype(np.float64)
    size = origin_v.shape[0]
    res = np.zeros(size)
    norm_L =  np.linalg.norm(origin_L, axis=1, keepdims=True)
    norm_v = np.linalg.norm(origin_v)
    L = origin_L / norm_L
    V = origin_v / norm_v
    V[np.abs(V) < 1e-10] = 0
    V = V / np.linalg.norm(V)
    num_qubits = int(np.log2(len(origin_v)))
    U_V = qiskit.QuantumCircuit(num_qubits)

    U_V.prepare_state(V, range(num_qubits))
    for i in range(size):
        qc = qiskit.QuantumCircuit(1 + 2 * num_qubits, 1)
        U_L = qiskit.QuantumCircuit(num_qubits)
        state = L[i, :]
        state = state / np.linalg.norm(state)
        state[np.abs(state) < 1e-10] = 0   
        state = state / np.linalg.norm(state) 

        U_L.prepare_state(state, range(num_qubits))

        qc.h(0)
        qc.append(U_L, range(1, num_qubits + 1))
        qc.append(U_V, range(num_qubits + 1, 2*num_qubits + 1))
        for j in range(num_qubits):
            qc.cswap(0, j + 1, j + num_qubits + 1)
        qc.h(0)
        qc.measure(0, 0)
        

        # simulator = AerSimulator()
        # result = simulator.run(qc, shots=10000).result()
        # counts = result.get_counts(qc)        
        sampler = Sampler()
        counts = sampler.run(qc, shots = 1000).result().quasi_dists[0]
        if counts.get(0,0) < 0.5:
            res[i] = 0.0
        else:
            res[i] = (norm_v * norm_L[i] * (np.sqrt(2*counts.get(0,0) - 1)))
        # print(f'_______{i}_________')
        # print(counts.get(0,0), counts.get(1, 0))
        # print(norm_v)
        # print(norm_L[i])
        # print(res[i])
    return res


def prepare_state(v):
    num_qubits = int(np.log2(len(v)))
    qsp = QuantumStatePreparation(
        u = WchainCNOT_xyz(num_qubits, num_qubits), 
        target_state = v).fit()
    return qsp.compiler.thetass[-1]

def qmatmul(origin_L, origin_v, num_layers = 1):
    size = origin_v.shape[0]
    res = np.zeros(size)
    norm_L =  np.linalg.norm(origin_L, axis=1, keepdims=True)
    norm_v = np.linalg.norm(origin_v)
    L = origin_L / norm_L
    v = origin_v / norm_v
    num_qubits = int(np.log2(len(origin_v)))
    US = [0]*size
    V = prepare_state(v)
    for i in range(size):
        US[i] = prepare_state(L[i, :])
        sub_qc = WchainCNOT_xyz(num_qubits, num_layers)
        qc = qiskit.QuantumCircuit(1 + 2 * num_qubits, 1)
        qc.h(0)
        qc.append(sub_qc.assign_parameters(US[i]), range(1, num_qubits + 1))
        qc.append(sub_qc.assign_parameters(V), range(num_qubits + 1, 2*num_qubits + 1))
        for j in range(num_qubits):
            qc.cswap(0, j + 1, j + num_qubits + 1)
        qc.h(0)
        qc.measure(0, 0)
        simulator = Aer.get_backend('aer_simulator_statevector', device = 'GPU')
        qc = qiskit.transpile(qc, backend = simulator)
        result = simulator.run(qc).result()
        counts = result.get_counts(qc)

        res[i] = (norm_v * norm_L[i] * (np.sqrt(counts.get('0',0) - counts.get('1', 0)))/(np.sqrt(counts.get('0',0) + counts.get('1', 0))))
    return res


# num_repeats = 20
# import pandas as pd
# for size in [32]:
#     for num_layers in range(1, int(np.log2(size)) + 1):
#         print(f"Size: {size}, Layers: {num_layers}")
#         mses = []
#         for _ in range(num_repeats):
#             origin_L = np.random.uniform(1, 4, (size, size))
#             origin_v = np.random.uniform(1, 2, size)
#             res = qmatmul(origin_L, origin_v, num_layers)
#             mse = (np.square(res - np.dot(origin_L, origin_v))).mean(axis=0)
#             mses.append(mse)
#         print(f"Mean MSE: {np.mean(mses):.4f}, Std MSE: {np.std(mses):.4f}")
#         record = {'size': size, 'num_layers': num_layers, 'mean_mse': np.mean(mses), 'std_mse': np.std(mses)}
#         if 'results_df' not in locals():
#             results_df = pd.DataFrame(columns=record.keys())
#         results_df = pd.concat([results_df, pd.DataFrame([record])], ignore_index=True)
#         results_df.to_csv(f'benchmark_acc3.csv', index=False)