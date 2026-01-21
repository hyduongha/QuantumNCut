import numpy as np
from memqoop import compilation

def generate_points_on_sphere(d, k):
    vecs = np.random.randn(k, d)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs

class MemCompilation:
    def __init__(self, num_qubits):
        self.num_qubits = num_qubits
        self.vecs = None
        self.vecs_costs = []
        self.vecs_thetass = []
        self.vecs_i = []
        self.v = None
    def gen_vecs(self, k):
        self.vecs = generate_points_on_sphere(2**self.num_qubits, k)
    def compile_vecs(self):
        for vec in self.vecs:
            
            costs, thetass, i = compilation.compilation(
				state = vec, 
				thetas = None,
				steps = 200,
				num_layers = self.num_qubits
			)
            
            self.vecs_costs.append(costs)
            self.vecs_thetass.append(thetass)
            self.vecs_i.append(i)
            
        return
    def compile_v_with_mem(self, v):
        if self.vecs is None or self.vecs_thetass is None:
            raise ValueError("Vectors have not been generated/compiled. Call gen_vecs() and compile_vecs() first.")
        self.v = v
        distances = np.linalg.norm(self.vecs - v, axis=1)
        closest_vector_index = np.argmin(distances)
        costs, thetass, i = compilation.compilation(
			state = v, 
			thetas = self.vecs_thetass[closest_vector_index][-1],
			steps = 200,
			num_layers = self.num_qubits,
		)
        return np.array(costs), np.array(thetass), i

    def compile_v_without_mem(self, v):
        self.v = v
        costs, thetass, i = compilation.compilation(
			state = v, 
			thetas = None,
			steps = 200,
			num_layers = self.num_qubits,
		)
        return np.array(costs), np.array(thetass), i
