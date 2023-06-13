
from pyqpanda import SWAP,QCircuit,QProg,prob_run_list,X,PauliOperator,H,RZ,RX,RY,CNOT,var,init_quantum_machine,QMachineType
import scipy
import time
import os
import numpy as np
from functools import partial
from pychemiq import Molecules,ChemiQ
from pychemiq.Transform.Mapping import jordan_wigner, bravyi_kitaev, parity, segment_parity, MappingType
from scipy.optimize import minimize
from Hamiltonian import Hamiltonian_data
import ast
import datetime
import argparse
    

class QpandaBackend:
    def __init__(self, machile, qlist, parameter, index, sparse_H, num_qubits=12):
        self.num_qubits = num_qubits
        self.machine = machile
        self.qlist = qlist
        self.parameter = parameter
        self.index = index
        self.ham_sparse_matrix = sparse_H
        
    def statevector_from_ansatz(self, var_parameters):
        vqc = QCircuit()
        layers = len(var_parameters)
        # 构造量子线路
        for i in range(6):
            vqc.insert(X(qlist[i]))

        # 构造量子线路
        for layer in range(layers):
            # print(layer)
            i,j,k,l = self.index[layer].tolist()
            if k != 12:
                vqc.insert(oneCircuit(qlist, i,j,k,l, var_parameters[layer])) 
            else:
                vqc.insert(single_Circuit(qlist, i,j, var_parameters[layer]))
                
        vqc.insert(SWAP(qlist[0], qlist[11]))
        vqc.insert(SWAP(qlist[1], qlist[10]))
        vqc.insert(SWAP(qlist[2], qlist[9]))
        vqc.insert(SWAP(qlist[3], qlist[8]))
        vqc.insert(SWAP(qlist[4], qlist[7]))
        vqc.insert(SWAP(qlist[5], qlist[6]))

        # 构建量子线路实例
        prog = QProg()
        prog.insert(vqc)

        result = prob_run_list(prog, qlist, -1)
        statevector = np.array(machine.get_qstate())
        return statevector

    # def ham_sparse_matrix(self):
    #     H_sparse_matrix = scipy.sparse.load_npz("my_Hamiltonian.npz")
    #     return  H_sparse_matrix

    def ham_expectation_value(self, var_parameters, init_state_qasm=None, cache=None, excited_state=0):

        statevector = self.statevector_from_ansatz(var_parameters)
        sparse_statevector = scipy.sparse.csr_matrix(statevector)
        H_sparse_matrix = self.ham_sparse_matrix

        expectation_value = \
            sparse_statevector.dot(H_sparse_matrix).dot(sparse_statevector.conj().transpose()).todense()[0, 0]

        return expectation_value.real

class VQERunner:
    # Works for a single geometry
    def __init__(self, backend, optimizer='L-BFGS-B',
                 optimizer_options={'gtol': 1e-08, 'maxiter': 3000}, print_var_parameters=False, hf_energy=-2.806471946359929, use_ansatz_gradient=False):

        self.backend = backend
        self.optimizer = optimizer
        self.print_var_parameters = print_var_parameters
        self.use_ansatz_gradient = use_ansatz_gradient
        self.optimizer_options = optimizer_options

        self.previous_energy = hf_energy
        self.hf_energy = hf_energy
        self.new_energy = None
        

        self.iteration = 0
        self.time_previous_iter = 0

    def get_energy_no_print(self, var_parameters):
        energy = self.backend.ham_expectation_value(var_parameters)
        return energy
    
    def callback(self, X):
        energy = backend.ham_expectation_value(X)
        if energy - self.hf_energy <= 1.6e-3:
            return True

    def get_energy(self, var_parameters, backend, multithread=False, multithread_iteration=None,
                    init_state_qasm=None, cache=None, excited_state=0):

        if multithread is False:
            iteration_duration = time.time() - self.time_previous_iter
            self.time_previous_iter = time.time()

        energy = backend.ham_expectation_value(var_parameters, cache=cache,
                                               init_state_qasm=init_state_qasm, excited_state=excited_state)

        self.new_energy = energy
        delta_e = self.new_energy - self.previous_energy
        self.previous_energy = self.new_energy

        message = 'Iteration: {}. Energy {}.  Energy change {} , Iteration dutation: {}' \
            .format(self.iteration, self.new_energy, '{:.3e}'.format(delta_e), iteration_duration)
        if True:
            # message += ' Params: ' + str(var_parameters)
            print(message, file=log)
            log.flush()

        self.iteration += 1

        return energy
    
    def vqe_run(self, init_guess_parameters=None, init_state_qasm=None, excited_state=0, cache=None):

        # assert len(ansatz) > 0
        # if init_guess_parameters is None:
        #     var_parameters = np.zeros(sum([element.n_var_parameters for element in ansatz]))
        # else:
        #     assert len(init_guess_parameters) == sum([element.n_var_parameters for element in ansatz])
        var_parameters = init_guess_parameters

        self.iteration = 1
        self.time_previous_iter = time.time()

        # functions to be called by the optimizer
        get_energy = partial(self.get_energy, backend=self.backend, init_state_qasm=init_state_qasm,
                             excited_state=excited_state, cache=cache)

        # get_gradient = partial(self.backend.ansatz_gradient, q_system=self.q_system,
        #                        init_state_qasm=init_state_qasm, cache=cache, excited_state=excited_state)

        # if self.use_ansatz_gradient:
        #     result = scipy.optimize.minimize(get_energy, var_parameters, jac=get_gradient, method=self.optimizer,
        #                                      options=self.optimizer_options, tol=config.optimizer_tol,
        #                                      bounds=config.optimizer_bounds)
        # else:

        result = scipy.optimize.minimize(get_energy, var_parameters, method=self.optimizer,
                                         options=self.optimizer_options, tol=1e-10,
                                         bounds=None)

        result['n_iters'] = self.iteration  # cheating

        return result

def get_Pauli(pauliOp):
    num_qubits = 12
    pauliOp = pauliOp.split()
    num = len(pauliOp)
    pauli_list = ['I'] * (num_qubits)
    if num > 0:
        for item in pauliOp:
            pauli, index = item[0], int(item[1:])
            pauli_list[index] = pauli
    pauli_list = ''.join(pauli_list)

    return pauli_list

def Hamiltonian_from_chemiq():
    # 计算所给问题对应的哈密尔顿量
    multiplicity = 1
    charge = 0
    basis =  "sto-3g"
    geom = "H 1.300000 2.250000 0.000000, \
            H 3.900000 2.250000 0.000000, \
            H 5.200000 0.000000 0.000000, \
            H 3.900000 -2.250000 0.000000, \
            H 1.300000 -2.250000 0.000000, \
            H 0.000000 0.000000 0.000000"

    mol = Molecules(
        geometry = geom,
        basis    = basis,
        multiplicity = multiplicity,
        charge = charge)

    # 利用JW变换得到泡利算符形式的氢分子哈密顿量
    fermion = mol.get_molecular_hamiltonian()
    pauli = jordan_wigner(fermion)
    H = PauliOperator(ast.literal_eval(pauli.to_string()))
    sparse_H = scipy.sparse.csr_matrix(H.to_matrix())

    return sparse_H
    

def Hamiltonian_from_dict(d):
    return PauliOperator(d)


def single_Circuit(qlist, i,k, theta):
    vqc = QCircuit()
    
    vqc.insert(H(qlist[i]))
    vqc.insert(RZ(qlist[i], np.pi/2))
    vqc.insert(CNOT(qlist[k], qlist[i]))
    vqc.insert(RZ(qlist[i], -np.pi/2))
    vqc.insert(RZ(qlist[k], np.pi/2))
    vqc.insert(H(qlist[i]))
    vqc.insert(RY(qlist[k], np.pi/2-theta))
    vqc.insert(CNOT(qlist[i], qlist[k]))
    vqc.insert(RY(qlist[k], theta-np.pi/2))
    vqc.insert(CNOT(qlist[k], qlist[i]))   
        
    return vqc


# def oneCircuit(qlist, a,b,c,d, theta):
#     vqc = QCircuit()

#     vqc.insert(CNOT(qlist[a], qlist[b]))
#     vqc.insert(CNOT(qlist[c], qlist[d]))
#     # vqc.insert(X(qlist[b]))
#     # vqc.insert(H(qlist[b]))
#     vqc.insert(RY(qlist[b], -np.pi/2))
#     # vqc.insert(X(qlist[d]))
#     # vqc.insert(H(qlist[d]))
#     vqc.insert(RY(qlist[d], -np.pi/2))

#     vqc.insert(CNOT(qlist[a], qlist[c]))
#     vqc.insert(RZ(qlist[a], np.pi/2))
    
#     vqc.insert(RX(qlist[a], theta/4))
    
#     vqc.insert(CNOT(qlist[a], qlist[b]))
#     # vqc.insert(H(qlist[b]))

#     vqc.insert(RX(qlist[a], -theta/4))
    
#     vqc.insert(CNOT(qlist[a], qlist[d]))
#     # vqc.insert(H(qlist[d]))
    
#     vqc.insert(RX(qlist[a], theta/4))
#     #vqc.insert(H(qlist[b]))
#     vqc.insert(CNOT(qlist[a], qlist[b]))
#     #vqc.insert(H(qlist[b]))
    
#     vqc.insert(RX(qlist[a], -theta/4))
#     vqc.insert(H(qlist[c]))
#     vqc.insert(CNOT(qlist[a], qlist[c]))
#     # vqc.insert(H(qlist[c]))
    
#     vqc.insert(RX(qlist[a], theta/4))
#     # vqc.insert(H(qlist[b]))
#     vqc.insert(CNOT(qlist[a], qlist[b]))
#     # vqc.insert(H(qlist[b]))
    
#     vqc.insert(RX(qlist[a], -theta/4))
#     #vqc.insert(H(qlist[d]))
#     vqc.insert(CNOT(qlist[a], qlist[d]))
#     # vqc.insert(H(qlist[d]))
#     # vqc.insert(X(qlist[d]))
#     vqc.insert(RY(qlist[d], np.pi/2))
    
#     vqc.insert(RX(qlist[a], theta/4))
#     # vqc.insert(H(qlist[b]))
#     vqc.insert(CNOT(qlist[a], qlist[b]))
#     # vqc.insert(H(qlist[b]))
#     # vqc.insert(X(qlist[b]))
#     vqc.insert(RY(qlist[b], np.pi/2))
    
#     vqc.insert(RX(qlist[a], -theta/4))
#     vqc.insert(RZ(qlist[a], -np.pi))
#     # vqc.insert(RZ(qlist[a], -np.pi/2))
#     # vqc.insert(H(qlist[c]))
#     vqc.insert(RZ(qlist[c], np.pi/2))
 
#     vqc.insert(CNOT(qlist[a], qlist[c]))
#     vqc.insert(RZ(qlist[c], -np.pi/2))
#     vqc.insert(H(qlist[c]))
#     # vqc.insert(RY(qlist[c], np.pi/2))
    
#     vqc.insert(CNOT(qlist[a], qlist[b]))
#     vqc.insert(CNOT(qlist[c], qlist[d]))
    
#     return vqc

def oneCircuit(qlist, a,b,c,d, theta):
    vqc = QCircuit()

    vqc.insert(CNOT(qlist[a], qlist[b]))
    vqc.insert(X(qlist[b]))
    vqc.insert(CNOT(qlist[c], qlist[d]))
    vqc.insert(X(qlist[d]))

    vqc.insert(CNOT(qlist[a], qlist[c]))
    vqc.insert(RZ(qlist[a], np.pi/2))
    
    vqc.insert(RX(qlist[a], theta/4))
    vqc.insert(H(qlist[b]))
    vqc.insert(CNOT(qlist[a], qlist[b]))
    vqc.insert(H(qlist[b]))
    
    vqc.insert(RX(qlist[a], theta/-4))
    vqc.insert(H(qlist[d]))
    vqc.insert(CNOT(qlist[a], qlist[d]))
    vqc.insert(H(qlist[d]))
    
    vqc.insert(RX(qlist[a], theta/4))
    vqc.insert(H(qlist[b]))
    vqc.insert(CNOT(qlist[a], qlist[b]))
    vqc.insert(H(qlist[b]))
    
    vqc.insert(RX(qlist[a], theta/-4))
    vqc.insert(H(qlist[c]))
    vqc.insert(CNOT(qlist[a], qlist[c]))
    vqc.insert(H(qlist[c]))
    
    vqc.insert(RX(qlist[a], theta/4))
    vqc.insert(H(qlist[b]))
    vqc.insert(CNOT(qlist[a], qlist[b]))
    vqc.insert(H(qlist[b]))
    
    vqc.insert(RX(qlist[a], theta/-4))
    vqc.insert(H(qlist[d]))
    vqc.insert(CNOT(qlist[a], qlist[d]))
    vqc.insert(H(qlist[d]))
    
    vqc.insert(RX(qlist[a], theta/4))
    vqc.insert(H(qlist[b]))
    vqc.insert(CNOT(qlist[a], qlist[b]))
    vqc.insert(H(qlist[b]))
    
    vqc.insert(RX(qlist[a], theta/-4))
    vqc.insert(RZ(qlist[a], -np.pi/2))
    vqc.insert(H(qlist[c]))
    vqc.insert(RZ(qlist[c], np.pi/2))
    vqc.insert(RZ(qlist[a], -np.pi/2))
    
    vqc.insert(CNOT(qlist[a], qlist[c]))
    vqc.insert(RZ(qlist[c], -np.pi/2))
    vqc.insert(H(qlist[c]))
    vqc.insert(X(qlist[b]))
    
    vqc.insert(CNOT(qlist[a], qlist[b]))
    vqc.insert(X(qlist[d]))
    vqc.insert(CNOT(qlist[c], qlist[d]))
    
    return vqc

def contruct_circuit(qlist, index, parameter):
    vqc = QCircuit()
    layers = len(parameter)
    parameter = var(parameter, True)

    # 初始化HF态
    for i in range(6):
        vqc.insert(X(qlist[i]))

    # 构造量子线路
    for layer in range(layers):
        # print(layer)
        i,j,k,l = index[layer].tolist()
        if k != 12:
            vqc.insert(oneCircuit(qlist, i,j,k,l, parameter[layer])) # 
        else:
            # print('i->k:',i,j)
            vqc.insert(single_Circuit(qlist, i,j, parameter[layer]))
            # vqc.insert(single_Circuit(qlist, i,j, parameter[layer]))
    
    vqc.insert(SWAP(qlist[0], qlist[11]))
    vqc.insert(SWAP(qlist[1], qlist[10]))
    vqc.insert(SWAP(qlist[2], qlist[9]))
    vqc.insert(SWAP(qlist[3], qlist[8]))
    vqc.insert(SWAP(qlist[4], qlist[7]))
    vqc.insert(SWAP(qlist[5], qlist[6]))
    
    return vqc



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--layers', type=int, default=20, help='Number of layers of the quantum circuit. Currently the number of layers only supports 20,48 or 65!')
    args = parser.parse_args()

    assert args.layers in [20, 48, 65], "Error: Ccurrently the number of layers only supports 20,48 or 65."

    now = datetime.datetime.now()
    timestamp = now.isoformat()
    save_folder = 'log'
    save_folder = '{}/exp{}/'.format(save_folder, timestamp)
    os.makedirs(save_folder)
    log_file = os.path.join(save_folder, 'log.txt')
    log = open(log_file, 'w')

    # set random seed
    np.random.seed(1234)

    # 利用JW变换得到泡利算符形式的氢分子哈密顿量
    # sparse_H = Hamiltonian_from_chemiq()
    sparse_H = scipy.sparse.load_npz("my_Hamiltonian.npz")
    sparse_H[abs(sparse_H) <= 0.04] = 0

    # 初始化量子虚拟机, 分配量子比特
    num_qubits = 12
    machine = init_quantum_machine(QMachineType.CPU)
    qlist = machine.qAlloc_many(num_qubits)

    # 初始化参数
    # data = np.loadtxt('index_48_layer.txt')
    data = np.loadtxt('layer_{}.txt'.format(args.layers))
    parameter = data[:,1].reshape(-1)
    index = data[:,-4:].astype(int)
    assert len(index) == args.layers, "Error: the number of layers does not match the length of the index file!"
    print('Number of parameters:', index.shape[0], file=log)
    if args.layers == 20:
        print("sparse Hamiltonian.")
        sparse_H[abs(sparse_H) <= 0.04] = 0

    ansatz_parameters = []
    delta_e_threshold =  1.6e-3 # 1e-3 for chemical accuracy
    max_ansatz_size = 250
    maxiter = 3000
    
    backend = QpandaBackend(machine, qlist, parameter, index, sparse_H, num_qubits)

    # create a vqe_runner object
    vqe_runner = VQERunner(backend=backend, optimizer='L-BFGS-B', optimizer_options={'gtol': 1e-08, 'maxiter': maxiter},
                           use_ansatz_gradient=False)
    
    hf_energy = -2.806471946359929
    iter_count = 0
    df_count = 0
    current_energy = np.inf
    previous_energy = 0
    initial_time = time.time()

    while current_energy - hf_energy >= delta_e_threshold and iter_count <= max_ansatz_size and iter_count < len(index):
        iter_count += 1
        print('New cycle ', iter_count, file=log)
        print(f"Elapsed time: {time.time() - initial_time}", file=log)
        previous_energy = current_energy

        result = vqe_runner.vqe_run(init_guess_parameters=ansatz_parameters + [0])
        current_energy = result.fun
        ansatz_parameters = list(result.x)
        delta_e = previous_energy - current_energy
        df_count += 1
        print(f"iter_count: {iter_count} \
              E: {current_energy} \
              error: {current_energy - hf_energy} \
              var_parameters:{list(result.x)[:df_count]}", file=log)
        log.flush()
print("Optimization Done! The final energy is:", file=log)
print(current_energy, file=log)
print(f"total time: {time.time() - initial_time}", file=log)
log.flush()