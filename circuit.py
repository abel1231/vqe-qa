import time
from pyqpanda import *
import numpy as np
import argparse
from pychemiq import Molecules,ChemiQ
from pychemiq.Transform.Mapping import jordan_wigner, bravyi_kitaev, parity, segment_parity, MappingType
from scipy.optimize import minimize
from qiskit.opflow import PauliSumOp
from Hamiltonian import Hamiltonian_data
import scipy


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

def qiskit_problem_PauliOperator(Hamiltonian):
    pauli_list = []
    for pauliOp in Hamiltonian.keys():
        pauli_list.append((get_Pauli(pauliOp), Hamiltonian[pauliOp]))
    return PauliSumOp.from_list(pauli_list, coeff=1.0)


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


def oneCircuit_new(qlist, a,b,c,d , theta):
    vqc = QCircuit()

    vqc.insert(CNOT(qlist[a], qlist[b]))
    vqc.insert(CNOT(qlist[c], qlist[d]))

    vqc.insert(X(qlist[b]))
    vqc.insert(X(qlist[d]))

    vqc.insert(CNOT(qlist[a], qlist[c]))

    vqc.insert(RY(qlist[a], theta/8))
    vqc.insert(H(qlist[b]))

    vqc.insert(CNOT(qlist[a], qlist[b]))

    vqc.insert(RY(qlist[a], -theta/8))
    vqc.insert(H(qlist[d]))

    vqc.insert(CNOT(qlist[a], qlist[d]))

    vqc.insert(RY(qlist[a], theta/8))

    vqc.insert(CNOT(qlist[a], qlist[b]))

    vqc.insert(RY(qlist[a], -theta/8))
    vqc.insert(H(qlist[c]))

    vqc.insert(CNOT(qlist[a], qlist[c]))

    vqc.insert(RY(qlist[a], theta/8))

    vqc.insert(CNOT(qlist[a], qlist[b]))

    vqc.insert(RY(qlist[a], -theta/8))

    vqc.insert(CNOT(qlist[a], qlist[d]))

    vqc.insert(RY(qlist[a], theta/8))
    vqc.insert(H(qlist[d]))

    vqc.insert(CNOT(qlist[a], qlist[b]))

    vqc.insert(RY(qlist[a], -theta/8))
    vqc.insert(H(qlist[b]))
    vqc.insert(RZ(qlist[c], -np.pi/2))

    vqc.insert(CNOT(qlist[a], qlist[c]))

    vqc.insert(RZ(qlist[a], np.pi/2))
    vqc.insert(RZ(qlist[c], -np.pi/2))

    vqc.insert(X(qlist[b]))
    vqc.insert(RY(qlist[c], -np.pi/2))
    vqc.insert(X(qlist[d]))

    vqc.insert(CNOT(qlist[a], qlist[b]))
    vqc.insert(CNOT(qlist[c], qlist[d]))
        
    return vqc


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
    
    vqc.insert(RX(qlist[a], -theta/4))
    vqc.insert(H(qlist[d]))
    vqc.insert(CNOT(qlist[a], qlist[d]))
    vqc.insert(H(qlist[d]))
    
    vqc.insert(RX(qlist[a], theta/4))
    vqc.insert(H(qlist[b]))
    vqc.insert(CNOT(qlist[a], qlist[b]))
    vqc.insert(H(qlist[b]))
    
    vqc.insert(RX(qlist[a], -theta/4))
    vqc.insert(H(qlist[c]))
    vqc.insert(CNOT(qlist[a], qlist[c]))
    vqc.insert(H(qlist[c]))
    
    vqc.insert(RX(qlist[a], theta/4))
    vqc.insert(H(qlist[b]))
    vqc.insert(CNOT(qlist[a], qlist[b]))
    vqc.insert(H(qlist[b]))
    
    vqc.insert(RX(qlist[a], -theta/4))
    vqc.insert(H(qlist[d]))
    vqc.insert(CNOT(qlist[a], qlist[d]))
    vqc.insert(H(qlist[d]))
    
    vqc.insert(RX(qlist[a], theta/4))
    vqc.insert(H(qlist[b]))
    vqc.insert(CNOT(qlist[a], qlist[b]))
    vqc.insert(H(qlist[b]))
    
    vqc.insert(RX(qlist[a], -theta/4))
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


def get_expectation(Hamiltonian_matrix, index, parameter, train=True):

    def execute_circ(theta):
        # p = len(theta) // 2
        # beta = theta[:p]
        # gamma = theta[p:]

        # beta = var(_beta.reshape(-1,1))
        # gamma = var(_gamma.reshape(-1,1))

        layers = len(theta)

        vqc = QCircuit()

        # 构造量子线路
        for i in range(layers):
            vqc.insert(oneCircuit(qlist, a,b,c,d,theta[i]))

        # 构建量子线路实例
        prog = QProg()
        prog.insert(vqc)

        if train:
            # 输出每个selection对应的probability, 例如: result = {'00': 0.5, '01': 0.0, '10': 0.5, '11': 0.0}
            result = prob_run_list(prog, qlist, -1)
            statevector = np.sqrt(np.array(result))  # vector representation of the output state

            loss = statevector @ Hamiltonian_matrix @ statevector
            assert np.imag(loss) < 1e-10
            return np.real(loss)
        else:
            result = prob_run_dict(prog, qlist, -1)
            result_tonumpy = np.array(prob_run_list(prog, qlist, -1))
            return result, result_tonumpy
    return execute_circ


def construct_circuit(qlist, parameter, index):
    layers = len(parameter)

    vqc = QCircuit()

    # 初始化HF态
    for i in range(6):
        vqc.insert(X(qlist[i]))

    # 构造量子线路
    for layer in range(layers):
        print(layer)
        i,j,k,l = index[layer].tolist()
        if k != 12:
            vqc.insert(oneCircuit(qlist, i,j,k,l, parameter[layer])) # 
        else:
            print('i->k:',i,j)
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

    # set random seed
    np.random.seed(1234)

    # 初始化参数
    data = np.loadtxt('./data_qeb_new.txt')
    # data = data[np.lexsort(-data[:,::-1].T)]
    print(data)
    parameter = data[:,1].reshape(-1)
    index = data[:,-4:].astype(int)
    
    
    # 初始化量子虚拟机, 分配量子比特
    num_qubits = 12
    
    qvm = CPUQVM()
    qvm.init_qvm()
    
    # machine = init_quantum_machine(QMachineType.CPU)

    qlist = qvm.qAlloc_many(num_qubits)
  
    # 计算所给问题对应的哈密尔顿量, 及其对应的矩阵
    # multiplicity = 1
    # charge = 0
    # basis =  "sto-3g"
    # geom = "H 1.300000 2.250000 0.000000, \
    #         H 3.900000 2.250000 0.000000, \
    #         H 5.200000 0.000000 0.000000, \
    #         H 3.900000 -2.250000 0.000000, \
    #         H 1.300000 -2.250000 0.000000, \
    #         H 0.000000 0.000000 0.000000"

    # mol = Molecules(
    #     geometry = geom,
    #     basis    = basis,
    #     multiplicity = multiplicity,
    #     charge = charge)

    # # 利用JW变换得到泡利算符形式的氢分子哈密顿量
    # fermion_H2 = mol.get_molecular_hamiltonian()
    # pauli_H2 = jordan_wigner(fermion_H2)

    vqc = construct_circuit(qlist, parameter, index)
    # 构建量子线路实例
    prog = QProg()
    prog.insert(vqc)
    # print(prog)
    qasm = convert_qprog_to_qasm(prog, qvm)
    # print(qasm)
    # print(prog)
    # result = prob_run_list(prog, qlist, -1)
    # statevector = np.sqrt(np.array(result))  # vector representation of the output state
    result = qvm.prob_run_list(prog, qlist, -1)
    # statevector = np.sqrt(np.array(result))
    statevector = np.array(qvm.get_qstate())# [:,np.newaxis]
    print(statevector.shape)

    # Hamiltonian = Hamiltonian_data
    # Pauli_sum = qiskit_problem_PauliOperator(Hamiltonian)
    # Hamiltonian_matrix = Pauli_sum.to_matrix()
    
    Hamiltonian_matrix = scipy.sparse.load_npz('sparse_matrix.npz').toarray()
    print('Hamiltonian_matrix', Hamiltonian_matrix.shape)
    
    loss = statevector @ Hamiltonian_matrix @ statevector.conjugate().T
    # loss = statevector.T.conjugate() @ Hamiltonian_matrix @ statevector
    np.save("Hamiltonian_matrix.npy", Hamiltonian_matrix)
    assert np.imag(loss) < 1e-10
    print(loss)
    qvm.finalize()

    # mat = get_unitary(prog)

    # mat = np.array(mat).reshape(2**12, 2**12)
    # print(mat, mat.shape)
    # np.save("first_unitary_all.npy", mat)
