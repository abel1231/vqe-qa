from pyqpanda import *
import numpy as np

def train(t):
     
    # t =  0.885414105  # This is not optimal; As an exercise, set this to the
           # value that will get the best results. See section 8 for solution.
    conf = complex(0, 1)

    nqubits = 4  # Total number of qubits
    nb = 1  # Number of qubits representing the solution
    nl = 2  # Number of qubits representing the eigenvalues


    # a = 1  # Matrix diagonal
    # b = -1/3  # Matrix off-diagonal

    A = np.array([[1.0,1],[1,-1]]) / np.sqrt(2) / 1e8
    b = np.array([1.0/4,-1/2]) * np.sqrt(2) /1e8

    # A = np.array([[1.0,-1/3],[-1/3,1]])
    # b = np.array([1.0,0])
    # A = np.array([[2,3],[3,-2]])
    # b = np.array([4.0, -7])
    norm_b = np.linalg.norm(b)
    b_norm = b / norm_b
    solution = np.linalg.solve(A,b)
    norm  = np.linalg.norm(solution)

    assert np.abs(b_norm.T @ b_norm - 1) < 1e-12
    assert b_norm[0] >= 0
    theta = np.arccos(b_norm[0])  # Angle defining |b>

    exp_iAt_mat = expMat(conf, A, t)
    # print(exp_iAt_mat @ exp_iAt_mat.T.conjugate())

    exp_iAt = []
    for i in range(exp_iAt_mat.shape[0]):
        for j in range(exp_iAt_mat.shape[1]):
            real = exp_iAt_mat[i][j].real
            imag = exp_iAt_mat[i][j].imag
            exp_iAt.append(complex(real, imag))
    # print(exp_iAt_mat)
    # print(exp_iAt)

    exp_2iAt_mat = expMat(conf, 2*A, t)
    # print(exp_2iAt_mat @ exp_2iAt_mat.T.conjugate())

    exp_2iAt = []
    for i in range(exp_2iAt_mat.shape[0]):
        for j in range(exp_2iAt_mat.shape[1]):
            real = exp_2iAt_mat[i][j].real
            imag = exp_2iAt_mat[i][j].imag
            exp_2iAt.append(complex(real, imag))

    # print("exp_iAt_mat is")
    # print(exp_iAt_mat)

    qvm = CPUQVM()
    qvm.init_qvm()

    # qubits = qvm.qAlloc_many(nqubits)
    # qrb = qubits[0]
    # qrl = qubits[1].append(qubits[2])
    # qra = qubits[nb+nl:nb+nl+1]

    qrb = qvm.qAlloc_many(nb)
    qrl = qvm.qAlloc_many(nl)
    qra = qvm.qAlloc_many(1)
    cbits = qvm.cAlloc_many(4)

    # 构建量子程序
    prog = QProg()
    circuit = QCircuit()
    circuit_qpe = create_empty_circuit()
    circuit_control_rotation = create_empty_circuit()
    circuit_inverse_qpe = create_empty_circuit()
    # circuit_x = create_empty_circuit()
    # circuit_x << U4(exp_iAt, qrb[0])
    # operator_x = QOperator(circuit_x)
    # unitary_x = operator_x.get_matrix()
    # print("unitary_x is")
    # print(unitary_x)
    # print(type(unitary_x))
    

    # State preparation
    circuit << RY(qrb[0], 2*theta) \
            << Z(qrb[0])

    # circuit << RY(qrb[0], 2*0)
    
    # circuit << H(qrb[0])
    
    # Hadamard gate
    circuit_qpe << H(qrl[0]) \
                << H(qrl[1])
            
    # Controlled e^{iAt} on \lambda_{1}:
    circuit_qpe << U4(exp_iAt, qrb[0]).control(qrl[1])

    # Controlled e^{2iAt} on \lambda_{1}
    circuit_qpe << U4(exp_2iAt, qrb[0]).control(qrl[0])

    # Inverse QFT
    circuit_qpe << CNOT(qrl[0],qrl[1]) \
                << CNOT(qrl[1],qrl[0]) \
                << CNOT(qrl[0],qrl[1]) \
                << H(qrl[1]) \
                << S(qrl[0]).dagger().control(qrl[1]) \
                << H(qrl[0]) \
                
    
    # Eigenvalue rotation
    t1=(-np.pi +np.pi/3 - 2*np.arcsin(1/3))/4
    t2=(-np.pi -np.pi/3 + 2*np.arcsin(1/3))/4
    t3=(np.pi -np.pi/3 - 2*np.arcsin(1/3))/4
    t4=(np.pi +np.pi/3 + 2*np.arcsin(1/3))/4

    circuit_control_rotation << CNOT(qrl[1], qra[0]) \
                             << RY(qra[0], t1) \
                             << CNOT(qrl[0], qra[0]) \
                             << RY(qra[0], t2) \
                             << CNOT(qrl[1], qra[0]) \
                             << RY(qra[0], t3) \
                             << CNOT(qrl[0], qra[0]) \
                             << RY(qra[0], t4) \
            
    # Inversive QPE
    circuit_inverse_qpe << circuit_qpe.dagger()

    # Construct the whole circuit
    circuit << circuit_qpe \
            << circuit_control_rotation \
            << circuit_inverse_qpe
    
    prog << circuit
    prog << Measure(qrb[0], cbits[0]) \
         << Measure(qra[0], cbits[-1])
    prog << measure_all(qrl,cbits[1:-1])
    # print(prog)

    # # 量子程序运行1000次，并返回测量结果
    result = qvm.run_with_configuration(prog, cbits, int(1e5))
    print(result)

    result = qvm.get_qstate()

    answer  = np.array([result[8],result[9]]).real
    # print(np.linalg.norm(result))
    answer = answer / np.linalg.norm(answer)

    # 打印量子态在量子程序多次运行结果中出现的次数
    # print(answer)
    # print(solution / norm)
    
    prob = 0.0
    for i in range(8):
        amplitude = np.array(result[8+i])
        prob += np.real(amplitude * amplitude.conjugate())

    return answer, solution / norm

if __name__ == "__main__":
    answer = []
    flag = 0
    for i in range(8854141000,8854141100):
        t = i / 1e10
        _answer, _solution = train(t)
        if flag == 0:
            print(_solution)
            flag = 1
        print(t, _answer)
