from pyqpanda import *
import numpy as np

t = 2.356194490192345  # This is not optimal; As an exercise, set this to the
        # value that will get the best results. See section 8 for solution.
conf = complex(0, 1)

nqubits = 4  # Total number of qubits
nb = 1  # Number of qubits representing the solution
nl = 2  # Number of qubits representing the eigenvalues

A = np.array([[1.0, -1/3],[-1/3, 1]])
b = np.array([-1.0/2, 5.0/6])

norm_b = np.linalg.norm(b)
b_norm = b / norm_b
solution = np.linalg.solve(A,b)
norm  = np.linalg.norm(solution)

assert np.abs(b_norm.T @ b_norm - 1) < 1e-12
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


# State preparation
circuit << RY(qrb[0], 2*theta) \
        # << U4([complex(-1),complex(0),complex(0),complex(1)], qrb[0])

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

# # 量子程序运行1000000次，并返回测量结果
result = qvm.run_with_configuration(prog, cbits, int(1e6))

result = qvm.get_qstate()

answer  = np.array([result[8],result[9]]).real

# norm
I = np.eye(2)
Z = np.array([[1.0,0],[0,-1]])

op_one = (I - Z) / 2
op_zero = (I + Z) / 2

observable = np.kron(np.kron(np.kron(op_one, op_zero), op_zero), I)
density_matrix = state_to_density_matrix(result)
norm_2 = np.trace(observable @ density_matrix)
scaling = np.abs(np.linalg.eig(A)[0]).min()
Eu_norm = np.sqrt(norm_2) / scaling
answer = answer / np.linalg.norm(answer) * Eu_norm * norm_b
answer = answer.tolist()


print(answer,type(answer))

print('######')

print(convert_qprog_to_originir(prog,qvm))

    


