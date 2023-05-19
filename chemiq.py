# 先导入需要用到的包
from pychemiq import Molecules,ChemiQ,QMachineType
from pychemiq.Transform.Mapping import jordan_wigner, bravyi_kitaev, parity, segment_parity, MappingType
from pychemiq.Optimizer import vqe_solver
from pychemiq.Circuit.Ansatz import UCC, HardwareEfficient, SymmetryPreserved
import numpy as np
from pyscf import gto, scf, fci
import datetime, os

# calculate the theoretical energy using pyscf
atom = '''
H 1.300000 2.250000 0.000000
H 3.900000 2.250000 0.000000
H 5.200000 0.000000 0.000000
H 3.900000 -2.250000 0.000000
H 1.300000 -2.250000 0.000000
H 0.000000 0.000000 0.000000
'''

mol = gto.M(atom=atom,   # in Angstrom
    basis='STO-3G',
    charge=0,
    spin=0)
myhf = scf.HF(mol).run()
cisolver = fci.FCI(myhf)
print(cisolver.kernel()[0])


def train(mapping, mapping_type, ansatz_name, optimizer):
    # 初始化分子的电子结构参数，包括电荷、基组、原子坐标、自旋多重度
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
    fermion_H2 = mol.get_molecular_hamiltonian()
    pauli_H2 = eval(mapping)(fermion_H2)
    print(f"number of qubits: {mol.n_qubits}")

    # 准备量子线路，需要指定的参数有量子虚拟机类型machine_type，拟设映射类型mapping_type，
    # 泡利哈密顿量的项数pauli_size，电子数目n_elec与量子比特的数目n_qubits
    chemiq = ChemiQ()
    machine_type = QMachineType.CPU_SINGLE_THREAD
    mapping_type = mapping_type
    pauli_size = len(pauli_H2.data())
    n_qubits = mol.n_qubits
    n_elec = mol.n_electrons
    chemiq.prepare_vqe(machine_type,mapping_type,n_elec,pauli_size,n_qubits)

    # 设置拟设类型，这里我们使用UCCSD拟设
    # ansatz = UCC("UCCSD",n_elec,mapping_type,chemiq=chemiq)
    # ansatz = HardwareEfficient(n_elec,chemiq=chemiq)
    # ansatz = SymmetryPreserved(n_elec,chemiq=chemiq)
    if ansatz_name == 'UCC':
        ansatz = eval(ansatz_name)("UCCSD",n_elec,mapping_type,chemiq=chemiq)
    elif ansatz_name == 'HardwareEfficient':
        ansatz = HardwareEfficient(n_elec,chemiq=chemiq)

    # 指定经典优化器与初始参数并迭代求解
    # method = "SLSQP"
    method = optimizer
    init_para = np.zeros(ansatz.get_para_num())
    solver = vqe_solver(
            method = method,
            pauli = pauli_H2,
            chemiq = chemiq,
            ansatz = ansatz,
            init_para=init_para)
    result = solver.fun_val
    n_calls = solver.fcalls
    print(result,f"函数共调用{n_calls}次")
    energies = chemiq.get_energy_history()
    print(energies)
    print(min(energies))

    return min(energies)

mapping_ls = ["jordan_wigner", "bravyi_kitaev", "parity", "segment_parity"]
mapping_type_ls = [MappingType.Jordan_Wigner, MappingType.Bravyi_Kitaev, MappingType.Parity, MappingType.SegmentParity]
optimizer_ls = ['NELDER-MEAD', 'POWELL', 'COBYLA', 'L-BFGS-B', 'Gradient-Descent', 'SLSQP']
ansatz_ls = ['UCC', 'HardwareEfficient']

now = datetime.datetime.now()
timestamp = now.isoformat()
save_folder = 'log'
save_folder = '{}/exp{}/'.format(save_folder, timestamp)
os.makedirs(save_folder)
log_file = os.path.join(save_folder, 'log.txt')
log = open(log_file, 'w')

for optimizer in optimizer_ls:
    for ansatz in ansatz_ls:
        for i in range(4):
            mapping = mapping_ls[i]
            mapping_type = mapping_type_ls[i]

            print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%", file=log)
            print(f"optimizer={optimizer}, ansatz={ansatz}, mapping={mapping}, mapping_type={mapping_type}", file=log)

            energy = train(mapping, mapping_type, ansatz, optimizer)
            print(energy, file=log)
            log.flush()


