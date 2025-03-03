import random
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
from utils.logger import get_logger

logger = get_logger(__name__)

def prepare_qubits(key_length, dimension, encoding_basis):
    logger.info(f"Preparing qubits with {dimension}-dimension {encoding_basis} encoding...")
    
    alice_bases = [random.randint(0, dimension-1) for _ in range(key_length)]
    qubits = []
    
    if encoding_basis == "Computational":
        for i in range(key_length):
            bit = random.randint(0, 1)
            basis = alice_bases[i]
            qubits.append((bit, basis))
    
    elif encoding_basis == "Fourier":
        for i in range(key_length):
            bit = random.randint(0, 1)
            basis = alice_bases[i]
            # Create a more complex encoding
            qubits.append((bit, basis))
    
    else:  # Custom encoding
        for i in range(key_length):
            bit = random.randint(0, 1)
            basis = alice_bases[i]
            qubits.append((bit, basis))
    
    logger.info(f"Prepared {len(qubits)} qubits")
    return alice_bases, qubits

def create_entangled_pairs(length):
    logger.info(f"Creating {length} entangled qubit pairs...")
    circuits = []
    
    for _ in range(length):
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        circuits.append(qc)
    
    return circuits

def transmit_qubits(quantum_channel, qubits):
    logger.info(f"Transmitting {len(qubits)} qubits...")
    transmitted_qubits = quantum_channel.transmit(qubits)
    quantum_channel.received_qubits = quantum_channel.transmit(qubits)
    # Count lost qubits
    lost_qubits = transmitted_qubits.count(None)
    logger.info(f"Transmission complete. {lost_qubits} qubits lost in transmission")
    
    return transmitted_qubits

def measure_entangled_qubits(circuits, bases):
    logger.info(f"Measuring {len(circuits)} entangled qubit pairs...")
    backend = Aer.get_backend('aer_simulator')
    results = []
    
    for qc, basis in zip(circuits, bases):
        qc_copy = qc.copy()
        
        # Apply measurement transformation based on basis
        if basis == 1:
            qc_copy.h(0)
            qc_copy.h(1)
        elif basis == 2:
            qc_copy.sdg(0)
            qc_copy.h(0)
            qc_copy.sdg(1)
            qc_copy.h(1)
        
        qc_copy.measure([0, 1], [0, 1])
        qc_copy = transpile(qc_copy, backend)
        job = backend.run(qc_copy, shots=1)
        result = job.result().get_counts()
        measured_bits = list(result.keys())[0] if result else '00'
        results.append((int(measured_bits[0]), int(measured_bits[1])))
    
    return results

def monitor_qber(transmitted_qubits, original_qubits, sample_indices):
    errors = 0
    valid_samples = 0
    
    for idx in sample_indices:
        if transmitted_qubits[idx] is not None:
            if transmitted_qubits[idx][0] != original_qubits[idx][0]:
                errors += 1
            valid_samples += 1
    
    qber = errors / valid_samples if valid_samples > 0 else 0
    logger.info(f"Monitored QBER: {qber:.4f} from {valid_samples} samples")
    return qber