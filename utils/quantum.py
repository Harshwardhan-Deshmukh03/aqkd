import random
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
from utils.logger import get_logger

logger = get_logger(__name__)

def create_bb84_circuit(bit, basis):
    """Create a BB84 quantum circuit for a single qubit"""
    qc = QuantumCircuit(1, 1)
    
    # Prepare the qubit based on bit value and basis
    if bit == 1:
        qc.x(0)  # |1⟩ state
    
    if basis == 1:
        qc.h(0)  # Apply Hadamard for X basis
    
    return qc

def create_e91_circuit():
    """Create an E91 (entanglement-based) quantum circuit"""
    qc = QuantumCircuit(2, 2)
    
    # Create a Bell state (entangled pair)
    qc.h(0)
    qc.cx(0, 1)
    
    return qc

def simulate_measurement(circuit, basis=0):
    """Simulate measurement of a quantum circuit"""
    backend = Aer.get_backend('aer_simulator')
    
    # Make a copy to avoid modifying the original
    qc = circuit.copy()
    
    # Apply basis transformation before measurement if needed
    if basis == 1:
        qc.h(0)
    elif basis == 2:
        qc.sdg(0)
        qc.h(0)
    
    qc.measure(0, 0)
    qc = transpile(qc, backend)
    
    job = backend.run(qc, shots=1)
    result = job.result().get_counts()
    
    # Get the measurement outcome
    bit = int(list(result.keys())[0]) if result else 0
    return bit

def get_measurement_statistics(circuit, shots=1000):
    """Get measurement statistics for a circuit"""
    backend = Aer.get_backend('aer_simulator')
    
    # Make a copy and add measurement
    qc = circuit.copy()
    num_qubits = qc.num_qubits
    
    qc.measure_all()
    qc = transpile(qc, backend)
    
    job = backend.run(qc, shots=shots)
    result = job.result().get_counts()
    
    logger.info(f"Measurement statistics (shots={shots}): {result}")
    return result

def decoy_state_simulation(mean_photon_numbers, num_qubits):
    """Simulate decoy state protocol with different mean photon numbers"""
    signal_states = []
    decoy_states = []
    vacuum_states = []
    
    # Distribute qubits across signal, decoy, and vacuum states
    total_states = sum(num_qubits)
    
    for i in range(total_states):
        photon_type = random.choices(range(len(mean_photon_numbers)), weights=num_qubits, k=1)[0]
        
        # Generate Poisson-distributed photon number
        photon_count = np.random.poisson(mean_photon_numbers[photon_type])
        
        if photon_type == 0:
            signal_states.append((i, photon_count))
        elif photon_type == 1:
            decoy_states.append((i, photon_count))
        else:
            vacuum_states.append((i, photon_count))
    
    logger.info(f"Decoy state simulation: {len(signal_states)} signal, {len(decoy_states)} decoy, {len(vacuum_states)} vacuum")
    return signal_states, decoy_states, vacuum_states