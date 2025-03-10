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





























# import random
# import numpy as np
# from qiskit import QuantumCircuit, transpile
# from qiskit_aer import Aer
# from qiskit.quantum_info import random_statevector
# from utils.logger import get_logger

# logger = get_logger(__name__)

# supported_encodings = {
#     "BB84": {
#         "dimension": 2,
#         "basis_sets": ["computational", "hadamard"],
#         "error_tolerance": 0.11,
#         "max_distance": 50
#     },
#     "SIX_STATE": {
#         "dimension": 2,
#         "basis_sets": ["computational", "hadamard", "circular"],
#         "error_tolerance": 0.126,
#         "max_distance": 40
#     },
#     "EIGHT_STATE": {
#         "dimension": 2,
#         "basis_sets": ["computational", "hadamard", "circular", "custom"],
#         "error_tolerance": 0.15,
#         "max_distance": 30
#     },
#     "DECOY_BB84": {
#         "dimension": 2,
#         "basis_sets": ["computational", "hadamard"],
#         "error_tolerance": 0.11,
#         "max_distance": 100,
#         "recommended_intensity": 0.5
#     },
#     "E91": {
#         "dimension": 2,
#         "basis_sets": ["bell_basis"],
#         "error_tolerance": 0.15,
#         "max_distance": 100,
#         "uses_entanglement": True
#     }
# }


# def prepare_qubits(key_length, encoding_scheme, random_seed=None):
   
#     if random_seed is not None:
#         random.seed(random_seed)
#         np.random.seed(random_seed)
    
#     # Get encoding parameters from supported_encodings
#     if encoding_scheme not in supported_encodings:
#         raise ValueError(f"Encoding scheme {encoding_scheme} not supported. Choose from: {list(supported_encodings.keys())}")
    
#     encoding_params = supported_encodings[encoding_scheme]
#     dimension = encoding_params["dimension"]
#     basis_sets = encoding_params["basis_sets"]
    
#     logger.info(f"Preparing {key_length} qubits using {encoding_scheme} protocol...")
    
#     # Special handling for E91 (entanglement-based)
#     if "uses_entanglement" in encoding_params and encoding_params["uses_entanglement"]:
#         return prepare_entangled_qubits(key_length)
    
#     # For non-entanglement-based protocols
#     alice_bases = [random.choice(range(len(basis_sets))) for _ in range(key_length)]
#     alice_bits = [random.randint(0, 1) for _ in range(key_length)]
    
#     # Create quantum circuits for each qubit
#     circuits = []
    
#     for i in range(key_length):
#         bit = alice_bits[i]
#         basis = basis_sets[alice_bases[i]]
#         qc = QuantumCircuit(1, 1)
        
#         # Prepare qubit based on selected basis and bit value
#         if basis == "computational":
#             # |0⟩ or |1⟩
#             if bit == 1:
#                 qc.x(0)
#         elif basis == "hadamard":
#             # |+⟩ or |-⟩
#             qc.h(0)
#             if bit == 1:
#                 qc.z(0)
#         elif basis == "circular":
#             # |R⟩ or |L⟩ (right/left circular polarization)
#             qc.h(0)
#             if bit == 0:
#                 qc.s(0)  # |R⟩ = |0⟩ + i|1⟩
#             else:
#                 qc.sdg(0)  # |L⟩ = |0⟩ - i|1⟩
#         elif basis == "custom":
#             # Custom basis for EIGHT_STATE protocol
#             # Using a basis at 67.5° (3π/8)
#             qc.ry(3 * np.pi / 4, 0)
#             if bit == 1:
#                 qc.z(0)
        
#         circuits.append(qc)
    
#     # For DECOY_BB84, add decoy states with different intensities
#     if encoding_scheme == "DECOY_BB84":
#         recommended_intensity = encoding_params.get("recommended_intensity", 0.5)
#         # Add intensity information to each qubit
#         decoy_status = []
#         for i in range(key_length):
#             # Randomly assign some qubits as decoy states with lower intensity
#             is_decoy = random.random() < 0.3  # 30% chance of being a decoy
#             intensity = recommended_intensity if is_decoy else 1.0
#             decoy_status.append((is_decoy, intensity))
        
#         return alice_bases, alice_bits, circuits, decoy_status
    
#     return alice_bases, alice_bits, circuits

# def prepare_entangled_qubits(key_length):

#     logger.info(f"Preparing {key_length} entangled qubit pairs for E91 protocol...")
    
#     # In E91, both Alice and Bob choose measurement bases independently
#     # They typically choose from 3 different angles each
#     alice_angles = [0, np.pi/4, np.pi/2]  # 0°, 45°, 90°
#     bob_angles = [np.pi/4, np.pi/2, 3*np.pi/4]  # 45°, 90°, 135°
    
#     alice_bases = [random.choice(range(len(alice_angles))) for _ in range(key_length)]
#     bob_bases = [random.choice(range(len(bob_angles))) for _ in range(key_length)]
    
#     # Create entangled bell pairs
#     circuits = []
#     for _ in range(key_length):
#         qc = QuantumCircuit(2, 2)
#         # Create Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2
#         qc.h(0)
#         qc.cx(0, 1)
#         circuits.append(qc)
    
#     return alice_bases, bob_bases, circuits

# def transmit_qubits(quantum_channel, qubits, encoding_scheme="BB84"):
#     """
#     Transmit qubits through a quantum channel.
    
#     Args:
#         quantum_channel: A quantum channel object with a transmit method
#         qubits: List of quantum circuits to transmit
#         encoding_scheme (str): The encoding scheme being used
        
#     Returns:
#         list: Transmitted qubits (may include None for lost qubits)
#     """
#     # Get max distance from encoding parameters
#     max_distance = supported_encodings[encoding_scheme].get("max_distance", 50)
    
#     logger.info(f"Transmitting {len(qubits)} qubits using {encoding_scheme} protocol...")
#     logger.info(f"Maximum recommended distance for {encoding_scheme}: {max_distance} km")
    
#     transmitted_qubits = quantum_channel.transmit(qubits)
#     quantum_channel.received_qubits = transmitted_qubits
    
#     # Count lost qubits
#     lost_qubits = transmitted_qubits.count(None)
#     logger.info(f"Transmission complete. {lost_qubits} qubits lost in transmission ({lost_qubits/len(qubits)*100:.2f}%)")
    
#     return transmitted_qubits

# def measure_qubits(qubits, bases, encoding_scheme="BB84"):
#     logger.info(f"Measuring {len(qubits)} qubits...")
    
#     backend = Aer.get_backend('aer_simulator')
#     results = []
    
#     encoding_params = supported_encodings[encoding_scheme]
#     basis_sets = encoding_params["basis_sets"]
    
#     for qc, basis_idx in zip(qubits, bases):
#         if qc is None:  # Skip lost qubits
#             results.append(None)
#             continue
            
#         basis = basis_sets[basis_idx]
#         qc_copy = qc.copy()
        
#         # Apply measurement in the specified basis
#         if basis == "computational":
#             # Measure in Z-basis (computational)
#             pass  # No transformation needed
#         elif basis == "hadamard":
#             # Measure in X-basis (Hadamard)
#             qc_copy.h(0)
#         elif basis == "circular":
#             # Measure in Y-basis (circular)
#             qc_copy.sdg(0)
#             qc_copy.h(0)
#         elif basis == "custom":
#             # Measure in custom basis for EIGHT_STATE
#             qc_copy.ry(-3 * np.pi / 4, 0)
#         elif basis == "bell_basis":
#             # For E91, this is handled separately in measure_entangled_qubits
#             pass
            
#         # Perform measurement
#         qc_copy.measure(0, 0)
#         qc_copy = transpile(qc_copy, backend)
#         job = backend.run(qc_copy, shots=1)
#         result = job.result().get_counts()
#         measured_bit = int(list(result.keys())[0]) if result else 0
#         results.append(measured_bit)
    
#     return results

# def measure_entangled_qubits(circuits, alice_bases, bob_bases):
   
#     logger.info(f"Measuring {len(circuits)} entangled qubit pairs for E91 protocol...")
    
#     backend = Aer.get_backend('aer_simulator')
#     alice_results = []
#     bob_results = []
    
#     # Alice and Bob's measurement angles
#     alice_angles = [0, np.pi/4, np.pi/2]  # 0°, 45°, 90°
#     bob_angles = [np.pi/4, np.pi/2, 3*np.pi/4]  # 45°, 90°, 135°
    
#     for qc, alice_basis, bob_basis in zip(circuits, alice_bases, bob_bases):
#         qc_copy = qc.copy()
        
#         # Apply rotation to first qubit (Alice) based on her chosen basis
#         alice_angle = alice_angles[alice_basis]
#         qc_copy.ry(2 * alice_angle, 0)
        
#         # Apply rotation to second qubit (Bob) based on his chosen basis
#         bob_angle = bob_angles[bob_basis]
#         qc_copy.ry(2 * bob_angle, 1)
        
#         # Measure both qubits
#         qc_copy.measure([0, 1], [0, 1])
#         qc_copy = transpile(qc_copy, backend)
#         job = backend.run(qc_copy, shots=1)
#         result = job.result().get_counts()
#         measured_bits = list(result.keys())[0] if result else '00'
        
#         alice_results.append(int(measured_bits[0]))
#         bob_results.append(int(measured_bits[1]))
    
#     return alice_results, bob_results


# def monitor_qber(transmitted_qubits, original_qubits, sample_indices):
#     errors = 0
#     valid_samples = 0
    
#     for idx in sample_indices:
#         if transmitted_qubits[idx] is not None:
#             if transmitted_qubits[idx][0] != original_qubits[idx][0]:
#                 errors += 1
#             valid_samples += 1
    
#     qber = errors / valid_samples if valid_samples > 0 else 0
#     logger.info(f"Monitored QBER: {qber:.4f} from {valid_samples} samples")
#     return qber







