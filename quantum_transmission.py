import random
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
from utils.logger import get_logger

logger = get_logger(__name__)


supported_encodings = {
    "BB84": {
        "dimension": 2,
        "basis_sets": ["computational", "hadamard"],
        "error_tolerance": 0.11,
        "max_distance": 50
    },
    "SIX_STATE": {
        "dimension": 2,
        "basis_sets": ["computational", "hadamard", "circular"],
        "error_tolerance": 0.126,
        "max_distance": 40
    },
    "EIGHT_STATE": {
        "dimension": 2,
        "basis_sets": ["computational", "hadamard", "circular", "custom"],
        "error_tolerance": 0.15,
        "max_distance": 30
    },
    "DECOY_BB84": {
        "dimension": 2,
        "basis_sets": ["computational", "hadamard"],
        "error_tolerance": 0.11,
        "max_distance": 100,
        "recommended_intensity": 0.5
    },
    "E91": {
        "dimension": 2,
        "basis_sets": ["bell_basis"],
        "error_tolerance": 0.15,
        "max_distance": 100,
        "uses_entanglement": True
    }
}


def prepare_qubits(key_length, encoding_method):
    """
    Prepare qubits based on the specified encoding method.
    
    Args:
        key_length (int): Number of qubits to prepare
        encoding_method (str): Name of the encoding method from supported_encodings
        supported_encodings (dict): Dictionary of supported encoding methods
    
    Returns:
        tuple: (alice_bits, alice_bases, quantum_circuits)
    """
    if encoding_method not in supported_encodings:
        logger.error(f"Unsupported encoding method: {encoding_method}")
        return None, None, None
    
    encoding_info = supported_encodings[encoding_method]
    dimension = encoding_info["dimension"]
    basis_sets = encoding_info["basis_sets"]
    num_bases = len(basis_sets)
    
    logger.info(f"Preparing {key_length} qubits using {encoding_method} encoding with {num_bases} basis sets")
    
    # Generate random bits and bases
    alice_bits = [random.randint(0, dimension-1) for _ in range(key_length)]
    
    alice_bases = [random.randint(0, num_bases-1) for _ in range(key_length)]
    # print("Alice bases" + str(alice_bases))
    
    # Create quantum circuits for each qubit
    quantum_circuits = []
    
    for i in range(key_length):
        bit = alice_bits[i]
        basis_idx = alice_bases[i]
        basis_name = basis_sets[basis_idx]
        
        # Create a quantum circuit
        qc = QuantumCircuit(1, 1)
        
        # Apply appropriate encoding based on the basis
        encode_qubit(qc, bit, basis_name, encoding_method)
        
        quantum_circuits.append(qc)
    
    logger.info(f"Successfully prepared {key_length} qubits with {encoding_method} encoding")
    return alice_bases,alice_bits, quantum_circuits

def encode_qubit(qc, bit, basis_name, encoding_method):
    """
    Encode a single qubit in the specified basis.
    
    Args:
        qc (QuantumCircuit): Quantum circuit to modify
        bit (int): Bit value to encode
        basis_name (str): Name of the basis to use
        encoding_method (str): Name of the encoding method
    """
    # Computational basis (|0⟩ or |1⟩)
    if basis_name == "computational":
        if bit == 1:
            qc.x(0)
    
    # Hadamard basis (|+⟩ or |-⟩)
    elif basis_name == "hadamard":
        qc.h(0)
        if bit == 1:
            qc.z(0)
    
    # Circular basis (|0⟩+i|1⟩ or |0⟩-i|1⟩)
    elif basis_name == "circular":
        qc.h(0)
        if bit == 0:
            qc.s(0)
        else:
            qc.sdg(0)
    
    # Custom basis for EIGHT_STATE protocol (tetrahedral states)
    elif basis_name == "custom":
        if bit == 0:
            theta = np.arccos(1/np.sqrt(3))
            phi = 0
        else:
            theta = np.arccos(1/np.sqrt(3))
            phi = 2*np.pi/3
        
        qc.ry(theta, 0)
        qc.rz(phi, 0)
    
    # Bell basis for E91 protocol
    elif basis_name == "bell_basis":
        # For E91, we would normally prepare entangled pairs
        # This is just a placeholder as E91 uses a different process
        qc.h(0)
    
    else:
        logger.warning(f"Unknown basis: {basis_name}, defaulting to computational")
        if bit == 1:
            qc.x(0)
    
    # Add intensity modulation for DECOY_BB84
    if encoding_method == "DECOY_BB84" and random.random() < 0.3:
        intensity = supported_encodings["DECOY_BB84"]["recommended_intensity"]
        # Simulate lower intensity pulse (conceptual representation)
        current_state = np.sqrt(intensity)
        qc.rx(2 * np.arcsin(current_state), 0)

def prepare_e91_states(key_length):
    """
    Prepare entangled Bell pairs for E91 protocol.
    
    Args:
        key_length (int): Number of Bell pairs to prepare
    
    Returns:
        tuple: (alice_bases, bob_bases, bell_pairs)
    """
    logger.info(f"Preparing {key_length} Bell pairs for E91 protocol")
    
    # In E91, both parties randomly choose measurement bases
    alice_bases = [random.randint(0, 2) for _ in range(key_length)]  # 3 possible bases
    bob_bases = [random.randint(0, 2) for _ in range(key_length)]    # 3 possible bases
    
    # Create Bell pairs
    bell_pairs = []
    for _ in range(key_length):
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        bell_pairs.append(qc)
    
    logger.info(f"Successfully prepared {key_length} Bell pairs for E91 protocol")
    return alice_bases, bob_bases, bell_pairs

def measure_qubit(qc, basis_name):
    """
    Apply appropriate measurement transformations based on the basis.
    
    Args:
        qc (QuantumCircuit): Quantum circuit to modify
        basis_name (str): Name of the basis to use
    """
    # Apply appropriate basis transformation before measurement
    if basis_name == "computational":
        pass  # Standard Z-basis measurement
    
    elif basis_name == "hadamard":
        qc.h(0)  # Convert to Z-basis
    
    elif basis_name == "circular":
        qc.sdg(0)
        qc.h(0)  # Convert to Z-basis
    
    elif basis_name == "custom":
        # Inverse transformation for tetrahedral basis
        theta = np.arccos(1/np.sqrt(3))
        qc.rz(-2*np.pi/3, 0)
        qc.ry(-theta, 0)
    
    # Add measurement
    qc.measure(0, 0)

def measure_qubits(quantum_circuits, bob_bases, encoding_method, supported_encodings):
    """
    Bob measures received qubits in randomly chosen bases.
    
    Args:
        quantum_circuits (list): List of quantum circuits to measure
        bob_bases (list): List of bases chosen by Bob
        encoding_method (str): Name of the encoding method
        supported_encodings (dict): Dictionary of supported encoding methods
    
    Returns:
        list: Measurement results
    """
    logger.info(f"Measuring {len(quantum_circuits)} qubits")
    
    encoding_info = supported_encodings[encoding_method]
    basis_sets = encoding_info["basis_sets"]
    
    backend = Aer.get_backend('aer_simulator')
    results = []
    
    for i, qc in enumerate(quantum_circuits):
        if qc is None:  # Skip lost qubits
            results.append(None)
            continue
        
        qc_copy = qc.copy()
        basis_idx = bob_bases[i]
        
        if basis_idx < len(basis_sets):
            basis_name = basis_sets[basis_idx]
            measure_qubit(qc_copy, basis_name)
            
            # Execute the circuit
            qc_copy = transpile(qc_copy, backend)
            job = backend.run(qc_copy, shots=1)
            result = job.result().get_counts()
            measured_bit = int(list(result.keys())[0]) if result else 0
            
            results.append(measured_bit)
        else:
            logger.warning(f"Invalid basis index: {basis_idx}")
            results.append(None)
    
    logger.info(f"Completed {len(results)} measurements")
    return results

def measure_e91_states(bell_pairs, alice_bases, bob_bases):
    """
    Measure entangled Bell pairs for E91 protocol.
    
    Args:
        bell_pairs (list): List of quantum circuits representing Bell pairs
        alice_bases (list): Alice's choice of measurement bases
        bob_bases (list): Bob's choice of measurement bases
    
    Returns:
        list: List of tuples (alice_result, bob_result)
    """
    logger.info(f"Measuring {len(bell_pairs)} Bell pairs")
    
    backend = Aer.get_backend('aer_simulator')
    results = []
    
    for i, qc in enumerate(bell_pairs):
        if qc is None:  # Skip lost qubits
            results.append(None)
            continue
        
        qc_copy = qc.copy()
        
        # Apply measurement transformations for Alice's qubit (0)
        if alice_bases[i] == 0:  # Z-basis
            pass
        elif alice_bases[i] == 1:  # X-basis
            qc_copy.h(0)
        elif alice_bases[i] == 2:  # Y-basis
            qc_copy.sdg(0)
            qc_copy.h(0)
        
        # Apply measurement transformations for Bob's qubit (1)
        if bob_bases[i] == 0:  # Z-basis
            pass
        elif bob_bases[i] == 1:  # X-basis
            qc_copy.h(1)
        elif bob_bases[i] == 2:  # Y-basis
            qc_copy.sdg(1)
            qc_copy.h(1)
        
        # Measure both qubits
        qc_copy.measure([0, 1], [0, 1])
        
        # Execute the circuit
        qc_copy = transpile(qc_copy, backend)
        job = backend.run(qc_copy, shots=1)
        result = job.result().get_counts()
        measured_bits = list(result.keys())[0] if result else '00'
        
        # Parse measurement result
        results.append((int(measured_bits[0]), int(measured_bits[1])))
    
    logger.info(f"Completed measurements of {len(results)} Bell pairs")
    return results


def transmit_qubits(quantum_channel, qubits, alice, bob):
    """
    Transmit qubits from Alice to Bob via a quantum channel.
    
    Args:
        quantum_channel: The quantum channel object used for transmission
        qubits: The prepared qubits array to be transmitted
        alice: The sender participant object
        bob: The receiver participant object
    
    Returns:
        received_qubits: The qubits as received by Bob (may include noise effects)
    """
    # Store the qubits in Alice's sent_qubits property
    alice.sent_qubits = qubits
    
    # Log the transmission start
    logger.info(f"Starting transmission of {len(qubits)} qubits from Alice to Bob...")
    
    # Apply channel effects to simulate real quantum transmission
    received_qubits = quantum_channel.send(qubits)
    
    # Store the received qubits in Bob's received_qubits property
    bob.received_qubits = received_qubits
    
    # Log transmission completion
    logger.info(f"Transmission complete. {len(received_qubits)} qubits delivered to Bob.")
    
    return received_qubits

def reconcile_bases(alice_bases, bob_bases, alice_bits, bob_results):
    """
    Reconcile bases between Alice and Bob to establish a shared key.
    
    Args:
        alice_bases (list): Alice's chosen bases
        bob_bases (list): Bob's chosen bases
        alice_bits (list): Alice's original bits
        bob_results (list): Bob's measurement results
    
    Returns:
        tuple: (matching_indices, raw_key, error_estimation_indices)
    """
    logger.info("Reconciling bases to establish raw key")
    
    matching_indices = []
    raw_key = []
    
    # Find positions where Alice and Bob used the same basis
    for i in range(len(alice_bases)):
        if bob_results[i] is not None and alice_bases[i] == bob_bases[i]:
            matching_indices.append(i)
            raw_key.append(alice_bits[i])
    
    # Select a subset for error estimation
    num_test_bits = min(len(matching_indices) // 5, 100)  # Use about 20% for testing, max 100
    error_estimation_indices = random.sample(matching_indices, num_test_bits)
    
    logger.info(f"Found {len(matching_indices)} matching bases, using {len(error_estimation_indices)} for error estimation")
    return matching_indices, raw_key, error_estimation_indices

def check_security(alice_bits, bob_results, error_indices, encoding_method, supported_encodings):
    """
    Check the security of the quantum channel by estimating QBER.
    
    Args:
        alice_bits (list): Alice's original bits
        bob_results (list): Bob's measurement results
        error_indices (list): Indices to use for error estimation
        encoding_method (str): Name of the encoding method
        supported_encodings (dict): Dictionary of supported encoding methods
    
    Returns:
        tuple: (qber, is_secure)
    """
    logger.info("Checking security by estimating QBER")
    
    errors = 0
    valid_samples = 0
    
    for idx in error_indices:
        if bob_results[idx] is not None:
            if bob_results[idx] != alice_bits[idx]:
                errors += 1
            valid_samples += 1
    
    qber = errors / valid_samples if valid_samples > 0 else 1.0
    
    # Check if QBER is below the tolerance for this encoding method
    if encoding_method in supported_encodings:
        error_tolerance = supported_encodings[encoding_method]["error_tolerance"]
        is_secure = qber < error_tolerance
    else:
        # Default tolerance if encoding not found
        is_secure = qber < 0.11  # BB84 standard
    
    logger.info(f"Estimated QBER: {qber:.4f}, Channel is {'secure' if is_secure else 'not secure'}")
    return qber, is_secure

