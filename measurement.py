import random
from utils.logger import get_logger

logger = get_logger(__name__)

# def perform_measurements(quantum_channel, bob_bases):
#     """Perform measurements on the received qubits"""
#     logger.info(f"Bob performing measurements with {len(bob_bases)} bases...")
    
#     # In a real system, this would be receiving and measuring actual qubits
#     # For simulation, we'll assume the qubits are already in the quantum_channel
#     received_qubits = quantum_channel.received_qubits
    
#     if not received_qubits:
#         logger.error("No qubits received for measurement")
#         return []
    
#     measurements = []
#     for i, basis in enumerate(bob_bases):
#         if i < len(received_qubits) and received_qubits[i] is not None:
#             # Simulating measurement
#             # If Bob's basis matches the qubit's basis, he'll get the correct result
#             # Otherwise, random result with 50% probability
#             qubit_basis = received_qubits[i][1]
#             qubit_value = received_qubits[i][0]
            
#             if basis == qubit_basis:
#                 measurements.append(qubit_value)
#             else:
#                 measurements.append(random.randint(0, 1))
#         else:
#             measurements.append(None)  # Lost qubit
    
#     logger.info(f"Measurements completed: {len(measurements)} qubits measured")
#     return measurements


# def adaptive_measurement(bob_bases, env_data):
#     """Adapt Bob's measurement settings based on channel conditions"""
#     logger.info("Adapting measurement settings based on environment...")
    
#     # Simple adaptation: if high QBER, adjust some bases
#     if env_data["QBER"] > 0.05:
#         for i in range(len(bob_bases)):
#             if random.random() < 0.1:  # Adjust 10% of bases
#                 bob_bases[i] = (bob_bases[i] + 1) % 3
    
#     return bob_bases

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


def measure_qubits(bob, method, transmitted_qubits):
    """
    Measure the received qubits on Bob's side using the selected encoding method.
    
    Args:
        bob: The receiver participant object
        method: The encoding method used
        transmitted_qubits: The qubits that Bob received from Alice
    
    Returns:
        tuple: (bob_bases, bob_measurements) - The randomly chosen bases and the measurement results
    """
    import random
    import logging
    logger = logging.getLogger()
    
    if method not in supported_encodings:
        logger.error(f"Unsupported encoding method for measurement: {method}")
        return None, None
    
    encoding_info = supported_encodings[method]
    dimension = encoding_info["dimension"]
    basis_sets = encoding_info["basis_sets"]
    num_bases = len(basis_sets)
    
    # Number of qubits received
    key_length = len(transmitted_qubits)
    logger.info(f"Bob measuring {key_length} qubits using {method} encoding")
    
    # Generate random measurement bases for Bob
    bob_bases = [random.randint(0, num_bases-1) for _ in range(key_length)]
    bob.bases = bob_bases  # Store the bases in Bob's object for later use
    
    # Perform measurements
    bob_measurements = []
    
    for i in range(key_length):
        qubit = transmitted_qubits[i]
        basis_idx = bob_bases[i]
        basis_name = basis_sets[basis_idx]
        
        # Measure the qubit in the chosen basis
        result = measure_in_basis(qubit, basis_name, method)
        bob_measurements.append(result)
    
    # Store measurements in Bob's object
    bob.measurements = bob_measurements
    
    logger.info(f"Bob completed measurements of {key_length} qubits with {method} encoding")
    return bob_bases, bob_measurements

def measure_in_basis(qubit, basis_name, method):
    """
    Measure a single qubit in the specified basis.
    
    Args:
        qubit: The quantum circuit or qubit to measure
        basis_name: The name of the measurement basis
        method: The encoding method used
    
    Returns:
        int: The measurement result (0 to dimension-1)
    """
    from qiskit_aer import Aer
    from qiskit import transpile
    
    # Get a copy of the qubit to measure
    if hasattr(qubit, 'copy'):
        qc = qubit.copy()
    else:
        # If it's already a QuantumCircuit, we can work with it directly
        qc = qubit
    
    # Apply appropriate basis transformation before measurement
    if method == "BB84":
        # BB84 protocol with two bases: computational (Z) and Hadamard (X)
        if basis_name == "X" or basis_name == "Hadamard":
            qc.h(0)  # Transform from X basis to Z basis for measurement
    
    elif method == "Six-State":
        # Six-State protocol with three bases: X, Y, Z
        if basis_name == "X":
            # Transform from X basis to Z basis
            qc.h(0)
        elif basis_name == "Y":
            # Transform from Y basis to Z basis
            qc.sdg(0)
            qc.h(0)
    
    # Check if measurement operations already exist in the circuit
    # by checking if there are any classical registers with measurements
    has_measurement = len(qc.clbits) > 0 and any(qc.get_instructions('measure'))
    
    # Add measurement if not already there
    if not has_measurement:
        qc.measure_all()
    
    # Execute the circuit
    backend = Aer.get_backend('qasm_simulator')
    qc = transpile(qc, backend)
    job = backend.run(qc, shots=1)
    result = job.result().get_counts()
    
    # Parse the measurement result
    result_key = list(result.keys())[0]
    # logger.debug(f"Raw measurement result: {result_key}")
    
    # Extract the first bit value regardless of format
    measured_bit = int(result_key.strip().split()[0])
    
    
    return measured_bit






# def reconcile_bases(classical_channel, alice, bob, bob_measurements,transmitted_qubits):
#     """Reconcile bases and perform key sifting"""
#     logger.info("Performing basis reconciliation and key sifting...")
    
#     # Send Bob's bases to Alice
#     bob_bases_msg = str(bob_bases)
#     classical_channel.send(bob_bases_msg)
    
#     # Alice identifies matching bases
#     matching_indices = [i for i in range(min(len(alice_bases), len(bob_bases))) 
#                        if alice_bases[i] == bob_bases[i] and i < len(measurements) and measurements[i] is not None]
    
#     # Alice sends matching indices to Bob
#     matching_indices_msg = str(matching_indices)
#     classical_channel.send(matching_indices_msg)
    
#     # Both extract the sifted key
#     sifted_key = []
#     for i in matching_indices:
#         if i < len(measurements) and measurements[i] is not None:
#             sifted_key.append(measurements[i])
    
#     # Calculate QBER using a portion of the sifted key
#     qber_check_size = min(100, len(sifted_key) // 10)
#     if qber_check_size > 0:
#         qber_check_indices = random.sample(range(len(sifted_key)), qber_check_size)
#         qber_check_indices.sort()
        
#         # Alice sends the bits at these positions
#         alice_check_bits = [measurements[matching_indices[i]] for i in qber_check_indices]
#         classical_channel.send(str(alice_check_bits))
        
#         # Bob compares with his bits
#         bob_check_bits = [sifted_key[i] for i in qber_check_indices]
#         error_count = sum(1 for a, b in zip(alice_check_bits, bob_check_bits) if a != b)
#         qber = error_count / qber_check_size
        
#         # Remove the check bits from the sifted key
#         sifted_key = [sifted_key[i] for i in range(len(sifted_key)) if i not in qber_check_indices]
#     else:
#         qber = 0.0
    
#     logger.info(f"Basis reconciliation complete. Sifted key length: {len(sifted_key)}, QBER: {qber:.4f}")
#     return sifted_key, qber




def reconcile_bases(classical_channel, alice, bob, bob_measurements, transmitted_qubits):
    """
    Reconcile bases between Alice and Bob and perform key sifting.
    
    Args:
        classical_channel: The classical communication channel
        alice: Alice participant object
        bob: Bob participant object
        bob_measurements: Measurement results from Bob
        transmitted_qubits: Qubits that were transmitted
        
    Returns:
        tuple: (sifted_key, qber) - The sifted key and quantum bit error rate
    """
    import random
    import logging
    logger = logging.getLogger()
    
    logger.info("Performing basis reconciliation and key sifting...")
    
    # Get Alice and Bob's bases
    alice_bases = alice.bases
    bob_bases = bob.bases
    
    # Send Bob's bases to Alice via classical channel
    bob_bases_msg = str(bob_bases)
    classical_channel.send(bob_bases_msg)
    
    # Alice identifies matching bases
    matching_indices = []
    for i in range(min(len(alice_bases), len(bob_bases))):
        if i < len(bob_measurements) and bob_measurements[i] is not None:
            if alice_bases[i] == bob_bases[i]:
                matching_indices.append(i)

    # print("Matching indices: ", matching_indices)
    
    # Alice sends matching indices to Bob
    matching_indices_msg = str(matching_indices)
    classical_channel.send(matching_indices_msg)
    
    # Both extract the sifted key
    bob_sifted_key = []
    alice_sifted_key = []

    for i in matching_indices:
        if i < len(bob_measurements) and bob_measurements[i] is not None:
            bob_sifted_key.append(bob_measurements[i])

    for i in matching_indices:
        if i < len(alice.bits):
            alice_sifted_key.append(alice.bits[i])

    # print("Bob sifted key: ", bob_sifted_key)
    # print("Alice sifted key: ", alice_sifted_key)
    
    min_length = min(len(alice_sifted_key), len(bob_sifted_key))
    alice_sifted_key = alice_sifted_key[:min_length]
    bob_sifted_key = bob_sifted_key[:min_length]

    # Find mismatched positions
    mismatched_positions = [i for i in range(min_length) if alice_sifted_key[i] != bob_sifted_key[i]]

    # print("Mismatched positions: ", mismatched_positions)
    print("Mismatched positions length: ", len(mismatched_positions))

    # Calculate QBER
    qber = len(mismatched_positions) / min_length

    # Calculate QBER using a portion of the sifted key
    # qber_check_size = min(100, len(sifted_key) // 10)
    # if qber_check_size > 0:
    #     qber_check_indices = random.sample(range(len(sifted_key)), qber_check_size)
    #     qber_check_indices.sort()
        
    #     # Alice gets the bits she originally sent at these positions
    #     alice_check_bits = []
    #     for i in qber_check_indices:
    #         original_index = matching_indices[i]
    #         if original_index < len(transmitted_qubits):
    #             # Get the original bit value from Alice's transmitted qubits
    #             # This might need adjustment based on how the original bit values are stored
    #             if hasattr(alice, 'sent_qubits') and alice.sent_qubits is not None:
    #                 alice_check_bits.append(alice.sent_qubits[original_index])
        
    #     classical_channel.send(str(alice_check_bits))
        
    #     # Bob compares with his bits
    #     bob_check_bits = [sifted_key[i] for i in qber_check_indices if i < len(sifted_key)]
        
    #     # Ensure we have matching lengths before comparing
    #     min_length = min(len(alice_check_bits), len(bob_check_bits))
    #     alice_check_bits = alice_check_bits[:min_length]
    #     bob_check_bits = bob_check_bits[:min_length]
        
    #     if min_length > 0:
    #         error_count = sum(1 for a, b in zip(alice_check_bits, bob_check_bits) if a != b)
    #         qber = error_count / min_length
    #     else:
    #         qber = 0.0
        
    #     # Remove the check bits from the sifted key
    #     new_sifted_key = []
    #     for i in range(len(sifted_key)):
    #         if i not in qber_check_indices:
    #             new_sifted_key.append(sifted_key[i])
    #     sifted_key = new_sifted_key
    # else:
    #     qber = 0.0
    
    # logger.info(f"Basis reconciliation complete. Sifted key length: {len(sifted_key)}, QBER: {qber:.4f}")
    
    # Store the sifted key in both Alice and Bob objects
    alice.sifted_key = alice_sifted_key
    bob.sifted_key = bob_sifted_key
    
    return alice_sifted_key, qber