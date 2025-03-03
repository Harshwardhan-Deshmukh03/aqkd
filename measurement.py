import random
from utils.logger import get_logger

logger = get_logger(__name__)

def perform_measurements(quantum_channel, bob_bases):
    """Perform measurements on the received qubits"""
    logger.info(f"Bob performing measurements with {len(bob_bases)} bases...")
    
    # In a real system, this would be receiving and measuring actual qubits
    # For simulation, we'll assume the qubits are already in the quantum_channel
    received_qubits = quantum_channel.received_qubits
    
    if not received_qubits:
        logger.error("No qubits received for measurement")
        return []
    
    measurements = []
    for i, basis in enumerate(bob_bases):
        if i < len(received_qubits) and received_qubits[i] is not None:
            # Simulating measurement
            # If Bob's basis matches the qubit's basis, he'll get the correct result
            # Otherwise, random result with 50% probability
            qubit_basis = received_qubits[i][1]
            qubit_value = received_qubits[i][0]
            
            if basis == qubit_basis:
                measurements.append(qubit_value)
            else:
                measurements.append(random.randint(0, 1))
        else:
            measurements.append(None)  # Lost qubit
    
    logger.info(f"Measurements completed: {len(measurements)} qubits measured")
    return measurements

def reconcile_bases(classical_channel, alice_bases, bob_bases, measurements):
    """Reconcile bases and perform key sifting"""
    logger.info("Performing basis reconciliation and key sifting...")
    
    # Send Bob's bases to Alice
    bob_bases_msg = str(bob_bases)
    classical_channel.send(bob_bases_msg)
    
    # Alice identifies matching bases
    matching_indices = [i for i in range(min(len(alice_bases), len(bob_bases))) 
                       if alice_bases[i] == bob_bases[i] and i < len(measurements) and measurements[i] is not None]
    
    # Alice sends matching indices to Bob
    matching_indices_msg = str(matching_indices)
    classical_channel.send(matching_indices_msg)
    
    # Both extract the sifted key
    sifted_key = []
    for i in matching_indices:
        if i < len(measurements) and measurements[i] is not None:
            sifted_key.append(measurements[i])
    
    # Calculate QBER using a portion of the sifted key
    qber_check_size = min(100, len(sifted_key) // 10)
    if qber_check_size > 0:
        qber_check_indices = random.sample(range(len(sifted_key)), qber_check_size)
        qber_check_indices.sort()
        
        # Alice sends the bits at these positions
        alice_check_bits = [measurements[matching_indices[i]] for i in qber_check_indices]
        classical_channel.send(str(alice_check_bits))
        
        # Bob compares with his bits
        bob_check_bits = [sifted_key[i] for i in qber_check_indices]
        error_count = sum(1 for a, b in zip(alice_check_bits, bob_check_bits) if a != b)
        qber = error_count / qber_check_size
        
        # Remove the check bits from the sifted key
        sifted_key = [sifted_key[i] for i in range(len(sifted_key)) if i not in qber_check_indices]
    else:
        qber = 0.0
    
    logger.info(f"Basis reconciliation complete. Sifted key length: {len(sifted_key)}, QBER: {qber:.4f}")
    return sifted_key, qber

def adaptive_measurement(bob_bases, env_data):
    """Adapt Bob's measurement settings based on channel conditions"""
    logger.info("Adapting measurement settings based on environment...")
    
    # Simple adaptation: if high QBER, adjust some bases
    if env_data["QBER"] > 0.05:
        for i in range(len(bob_bases)):
            if random.random() < 0.1:  # Adjust 10% of bases
                bob_bases[i] = (bob_bases[i] + 1) % 3
    
    return bob_bases