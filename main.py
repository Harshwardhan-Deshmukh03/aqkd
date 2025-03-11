
import argparse
from channel_setup import setup_channels
from adaptive_encoding import analyze_environment, select_encoding, add_decoy_states, generate_random_bases
from quantum_transmission import prepare_qubits, transmit_qubits
from measurement import measure_qubits,reconcile_bases
from error_correction import cascade_correction
from privacy_amplification import adaptive_privacy_amplification
from key_verification import verify_key, UniversalHashFamily, calculate_key_hash_with_params
from utils.logger import setup_logger
from participants import create_participants
import random,json


def parse_arguments(args=None):
    parser = argparse.ArgumentParser(description='Adaptive Quantum Key Distribution System')
    parser.add_argument('--key-length', type=int, default=1024, help='Length of the quantum key')
    parser.add_argument('--decoy-states', action='store_true', help='Use decoy states')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    return parser.parse_args(args)

def match_encoding_methods(list1, list2):
    return list(set(list1) & set(list2))

def main(args=None):
    args = parse_arguments(args)
    logger = setup_logger(debug=args.debug)
    
    if logger is None:
        raise ValueError("Logger initialization failed!")
    
    logger.info("Starting AQKD protocol...")
    
    # Phase 1: Channel Setup and Authentication
    alice, bob = create_participants()
    print(str(alice.get_list()))
    
    
    supported_encoding_methods= match_encoding_methods(alice.encoding_supported,bob.encoding_supported)
    
    logger.info(f"Supported encoding methods: {supported_encoding_methods}")
    quantum_channel, classical_channel = setup_channels(alice,bob,supported_encoding_methods)
    print(str(quantum_channel.encoding))

    ### 

    # Phase 2: Environmental Analysis
    env_data=analyze_environment(quantum_channel,classical_channel,alice,bob,10)
    print(str(env_data))
    method, dimension, encoding_basis = select_encoding(env_data,supported_encoding_methods)

    logger.info(f"Selected encoding method: {method}  {dimension}D {encoding_basis}")
    
    # Phase 3: Quantum Data Transmission
    alice.bases,alice.bits, qubits = prepare_qubits(args.key_length, method)

    print(len(alice.bases))

    print(len(qubits))

    if args.decoy_states:
        qubits, decoy_pos = add_decoy_states(qubits)
        print(len(decoy_pos))


    transmitted_qubits =  transmit_qubits(quantum_channel, qubits, alice, bob)

    # print(str(transmitted_qubits[0]))
    
    # Phase 4: Measurement and Data Acquisition
    # bob_bases = generate_random_bases(args.key_length)
    bob_bases, bob_measurements = measure_qubits(bob, method, transmitted_qubits)
    print(len(bob_bases))
    # print(str(bob_measurements))
    print("Done")
    sifted_key, qber = reconcile_bases(classical_channel, alice, bob, bob_measurements,transmitted_qubits)
    logger.info(f"QBER: {qber:.4f}")
    


    # Phase 5: Error Correction
    corrected_key = cascade_correction(classical_channel,bob.sifted_key,alice.sifted_key, qber)
    bob.corrected_key = corrected_key
    alice.corrected_key = corrected_key
    print("Done correction=========================")

    # # Phase 6: Privacy amplification
    # final_key = adaptive_privacy_amplification(corrected_key, qber)


    # # Phase 7:  Key Verification
    # key_verified = verify_key(classical_channel, final_key)


    secure_seed = random.randint(0, 2**32 - 1)
    logger.info(f"Generated secure seed for privacy amplification: {secure_seed}")

    seed_data = {
        "type": "PRIVACY_AMPLIFICATION_SEED",
        "seed": secure_seed
    }
    classical_channel.send(json.dumps(seed_data))


    alice_final_key = adaptive_privacy_amplification(alice.corrected_key, qber, security_parameter=0.1, seed=secure_seed)
    bob_final_key = adaptive_privacy_amplification(bob.corrected_key, qber, security_parameter=0.1, seed=secure_seed)

    print(str(alice_final_key))
    print(str(bob_final_key))

    # Phase 7: Key Verification



# In main.py

# Phase 7: Key Verification
# Generate the universal hash function parameters
    uhash = UniversalHashFamily()
    a, b = uhash.select_function()  # Generate a and b in main

    # Send parameters to Bob via classical channel
    hash_params = {
        "type": "HASH_PARAMS",
        "a": str(a),
        "b": str(b)
    }
    classical_channel.send(json.dumps(hash_params))

    # Calculate Alice's hash
    alice_hash = calculate_key_hash_with_params(alice_final_key, a, b)

    # Bob would do the same on his side with the received parameters
    # Simulate receiving Bob's hash
    bob_hash_msg = classical_channel.receive(json.dumps({
        "type": "HASH_VALUE",
        "hash": str(calculate_key_hash_with_params(bob_final_key, a, b))
    }))
    bob_hash_data = json.loads(bob_hash_msg)
    bob_hash = int(bob_hash_data["hash"])

# Compare hashes in main
    key_verified = (alice_hash == bob_hash)
    logger.info(f"Key verification: {'Successful' if key_verified else 'Failed'}")
    key_verified = verify_key(classical_channel, alice_final_key, bob_final_key)

     
    print(str(bob_final_key))

    if key_verified:
        logger.info(f"AQKD protocol completed successfully. Key length: {len(bob_final_key)} bits")
        return bob_final_key
    else:
        logger.error("Key verification failed!")
        return None

if __name__ == "__main__":
    main()