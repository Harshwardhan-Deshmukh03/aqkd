
import argparse
from channel_setup import setup_channels
from adaptive_encoding import analyze_environment, select_encoding, add_decoy_states, generate_random_bases
from quantum_transmission import prepare_qubits, transmit_qubits
from measurement import measure_qubits
from error_correction import cascade_correction
from privacy_amplification import privacy_amplification
from key_verification import verify_key
from utils.logger import setup_logger
from participants import create_participants


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
    env_data=analyze_environment(quantum_channel,classical_channel,alice,bob,100)
    print(str(env_data))
    method, dimension, encoding_basis = select_encoding(env_data,supported_encoding_methods)

    logger.info(f"Selected encoding method: {method}  {dimension}D {encoding_basis}")
    
    # Phase 3: Quantum Data Transmission
    alice.bases, qubits = prepare_qubits(args.key_length, method)

    print(len(alice.bases))

    print(len(qubits))

    if args.decoy_states:
        qubits, decoy_pos = add_decoy_states(qubits)
        print(len(decoy_pos))


    transmitted_qubits =  transmit_qubits(quantum_channel, qubits, alice, bob)

    # print(str(transmitted_qubits[0]))
    
    # Phase 4: Measurement and Data Acquisition
    # bob_bases = generate_random_bases(args.key_length)
    bob_bases, measurements = measure_qubits(bob, method, transmitted_qubits)
    print(len(bob_bases))
    print(len(measurements))
    print("Done")
    
    # Phase 5: Basis Reconciliation and Key Sifting
    sifted_key, qber = reconcile_bases(classical_channel, alice_bases, bob_bases, measurements)
    logger.info(f"QBER: {qber:.4f}")
    
    # Phase 6: Error Correction
    corrected_key = cascade_correction(classical_channel, sifted_key, qber)
    
    # Phase 7: Privacy Amplification and Key Verification
    final_key = privacy_amplification(corrected_key, qber)
    key_verified = verify_key(classical_channel, final_key)
    
    if key_verified:
        logger.info(f"AQKD protocol completed successfully. Key length: {len(final_key)} bits")
        return final_key
    else:
        logger.error("Key verification failed!")
        return None

if __name__ == "__main__":
    main()