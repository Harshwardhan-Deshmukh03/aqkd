
import argparse
from channel_setup import setup_channels
from adaptive_encoding import analyze_environment, select_encoding, add_decoy_states, generate_random_bases
from quantum_transmission import prepare_qubits, transmit_qubits
from measurement import perform_measurements, reconcile_bases
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

def main(args=None):
    args = parse_arguments(args)
    logger = setup_logger(debug=args.debug)
    
    if logger is None:
        raise ValueError("Logger initialization failed!")
    
    logger.info("Starting AQKD protocol...")
    
    # Phase 1: Channel Setup and Authentication
    alice, bob = create_participants()
    
    quantum_channel, classical_channel = setup_channels(alice,bob)
    
    # Phase 2: Environmental Analysis
    env_data = analyze_environment(quantum_channel)
    dimension, encoding_basis = select_encoding(env_data)
    
    # Phase 3: Quantum Data Transmission
    alice_bases, qubits = prepare_qubits(args.key_length, dimension, encoding_basis)
    if args.decoy_states:
        qubits = add_decoy_states(qubits)
    transmit_qubits(quantum_channel, qubits)
    
    # Phase 4: Measurement and Data Acquisition
    bob_bases = generate_random_bases(args.key_length)
    measurements = perform_measurements(quantum_channel, bob_bases)
    
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