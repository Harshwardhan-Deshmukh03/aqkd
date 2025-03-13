import argparse
import time
import random
import json
import os
from channel_setup import setup_channels
from adaptive_encoding import analyze_environment, select_encoding, add_decoy_states, generate_random_bases
from quantum_transmission import prepare_qubits, transmit_qubits
from measurement import measure_qubits, reconcile_bases
from error_correction import cascade_correction
from privacy_amplification import adaptive_privacy_amplification
from key_verification import verify_key, UniversalHashFamily, calculate_key_hash_with_params
from utils.logger import setup_logger
from participants import create_participants

# File for temporary run results
TEMP_RESULTS_FILE = "temp_results.json"

def parse_arguments(args=None):
    parser = argparse.ArgumentParser(description='Adaptive Quantum Key Distribution System')
    parser.add_argument('--key-length', type=int, default=1024, help='Length of the quantum key')
    parser.add_argument('--decoy-states', action='store_true', help='Use decoy states')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--mitm', action='store_true', help='Simulate a man-in-the-middle attack with higher QBER')
    
    return parser.parse_args(args)

def match_encoding_methods(list1, list2):
    return list(set(list1) & set(list2))

def main(args=None):
    # Initialize results for this run
    results_log = {}
    start_time = time.time()
    phase_times = {}
    
    try:
        args = parse_arguments(args)
        logger = setup_logger(debug=args.debug)
        
        if logger is None:
            raise ValueError("Logger initialization failed!")
        
        # Record the configuration
        results_log['key_length'] = args.key_length
        results_log['decoy_states'] = args.decoy_states
        results_log['mitm'] = args.mitm
        
        logger.info("Starting AQKD protocol...")
        
        # Phase 1: Channel Setup and Authentication
        phase_start = time.time()
        alice, bob = create_participants()
        supported_encoding_methods = match_encoding_methods(alice.encoding_supported, bob.encoding_supported)
        logger.info(f"Supported encoding methods: {supported_encoding_methods}")
        quantum_channel, classical_channel = setup_channels(alice, bob, supported_encoding_methods)
        phase_times['setup'] = time.time() - phase_start
        
        # Phase 2: Environmental Analysis
        phase_start = time.time()
        env_data = analyze_environment(quantum_channel, classical_channel, alice, bob, 128)
        method, dimension, encoding_basis = select_encoding(env_data, supported_encoding_methods)
        logger.info(f"Selected encoding method: {method} {dimension}D {encoding_basis}")
        phase_times['env_analysis'] = time.time() - phase_start
        
        # Log encoding information
        results_log['encoding_method'] = method
        results_log['encoding_dimension'] = dimension
        results_log['encoding_basis'] = str(encoding_basis)
        
        # Phase 3: Quantum Data Transmission
        phase_start = time.time()
        alice.bases, alice.bits, qubits = prepare_qubits(args.key_length, method)
        
        if args.decoy_states:
            qubits, decoy_pos = add_decoy_states(qubits)
            results_log['decoy_count'] = len(decoy_pos)
        
        transmitted_qubits = transmit_qubits(quantum_channel, qubits, alice, bob, args.mitm)
        phase_times['transmission'] = time.time() - phase_start
        
        # Phase 4: Measurement and Data Acquisition
        phase_start = time.time()
        bob_bases, bob_measurements = measure_qubits(bob, method, transmitted_qubits)
        sifted_key, qber = reconcile_bases(classical_channel, alice, bob, bob_measurements, transmitted_qubits)
        logger.info(f"QBER: {qber:.4f}")
        phase_times['measurement'] = time.time() - phase_start
        
        # Log sifted key length
        results_log['sifted_key_length'] = len(alice.sifted_key) if hasattr(alice, 'sifted_key') else 0
        results_log['qber'] = qber
        
        # Save intermediate results
        save_temp_results(results_log)
        
        # Phase 5: Error Correction
        phase_start = time.time()
        corrected_key = cascade_correction(classical_channel, bob.sifted_key, alice.sifted_key, qber)
        bob.corrected_key = corrected_key
        alice.corrected_key = corrected_key
        phase_times['error_correction'] = time.time() - phase_start
        
        # Log corrected key length
        results_log['corrected_key_length'] = len(corrected_key)
        save_temp_results(results_log)
        
        # Generate secure seed for privacy amplification
        secure_seed = random.randint(0, 2**32 - 1)
        logger.info(f"Generated secure seed for privacy amplification: {secure_seed}")
        
        seed_data = {
            "type": "PRIVACY_AMPLIFICATION_SEED",
            "seed": secure_seed
        }
        classical_channel.send(json.dumps(seed_data))
        
        # Phase 6: Privacy Amplification
        phase_start = time.time()
        alice_final_key = adaptive_privacy_amplification(alice.corrected_key, qber, security_parameter=0.1, seed=secure_seed)
        bob_final_key = adaptive_privacy_amplification(bob.corrected_key, qber, security_parameter=0.1, seed=secure_seed)
        phase_times['privacy_amplification'] = time.time() - phase_start
        
        # Log final key length
        results_log['final_key_length'] = len(bob_final_key)
        save_temp_results(results_log)
        
        # Phase 7: Key Verification
        phase_start = time.time()
        uhash = UniversalHashFamily()
        a, b = uhash.select_function()
        
        hash_params = {
            "type": "HASH_PARAMS",
            "a": str(a),
            "b": str(b)
        }
        classical_channel.send(json.dumps(hash_params))
        
        alice_hash = calculate_key_hash_with_params(alice_final_key, a, b)
        
        bob_hash_msg = classical_channel.receive(json.dumps({
            "type": "HASH_VALUE",
            "hash": str(calculate_key_hash_with_params(bob_final_key, a, b))
        }))
        bob_hash_data = json.loads(bob_hash_msg)
        bob_hash = int(bob_hash_data["hash"])
        
        key_verified = (alice_hash == bob_hash)
        print(str(bob_final_key))
        logger.info(f"Key verification: {'Successful' if key_verified else 'Failed'}")
        phase_times['verification'] = time.time() - phase_start
        
        # Log verification success
        results_log['verification_success'] = "Success" if key_verified else "Failed"
        
        # Log phase times
        results_log['phase_times'] = phase_times
        results_log['total_time'] = time.time() - start_time
        
        # Additional environmental metrics
        if isinstance(env_data, dict):
            results_log['noise_level'] = env_data.get('noise_level', 'N/A')
            results_log['coherence_time'] = env_data.get('coherence_time', 'N/A')
        
        # Final save of results
        save_temp_results(results_log)
        
        if key_verified:
            logger.info(f"AQKD protocol completed successfully. Key length: {len(bob_final_key)} bits")
            return qber
        else:
            logger.error("Key verification failed!")
            return None
            
    except Exception as e:
        # Log error information
        results_log['error'] = str(e)
        results_log['total_time'] = time.time() - start_time
        save_temp_results(results_log)
        raise

def save_temp_results(results):
    """Save results to temporary file for the table generator"""
    try:
        with open(TEMP_RESULTS_FILE, 'w') as f:
            json.dump(results, f, indent=2)
    except Exception as e:
        print(f"Error saving temporary results: {e}")

if __name__ == "__main__":
    main()