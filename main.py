import argparse
import time
import random
import json
import os
from channel_setup import setup_channels
from adaptive_encoding import analyze_environment, select_encoding, add_decoy_states
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
    parser.add_argument('--block-size', type=int, default=128, help='Size of each block')
    parser.add_argument('--test-size', type=int, default=32, help='Number of test qubits per analysis')
    parser.add_argument('--decoy-states', action='store_true', help='Use decoy states')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--mitm', action='store_true', help='Simulate a man-in-the-middle attack with higher QBER')
    return parser.parse_args(args)

def match_encoding_methods(list1, list2):
    return list(set(list1) & set(list2))

# NEW: Function to determine action based on QBER and encoding changes
def determine_action(qber, prev_method, new_method, dimension, new_dimension, encoding_basis, new_encoding_basis):
    if qber >= 0.90:
        return "Abort"
    elif qber >= 0.85:
        return "Retransmit"
    elif (new_method != prev_method or new_dimension != dimension or new_encoding_basis != encoding_basis):
        return "Switch Encoding"
    else:
        return "Continue"

def main(args=None):
    results_log = {}
    start_time = time.time()
    phase_times = {}
    
    try:
        args = parse_arguments(args)
        logger = setup_logger(debug=args.debug)
        if logger is None:
            raise ValueError("Logger initialization failed!")
        
        results_log['key_length'] = args.key_length
        results_log['block_size'] = args.block_size
        results_log['test_size'] = args.test_size
        results_log['decoy_states'] = args.decoy_states
        results_log['mitm'] = args.mitm
        logger.info("Starting AQKD protocol...")
        
        phase_start = time.time()
        alice, bob = create_participants()
        supported_encoding_methods = match_encoding_methods(alice.encoding_supported, bob.encoding_supported)
        logger.info(f"Supported encoding methods: {supported_encoding_methods}")
        quantum_channel, classical_channel = setup_channels(alice, bob, supported_encoding_methods)
        phase_times['setup'] = time.time() - phase_start
        
        phase_start = time.time()
        env_data = analyze_environment(quantum_channel, classical_channel, alice, bob, args.test_size)
        method, dimension, encoding_basis = select_encoding(env_data, supported_encoding_methods)
        logger.info(f"Initial encoding method: {method} {dimension}D {encoding_basis}")
        phase_times['env_analysis'] = time.time() - phase_start
        
        results_log['encoding_method'] = method
        results_log['encoding_dimension'] = dimension
        results_log['encoding_basis'] = str(encoding_basis)
        
        alice.bases, alice.bits, qubits = prepare_qubits(args.key_length, method)
        num_blocks = args.key_length // args.block_size
        alice_blocks_bases = [alice.bases[i * args.block_size:(i + 1) * args.block_size] for i in range(num_blocks)]
        alice_blocks_bits = [alice.bits[i * args.block_size:(i + 1) * args.block_size] for i in range(num_blocks)]
        alice_blocks_qubits = [qubits[i * args.block_size:(i + 1) * args.block_size] for i in range(num_blocks)]
        bob_blocks = []
        all_qbers = []
        
        # NEW: Initialize block-wise logging
        block_logs = []
        
        phase_start = time.time()
        prev_method, prev_dimension, prev_encoding_basis = method, dimension, encoding_basis
        for block_idx in range(num_blocks):
            logger.info(f"Processing block {block_idx + 1}/{num_blocks}...")
            
            alice.bases = alice_blocks_bases[block_idx]
            alice.bits = alice_blocks_bits[block_idx]
            block_qubits = alice_blocks_qubits[block_idx]
            
            if args.decoy_states:
                block_qubits, decoy_pos = add_decoy_states(block_qubits)
                results_log[f'block_{block_idx}_decoy_count'] = len(decoy_pos)
            
            transmitted_qubits = transmit_qubits(quantum_channel, block_qubits, alice, bob, args.mitm)
            
            bob_bases, bob_measurements = measure_qubits(bob, method, transmitted_qubits)
            sifted_key, qber = reconcile_bases(classical_channel, alice, bob, bob_measurements, transmitted_qubits)
            bob_blocks.append(bob.sifted_key)
            all_qbers.append(qber)
            logger.info(f"Block {block_idx + 1} QBER: {qber:.4f}")
            
            # NEW: Log block data
            encoding_method_str = f"{method} ({dimension}D, {encoding_basis})"
            if qber >= 0.80:
                block_logs.append({
                    'block_number': block_idx + 1,
                    'qber': qber,
                    'encoding_method': "N/A",
                    'action': "Abort"
                })
                break
            
            block_logs.append({
                'block_number': block_idx + 1,
                'qber': qber,
                'encoding_method': encoding_method_str,
                'action': determine_action(qber, prev_method, method, prev_dimension, dimension, prev_encoding_basis, encoding_basis)
            })
            
            if block_idx < num_blocks - 1:
                env_data = analyze_environment(quantum_channel, classical_channel, alice, bob, args.test_size)
                new_method, new_dimension, new_encoding_basis = select_encoding(env_data, supported_encoding_methods)
                if new_method != method or new_dimension != dimension or new_encoding_basis != encoding_basis:
                    logger.info(f"Updating encoding for block {block_idx + 2}: {new_method} {new_dimension}D {new_encoding_basis}")
                    method, dimension, encoding_basis = new_method, new_dimension, new_encoding_basis
                    next_block_idx = block_idx + 1
                    alice_blocks_bases[next_block_idx], alice_blocks_bits[next_block_idx], alice_blocks_qubits[next_block_idx] = prepare_qubits(args.block_size, method)
                
                prev_method, prev_dimension, prev_encoding_basis = method, dimension, encoding_basis
        
        phase_times['transmission_measurement'] = time.time() - phase_start
        results_log['block_logs'] = block_logs  # NEW: Save block-wise data
        
        alice_full_sifted_key = []
        bob_full_sifted_key = []
        for alice_block_bits, bob_block in zip(alice_blocks_bits, bob_blocks):
            alice_full_sifted_key.extend(alice_block_bits[:len(bob_block)])
            bob_full_sifted_key.extend(bob_block)
        alice.sifted_key = alice_full_sifted_key
        bob.sifted_key = bob_full_sifted_key
        results_log['sifted_key_length'] = len(alice.sifted_key)
        results_log['avg_qber'] = sum(all_qbers) / len(all_qbers) if all_qbers else 0
        
        phase_start = time.time()
        corrected_key = cascade_correction(classical_channel, bob.sifted_key, alice.sifted_key, results_log['avg_qber'])
        bob.corrected_key = corrected_key
        alice.corrected_key = corrected_key
        phase_times['error_correction'] = time.time() - phase_start
        results_log['corrected_key_length'] = len(corrected_key)
        
        phase_start = time.time()
        secure_seed = random.randint(0, 2**32 - 1)
        logger.info(f"Generated secure seed for privacy amplification: {secure_seed}")
        seed_data = {"type": "PRIVACY_AMPLIFICATION_SEED", "seed": secure_seed}
        classical_channel.send(json.dumps(seed_data))
        alice_final_key = adaptive_privacy_amplification(alice.corrected_key, results_log['avg_qber'], security_parameter=0.1, seed=secure_seed)
        bob_final_key = adaptive_privacy_amplification(bob.corrected_key, results_log['avg_qber'], security_parameter=0.1, seed=secure_seed)
        phase_times['privacy_amplification'] = time.time() - phase_start
        results_log['final_key_length'] = len(bob_final_key)
        
        phase_start = time.time()
        uhash = UniversalHashFamily()
        a, b = uhash.select_function()
        hash_params = {"type": "HASH_PARAMS", "a": str(a), "b": str(b)}
        classical_channel.send(json.dumps(hash_params))
        alice_hash = calculate_key_hash_with_params(alice_final_key, a, b)
        bob_hash_msg = classical_channel.receive(json.dumps({"type": "HASH_VALUE", "hash": str(calculate_key_hash_with_params(bob_final_key, a, b))}))
        bob_hash_data = json.loads(bob_hash_msg)
        bob_hash = int(bob_hash_data["hash"])
        key_verified = (alice_hash == bob_hash)
        logger.info(f"Key verification: {'Successful' if key_verified else 'Failed'}")
        phase_times['verification'] = time.time() - phase_start
        results_log['verification_success'] = "Success" if key_verified else "Failed"
        
        results_log['phase_times'] = phase_times
        results_log['total_time'] = time.time() - start_time
        if isinstance(env_data, dict):
            results_log['noise_level'] = env_data.get('noise_level', 'N/A')
            results_log['coherence_time'] = env_data.get('coherence_time', 'N/A')
        save_temp_results(results_log)
        
        if key_verified:
            logger.info(f"AQKD protocol completed successfully. Final key length: {len(bob_final_key)} bits")
            return results_log['avg_qber']
        else:
            logger.error("Key verification failed!")
            return None
            
    except Exception as e:
        results_log['error'] = str(e)
        results_log['total_time'] = time.time() - start_time
        save_temp_results(results_log)
        raise

def save_temp_results(results):
    try:
        with open(TEMP_RESULTS_FILE, 'w') as f:
            json.dump(results, f, indent=2)
    except Exception as e:
        print(f"Error saving temporary results: {e}")

if __name__ == "__main__":
    main()