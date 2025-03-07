
# import argparse
# from channel_setup import setup_channels
# from adaptive_encoding import analyze_environment, select_encoding, add_decoy_states, generate_random_bases
# from quantum_transmission import prepare_qubits, transmit_qubits
# from measurement import perform_measurements, reconcile_bases
# from error_correction import cascade_correction
# from privacy_amplification import privacy_amplification
# from key_verification import verify_key
# from utils.logger import setup_logger

# def parse_arguments(args=None):
#     parser = argparse.ArgumentParser(description='Adaptive Quantum Key Distribution System')
#     parser.add_argument('--key-length', type=int, default=1024, help='Length of the quantum key')
#     parser.add_argument('--decoy-states', action='store_true', help='Use decoy states')
#     parser.add_argument('--debug', action='store_true', help='Enable debug logging')
#     parser.add_argument('--qber', type=float, default=0.03, help='QBER value for testing')
#     return parser.parse_args(args)

# def main(args=None):
#     args = parse_arguments(args)
#     logger = setup_logger(debug=args.debug)
    
#     if logger is None:
#         raise ValueError("Logger initialization failed!")
    
#     logger.info("Starting AQKD protocol...")
    
#     # Phase 1: Channel Setup and Authentication
#     quantum_channel, classical_channel = setup_channels()
    
#     # Phase 2: Environmental Analysis
#     env_data = analyze_environment(quantum_channel)
#     dimension, encoding_basis = select_encoding(env_data)
    
#     # Phase 3: Quantum Data Transmission
#     alice_bases, qubits = prepare_qubits(args.key_length, dimension, encoding_basis)
#     if args.decoy_states:
#         qubits = add_decoy_states(qubits)
#     transmit_qubits(quantum_channel, qubits)
    
#     # Phase 4: Measurement and Data Acquisition
#     bob_bases = generate_random_bases(args.key_length)
#     measurements = perform_measurements(quantum_channel, bob_bases)
    
#     # Phase 5: Basis Reconciliation and Key Sifting
#     sifted_key, qber = reconcile_bases(classical_channel, alice_bases, bob_bases, measurements)
#     logger.info(f"QBER: {qber:.4f}")
    
#     # Phase 6: Error Correction
#     corrected_key = cascade_correction(classical_channel, sifted_key, qber)
    
#     # Phase 7: Privacy Amplification and Key Verification
#     final_key = privacy_amplification(corrected_key, qber)
#     key_verified = verify_key(classical_channel, final_key)
    
#     if key_verified:
#         logger.info(f"AQKD protocol completed successfully. Key length: {len(final_key)} bits")
#         return final_key
#     else:
#         logger.error("Key verification failed!")
#         return None

# if __name__ == "__main__":
#     main()

import argparse
from channel_setup import setup_channels
from adaptive_encoding import analyze_environment, select_encoding, add_decoy_states, generate_random_bases
from quantum_transmission import prepare_qubits, transmit_qubits
from measurement import perform_measurements, reconcile_bases
from error_correction import cascade_correction
from privacy_amplification import privacy_amplification
from key_verification import verify_key
from utils.logger import setup_logger
from typing import Dict, Optional, Union
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def parse_arguments(args=None):
    parser = argparse.ArgumentParser(description='Adaptive Quantum Key Distribution System')
    parser.add_argument('--key-length', type=int, default=1024, help='Length of the quantum key')
    parser.add_argument('--decoy-states', action='store_true', help='Use decoy states')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--qber', type=float, default=0.03, help='QBER value for testing')
    parser.add_argument('--analysis-mode', action='store_true', help='Run in analysis mode')

    if args is None:
        return parser.parse_args()
    elif isinstance(args, dict):
        arg_list = []
        for key, value in args.items():
            if isinstance(value, bool):
                if value:
                    arg_list.append(f'--{key.replace("_", "-")}')
            else:
                arg_list.append(f'--{key.replace("_", "-")}')
                arg_list.append(str(value))
        return parser.parse_args(arg_list)
    return parser.parse_args(args)

def run_protocol(args, logger) -> Dict[str, Union[bool, float, int, str]]:
    """
    Run the QKD protocol and return performance metrics
    """
    metrics = {
        'success': False,
        'qber': 0.0,
        'initial_length': args.key_length,
        'final_length': 0,
        'key_rate': 0.0,
        'encoding_scheme': None,
        'final_key': None
    }

    try:
        # Phase 1: Channel Setup and Authentication
        quantum_channel, classical_channel = setup_channels()
        if not quantum_channel or not classical_channel:
            logger.error("Channel setup failed")
            return metrics

        # Phase 2: Environmental Analysis
        env_data = analyze_environment(quantum_channel)
        dimension, encoding_basis = select_encoding(env_data)
        metrics['encoding_scheme'] = encoding_basis

        # Phase 3: Quantum Data Transmission
        alice_bases, qubits = prepare_qubits(args.key_length, dimension, encoding_basis)
        if args.decoy_states:
            qubits = add_decoy_states(qubits)
        transmitted_qubits = transmit_qubits(quantum_channel, qubits)

        # Phase 4: Measurement
        bob_bases = generate_random_bases(args.key_length)
        measurements = perform_measurements(quantum_channel, bob_bases)

        # Phase 5: Reconciliation
        sifted_key, qber = reconcile_bases(classical_channel, alice_bases, bob_bases, measurements)
        metrics['qber'] = qber
        logger.info(f"QBER: {qber:.4f}")

        # Phase 6: Error Correction
        corrected_key = cascade_correction(classical_channel, sifted_key, qber)

        # Phase 7: Privacy Amplification and Verification
        final_key = privacy_amplification(corrected_key, qber)
        key_verified = verify_key(classical_channel, final_key)

        if key_verified and final_key:
            metrics.update({
                'success': True,
                'final_length': len(final_key),
                'key_rate': len(final_key) / args.key_length,
                'final_key': final_key
            })
            logger.info(f"Protocol successful. Final key length: {len(final_key)}")
        
        return metrics

    except Exception as e:
        logger.error(f"Protocol failed: {str(e)}")
        return metrics

def main(args=None) -> Optional[Union[list, Dict]]:
    if isinstance(args, argparse.Namespace):
        args = vars(args)
    args = parse_arguments(args)
    logger = setup_logger(debug=args.debug)
    
    if logger is None:
        raise ValueError("Logger initialization failed!")
    
    logger.info("Starting AQKD protocol...")
    
    # Run protocol
    metrics = run_protocol(args, logger)
    
    if args.analysis_mode:
        return metrics
    else:
        if metrics['success']:
            return metrics['final_key']
        return None

def get_theoretical_key_rate(qber: float) -> float:
    """Calculate theoretical key rate"""
    if qber >= 0.11:  # BB84 threshold
        return 0.0
    return 1 - 2 * binary_entropy(qber)

def binary_entropy(p: float) -> float:
    """Calculate binary entropy"""
    if p == 0 or p == 1:
        return 0.0
    return -p * np.log2(p) - (1-p) * np.log2(1-p)

def run_qber_analysis(qber_range=None, num_trials=5):
    """Run analysis across different QBER values"""
    if qber_range is None:
        qber_range = np.arange(0, 0.15, 0.01)  # 0 to 15% QBER
        
    results = {
        'qber': [],
        'key_rate': [],
        'theoretical_rate': [],
        'encoding_scheme': [],
        'success_rate': [],
        'final_lengths': []
    }
    
    print("Running QBER analysis...")
    for qber in tqdm(qber_range):
        trial_results = []
        successful_trials = 0
        
        for _ in range(num_trials):
            args = {
                'key_length': 1024,
                'qber': qber,
                'analysis_mode': True
            }
            metrics = main(args)
            
            if metrics['success']:
                trial_results.append(metrics)
                successful_trials += 1
        
        if trial_results:
            avg_rate = np.mean([r['key_rate'] for r in trial_results])
            avg_length = np.mean([r['final_length'] for r in trial_results])
            success_rate = successful_trials / num_trials
            common_encoding = max(set([r['encoding_scheme'] for r in trial_results]), 
                                key=lambda x: trial_results.count({'encoding_scheme': x}))
            
            results['qber'].append(qber)
            results['key_rate'].append(avg_rate)
            results['theoretical_rate'].append(get_theoretical_key_rate(qber))
            results['encoding_scheme'].append(common_encoding)
            results['success_rate'].append(success_rate)
            results['final_lengths'].append(avg_length)
    
    return results

def plot_key_rate_analysis(results):
    """Plot key rate vs QBER with theoretical bound"""
    plt.figure(figsize=(12, 8))
    
    # Plot experimental results
    plt.plot(results['qber'], results['key_rate'], 'bo-', 
             label='Experimental', linewidth=2, markersize=8)
    
    # Plot theoretical bound
    plt.plot(results['qber'], results['theoretical_rate'], 'r--', 
             label='Theoretical Bound', linewidth=2)
    
    # Add encoding scheme transitions
    schemes = np.array(results['encoding_scheme'])
    unique_schemes = np.unique(schemes)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_schemes)))
    
    for scheme, color in zip(unique_schemes, colors):
        mask = schemes == scheme
        if any(mask):
            plt.plot(np.array(results['qber'])[mask], 
                    np.array(results['key_rate'])[mask], 
                    'o-', color=color, label=f'Scheme: {scheme}')
    
    plt.xlabel('QBER', fontsize=12)
    plt.ylabel('Key Rate', fontsize=12)
    plt.title('QKD Performance: Key Rate vs QBER', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    
    # Add annotations
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.axvline(x=0.11, color='r', linestyle=':', alpha=0.5, 
                label='BB84 Threshold')
    
    plt.tight_layout()
    plt.savefig('qkd_performance.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Plot success rate
    plt.figure(figsize=(12, 8))
    plt.plot(results['qber'], results['success_rate'], 'go-', 
             label='Success Rate', linewidth=2)
    plt.xlabel('QBER', fontsize=12)
    plt.ylabel('Success Rate', fontsize=12)
    plt.title('Protocol Success Rate vs QBER', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig('success_rate.png', dpi=300, bbox_inches='tight')
    plt.show()

def analyze_and_plot():
    """Run analysis and create plots"""
    # Run analysis with finer QBER resolution
    qber_range = np.arange(0, 0.15, 0.005)  # Smaller steps for smoother plots
    results = run_qber_analysis(qber_range=qber_range, num_trials=5)
    
    # Plot results
    plot_key_rate_analysis(results)
    
    # Print summary statistics
    print("\nPerformance Summary:")
    print(f"Average Key Rate: {np.mean(results['key_rate']):.4f}")
    print(f"Maximum QBER with positive key rate: {max(results['qber']):.4f}")
    print(f"Average Success Rate: {np.mean(results['success_rate']):.4f}")
    
    # Save results
    np.save('qkd_performance_results.npy', results)
    
    return results

if __name__ == "__main__":
    args = parse_arguments()
    
    if args.analysis_mode:
        # Run analysis and plotting
        results = analyze_and_plot()
    else:
        # Run normal protocol
        result = main(args)
        if result:
            print(f"Protocol completed successfully. Final key length: {len(result)}")
        else:
            print("Protocol failed")