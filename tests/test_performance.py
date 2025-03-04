# test_performance.py

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from channel_setup import QuantumChannel, ClassicalChannel
from main import main  # Your main QKD protocol implementation

class QKDPerformanceAnalyzer:
    def __init__(self):
        self.qber_range = np.arange(0, 0.15, 0.01)  # 0 to 15% QBER
        self.results = {}
        
    def run_qber_analysis(self, num_trials: int = 5) -> Dict:
        """Run QKD protocol with different QBER values"""
        results = {
            'qber': [],
            'key_rate': [],
            'final_key_length': [],
            'encoding_scheme': []
        }
        
        for qber in self.qber_range:
            trial_results = []
            for _ in range(num_trials):
                # Initialize channels with specific QBER
                quantum_channel = QuantumChannel(error_rate=qber)
                classical_channel = ClassicalChannel()
                
                # Run protocol
                initial_length = 1024  # Initial number of qubits
                final_key = main(['--key-length', str(initial_length), '--qber', str(qber)])
                
                if final_key:
                    key_rate = len(final_key) / initial_length
                    trial_results.append({
                        'key_rate': key_rate,
                        'final_length': len(final_key),
                        'encoding': quantum_channel.current_scheme.value
                    })
            
            # Average results across trials
            if trial_results:
                avg_rate = np.mean([r['key_rate'] for r in trial_results])
                avg_length = np.mean([r['final_length'] for r in trial_results])
                common_encoding = max(set([r['encoding'] for r in trial_results]), 
                                   key=lambda x: trial_results.count({'encoding': x}))
                
                results['qber'].append(qber)
                results['key_rate'].append(avg_rate)
                results['final_key_length'].append(avg_length)
                results['encoding_scheme'].append(common_encoding)
        
        self.results = results
        return results

    def plot_key_rate_vs_qber(self):
        """Plot key rate vs QBER for different encoding schemes"""
        plt.figure(figsize=(10, 6))
        
        # Plot points with different colors for different encoding schemes
        schemes = set(self.results['encoding_scheme'])
        colors = plt.cm.rainbow(np.linspace(0, 1, len(schemes)))
        
        for scheme, color in zip(schemes, colors):
            mask = [s == scheme for s in self.results['encoding_scheme']]
            qber_values = [self.results['qber'][i] for i in range(len(mask)) if mask[i]]
            rate_values = [self.results['key_rate'][i] for i in range(len(mask)) if mask[i]]
            
            plt.plot(qber_values, rate_values, 'o-', 
                    label=scheme, color=color, markersize=8)
        
        plt.xlabel('QBER')
        plt.ylabel('Key Rate')
        plt.title('QKD Performance: Key Rate vs QBER')
        plt.grid(True)
        plt.legend()
        plt.savefig('qkd_performance.png')
        plt.show()

    def plot_theoretical_comparison(self):
        """Plot comparison with theoretical bounds"""
        plt.figure(figsize=(10, 6))
        
        # Plot actual results
        plt.plot(self.results['qber'], self.results['key_rate'], 
                'bo-', label='Experimental', markersize=8)
        
        # Plot theoretical bounds
        qber = np.array(self.results['qber'])
        theoretical_rate = 1 - 2 * self.binary_entropy(qber)
        plt.plot(qber, theoretical_rate, 'r--', 
                label='Theoretical Bound', linewidth=2)
        
        plt.xlabel('QBER')
        plt.ylabel('Key Rate')
        plt.title('QKD Performance: Experimental vs Theoretical')
        plt.grid(True)
        plt.legend()
        plt.savefig('theoretical_comparison.png')
        plt.show()

    @staticmethod
    def binary_entropy(p):
        """Calculate binary entropy"""
        return -p * np.log2(p) - (1-p) * np.log2(1-p)

def main_analysis():
    analyzer = QKDPerformanceAnalyzer()
    
    # Run analysis
    print("Running QKD performance analysis...")
    analyzer.run_qber_analysis()
    
    # Generate plots
    print("Generating performance plots...")
    analyzer.plot_key_rate_vs_qber()
    analyzer.plot_theoretical_comparison()
    
    # Print summary statistics
    print("\nPerformance Summary:")
    print(f"Average Key Rate: {np.mean(analyzer.results['key_rate']):.4f}")
    print(f"Maximum QBER with positive key rate: {max(analyzer.results['qber']):.4f}")
    
    # Save results
    np.save('qkd_performance_results.npy', analyzer.results)

if __name__ == "__main__":
    main_analysis()