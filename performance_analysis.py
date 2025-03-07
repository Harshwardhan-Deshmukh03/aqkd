# performance_analysis.py

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from channel_setup import QuantumChannel, ClassicalChannel
import argparse
from utils.logger import setup_logger

class QKDPerformanceAnalyzer:
    def __init__(self):
        self.qber_range = np.arange(0, 0.15, 0.01)  # 0 to 15% QBER
        self.results = {
            'qber': [],
            'key_rate': [],
            'final_key_length': [],
            'encoding_scheme': []
        }
        self.logger = setup_logger(debug=False)

    def run_single_test(self, qber: float, key_length: int = 1024) -> Dict:
        """Run a single QKD test with specified QBER"""
        from main import main  # Import here to avoid circular import
        
        args = ['--key-length', str(key_length), '--qber', str(qber)]
        result = main(args)
        
        if result:
            return {
                'success': True,
                'key_rate': len(result) / key_length,
                'final_length': len(result)
            }
        return {'success': False}

    def run_qber_analysis(self, num_trials: int = 5) -> Dict:
        """Run QKD protocol with different QBER values"""
        for qber in self.qber_range:
            self.logger.info(f"Testing QBER: {qber:.3f}")
            trial_results = []
            
            for trial in range(num_trials):
                result = self.run_single_test(qber)
                if result['success']:
                    trial_results.append(result)
            
            if trial_results:
                avg_rate = np.mean([r['key_rate'] for r in trial_results])
                avg_length = np.mean([r['final_length'] for r in trial_results])
                
                self.results['qber'].append(qber)
                self.results['key_rate'].append(avg_rate)
                self.results['final_key_length'].append(avg_length)
        
        return self.results

    def plot_key_rate_vs_qber(self):
        """Plot key rate vs QBER"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.results['qber'], self.results['key_rate'], 'bo-', label='Experimental')
        
        # Plot theoretical bound
        qber = np.array(self.results['qber'])
        theoretical_rate = 1 - 2 * self.binary_entropy(qber)
        plt.plot(qber, theoretical_rate, 'r--', label='Theoretical Bound')
        
        plt.xlabel('QBER')
        plt.ylabel('Key Rate')
        plt.title('QKD Performance: Key Rate vs QBER')
        plt.grid(True)
        plt.legend()
        plt.savefig('qkd_performance.png')
        plt.show()

    @staticmethod
    def binary_entropy(p):
        """Calculate binary entropy"""
        return -p * np.log2(p) - (1-p) * np.log2(1-p)

def main_analysis():
    analyzer = QKDPerformanceAnalyzer()
    print("Running QKD performance analysis...")
    results = analyzer.run_qber_analysis()
    analyzer.plot_key_rate_vs_qber()
    np.save('qkd_performance_results.npy', results)

if __name__ == "__main__":
    main_analysis()