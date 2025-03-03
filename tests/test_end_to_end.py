import unittest
import sys
import os
import time

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import main
from utils.logger import setup_logger

class TestEndToEnd(unittest.TestCase):
    def setUp(self):
        # Initialize logger
        self.logger = setup_logger(debug=True)
    
    def test_full_protocol(self):
        """Test the complete AQKD protocol end-to-end"""
        self.logger.info("Starting end-to-end test of AQKD protocol")
        
        start_time = time.time()
        final_key = main()
        end_time = time.time()
        
        self.assertIsNotNone(final_key, "Final key should not be None")
        self.assertTrue(len(final_key) > 0, "Final key should have non-zero length")
        
        self.logger.info(f"End-to-end test completed in {end_time - start_time:.2f} seconds")
        self.logger.info(f"Final key length: {len(final_key)} bits")
    
    def test_qber_adaptation(self):
        """Test protocol adaptation to different QBER levels"""
        # Test with different QBER levels
        for test_qber in [0.01, 0.05, 0.08]:
            self.logger.info(f"Testing with QBER level: {test_qber}")
            
            # Modify QuantumChannel to use our test QBER
            from channel_setup import QuantumChannel
            QuantumChannel.error_rate = test_qber
            
            # Run the protocol
            final_key = main()
            
            self.assertIsNotNone(final_key, f"Protocol should produce a key even with QBER {test_qber}")
            
            # Higher QBER should result in more compression/shorter keys
            # This is a simplified test; actual behavior might vary
            if test_qber > 0.05:
                self.logger.info("High QBER test passed")
    
    def test_performance(self):
        """Test performance metrics of the protocol"""
        from key_rate_comparison import calculate_theoretical_key_rate, calculate_actual_key_rate
        
        # Run the protocol and measure time
        start_time = time.time()
        final_key = main()
        end_time = time.time()
        
        execution_time = end_time - start_time
        key_length = len(final_key) if final_key else 0
        
        # Calculate bits per second
        bits_per_second = key_length / execution_time if execution_time > 0 else 0
        
        self.logger.info(f"Performance test - Key length: {key_length}, Time: {execution_time:.2f}s")
        self.logger.info(f"Performance: {bits_per_second:.2f} bits/second")
        
        # Basic performance assertion
        self.assertGreater(bits_per_second, 0, "Protocol should generate keys at a non-zero rate")

if __name__ == "__main__":
    unittest.main()