import unittest
import sys
import os
import random
# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from channel_setup import setup_channels
from adaptive_encoding import analyze_environment, select_encoding
from quantum_transmission import prepare_qubits, transmit_qubits
from measurement import perform_measurements, reconcile_bases

class TestBasicFunctionality(unittest.TestCase):
    def setUp(self):
        # Initialize channels for testing
        self.quantum_channel, self.classical_channel = setup_channels()
    
    def test_channel_setup(self):
        """Test channel setup and authentication"""
        self.assertIsNotNone(self.quantum_channel)
        self.assertIsNotNone(self.classical_channel)
        self.assertTrue(self.classical_channel.authenticated)
    
    def test_environment_analysis(self):
        """Test environment analysis"""
        env_data = analyze_environment(self.quantum_channel)
        self.assertIn("QBER", env_data)
        self.assertIn("channel_loss", env_data)
        self.assertIn("noise_levels", env_data)
        
        # Check valid ranges for environmental parameters
        self.assertGreaterEqual(env_data["QBER"], 0)
        self.assertLessEqual(env_data["QBER"], 1)
        self.assertGreaterEqual(env_data["channel_loss"], 0)
        self.assertLessEqual(env_data["channel_loss"], 1)
    
    def test_adaptive_encoding(self):
        """Test adaptive encoding selection"""
        env_data = {
            "QBER": 0.03,
            "channel_loss": 0.1,
            "noise_levels": 0.05
        }
        dimension, basis = select_encoding(env_data)
        self.assertGreater(dimension, 0)
        self.assertIn(basis, ["Computational", "Fourier", "Custom"])
    
    def test_prepare_qubits(self):
        """Test qubit preparation"""
        dimension = 4
        basis = "Computational"
        key_length = 100
        
        alice_bases, qubits = prepare_qubits(key_length, dimension, basis)
        
        self.assertEqual(len(alice_bases), key_length)
        self.assertEqual(len(qubits), key_length)
        
        for base in alice_bases:
            self.assertLess(base, dimension)
        
        for qubit in qubits:
            self.assertIn(qubit[0], [0, 1])  # Check valid bit values
    
    def test_transmit_and_measure(self):
        """Test qubit transmission and measurement"""
        # Prepare qubits
        dimension = 4
        basis = "Computational"
        key_length = 100
        
        alice_bases, qubits = prepare_qubits(key_length, dimension, basis)
        
        # Simulate transmission
        self.quantum_channel.received_qubits = qubits  # For testing
        
        # Simulate Bob's measurements
        bob_bases = [random.randint(0, dimension-1) for _ in range(key_length)]
        measurements = perform_measurements(self.quantum_channel, bob_bases)
        
        # Check measurements
        self.assertEqual(len(measurements), key_length)
        
        # Test basis reconciliation
        sifted_key, qber = reconcile_bases(self.classical_channel, alice_bases, bob_bases, measurements)
        
        self.assertGreaterEqual(qber, 0)
        self.assertLessEqual(qber, 1)

if __name__ == "__main__":
    unittest.main()