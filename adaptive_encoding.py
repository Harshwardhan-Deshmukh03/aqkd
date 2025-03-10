from enum import Enum
import random
import numpy as np
from utils.logger import get_logger

logger = get_logger(__name__)

class AdaptiveEncoder:
    def __init__(self):
        self.ann = SimpleANN()
        self.encoding_options = {
            'low_noise': {'dimension': 4, 'basis': 'Computational'},
            'medium_noise': {'dimension': 8, 'basis': 'Fourier'},
            'high_noise': {'dimension': 16, 'basis': 'Custom'}
        }
    
    def select_encoding(self, env_data):
        encoding_params = self.ann.predict(env_data)
        return encoding_params

class SimpleANN:
    def __init__(self):
        # Simple weights for demonstration
        self.weights = np.random.rand(3, 3)
    
    def predict(self, env_data):
        inputs = np.array([
            env_data['QBER'],
            env_data['channel_loss'],
            env_data['noise_levels']
        ])
        
        outputs = np.dot(inputs, self.weights)
        encoding_type = np.argmax(outputs)
        
        if encoding_type == 0:
            return 4, "Computational"
        elif encoding_type == 1:
            return 8, "Fourier"
        else:
            return 16, "Custom"

# def analyze_environment(quantum_channel, sample_size=100):
#     logger.info("Analyzing environment and channel conditions...")
    
#     # Generate and send test qubits
#     test_qubits = [(random.randint(0, 1), random.randint(0, 2)) for _ in range(sample_size)]
#     received_qubits = quantum_channel.transmit(test_qubits)
    
#     # Calculate environment parameters
#     loss_count = received_qubits.count(None)
#     channel_loss = loss_count / sample_size
    
#     # Count errors in non-lost qubits
#     error_count = sum(1 for i, qubit in enumerate(received_qubits) 
#                      if qubit is not None and qubit[0] != test_qubits[i][0])
#     valid_qubits = sample_size - loss_count
#     qber = error_count / valid_qubits if valid_qubits > 0 else 0
    
#     # Estimate noise levels
#     noise_levels = qber * 2  # Simplified model
    
#     env_data = {
#         "QBER": qber,
#         "channel_loss": channel_loss,
#         "noise_levels": noise_levels
#     }
    
#     logger.info(f"Environment analysis - QBER: {qber:.4f}, Loss: {channel_loss:.4f}, Noise: {noise_levels:.4f}")
#     return env_data


        

import logging
# Configure logger
logger = logging.getLogger(__name__)

# Dictionary of encoding parameters instead of classes
supported_encodings = {
    "BB84": {
        "dimension": 2,
        "basis_sets": ["computational", "hadamard"],
        "error_tolerance": 0.11,
        "max_distance": 50
    },
    "SIX_STATE": {
        "dimension": 2,
        "basis_sets": ["computational", "hadamard", "circular"],
        "error_tolerance": 0.126,
        "max_distance": 40
    },
    "EIGHT_STATE": {
        "dimension": 2,
        "basis_sets": ["computational", "hadamard", "circular", "custom"],
        "error_tolerance": 0.15,
        "max_distance": 30
    },
    "DECOY_BB84": {
        "dimension": 2,
        "basis_sets": ["computational", "hadamard"],
        "error_tolerance": 0.11,
        "max_distance": 100,
        "recommended_intensity": 0.5
    },
    "E91": {
        "dimension": 2,
        "basis_sets": ["bell_basis"],
        "error_tolerance": 0.15,
        "max_distance": 100,
        "uses_entanglement": True
    }
}

def select_encoding(ava_encodings):
    model_output = ['DECOY_BB84', 'E91', 'EIGHT_STATE', 'SIX_STATE', 'BB84']
    available_encodings = ava_encodings
    
    # Find the intersection between model_output and available_encodings
    common_encodings = list(set(model_output) & set(available_encodings))
    
    if not common_encodings:
        # Handle the case when there are no common encodings
        logger.warning("No common encoding schemes found")
        return None, None
    
    # Take the first common encoding
    selected_encoding = common_encodings[0]
    
    # Get encoding parameters for the selected scheme
    encoding_params = supported_encodings[selected_encoding]
    
    # Extract dimension and basis sets
    dimension = encoding_params["dimension"]
    basis_sets = encoding_params["basis_sets"]
    
    logger.info(f"Selected encoding: {selected_encoding} with dimension {dimension} and basis sets {basis_sets}")
    
    return dimension, basis_sets

    # encoder = AdaptiveEncoder()
    # dimension, basis = encoder.select_encoding(env_data)
    # logger.info(f"Selected encoding - Dimension: {dimension}, Basis: {basis}")
    # return dimension, basis

def generate_random_bases(length):
    return [random.choice([0, 1, 2]) for _ in range(length)]

def add_decoy_states(qubits, decoy_ratio=0.1):
    decoy_positions = []
    for i in range(len(qubits)):
        if random.random() < decoy_ratio:
            decoy_positions.append(i)
            # Replace with decoy state (using different intensity)
            qubits[i] = (qubits[i][0], qubits[i][1], 'decoy')
    
    return qubits





from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
import random

def generate_random_key(length):
    return [random.randint(0, 1) for _ in range(length)]

def generate_random_bases(length):
    return [random.randint(0, 1) for _ in range(length)]

def encode_qubits(key, bases):
    circuits = []
    for k, b in zip(key, bases):
        qc = QuantumCircuit(1, 1)
        if k == 1:
            qc.x(0)  # Apply X-gate to represent 1
        if b == 1:
            qc.h(0)  # Apply H-gate for diagonal basis
        circuits.append(qc)
    return circuits

def measure_qubits(circuits, bob_bases):
    backend = Aer.get_backend('aer_simulator')
    results = []
    for qc, basis in zip(circuits, bob_bases):
        if basis == 1:
            qc.h(0)  # Apply H-gate for diagonal basis measurement
        qc.measure(0, 0)  # Add measurement operation
        qc = transpile(qc, backend)  # Transpile the circuit for the Aer simulator
        job = backend.run(qc, shots=1)
        result = job.result().get_counts()
        measured_bit = int(list(result.keys())[0])  # Extract measured bit
        results.append(measured_bit)
    return results


def calculate_qber(alice_key, bob_key):
    if len(alice_key) == 0:
        return 0  # Avoid division by zero
    errors = sum(a != b for a, b in zip(alice_key, bob_key))
    return errors / len(alice_key)

def analyze_environment(quantum_channel, classical_channel, alice, bob, sample_size):
    # Import the necessary noise modules at the top of your file
    from qiskit_aer.noise import NoiseModel, depolarizing_error, amplitude_damping_error
    from enum import Enum
    
    # Create noise model for the quantum channel
    noise_model = NoiseModel()
    
    # Configure noise parameters
    p_meas = 0.05  # measurement error probability
    p_gate1 = 0.02  # 1-qubit gate error probability
    gamma = 0.03    # amplitude damping parameter (for channel loss)
    
    # Add measurement error
    error_meas = depolarizing_error(p_meas, 1)
    noise_model.add_all_qubit_quantum_error(error_meas, "measure")
    
    # Add gate errors - keep this as in your original working code
    error_gate1 = depolarizing_error(p_gate1, 1)
    noise_model.add_all_qubit_quantum_error(error_gate1, ["x", "h"])
    
    # Step 1: Alice generates a random key and bases
    alice_key = generate_random_key(sample_size)
    alice_bases = generate_random_bases(sample_size)

    # Step 2: Alice encodes the key into qubits
    alice_qubits = encode_qubits(alice_key, alice_bases)

    # Step 3: Bob selects random bases to measure
    bob_bases = generate_random_bases(sample_size)
    
    # Modified measure_qubits function that uses noise model inline
    backend = Aer.get_backend('aer_simulator')
    bob_measurements = []
    
    # Run a separate simulation to estimate channel loss
    loss_counts = 0
    damping_model = NoiseModel()
    damping_error = amplitude_damping_error(gamma)
    damping_model.add_all_qubit_quantum_error(damping_error, ["x", "h"])
    
    # Step 4: Measure qubits with noise (exactly as in your original working code)
    for qc, basis in zip(alice_qubits, bob_bases):
        if basis == 1:
            qc.h(0)
        qc.measure(0, 0)
        qc = transpile(qc, backend)
        
        # Run with noise model
        job = backend.run(qc, shots=1, noise_model=noise_model)
        result = job.result().get_counts()
        measured_bit = int(list(result.keys())[0])
        bob_measurements.append(measured_bit)

    # Estimate channel loss in a separate calculation (not affecting the measurement results)
    # This is a simplified simulation of channel loss based on gamma parameter
    channel_loss = 1 - (1 - gamma)**5  # Simplified loss model based on damping parameter
    
    # Step 5: Alice and Bob compare bases publicly
    matching_indices = [i for i in range(sample_size) if alice_bases[i] == bob_bases[i]]
    alice_shared_key = [alice_key[i] for i in matching_indices]
    bob_shared_key = [bob_measurements[i] for i in matching_indices]

    # Step 6: Calculate QBER
    qber = calculate_qber(alice_shared_key, bob_shared_key)
    
    # Calculate noise levels (simplified model)
    noise_levels = p_gate1 + p_meas  # Sum of noise parameters
    
    # Create environment data dictionary
    env_data = {
        "QBER": qber,
        "channel_loss": channel_loss,
        "noise_levels": noise_levels,
        "p_gate": p_gate1,
        "p_measurement": p_meas,
        "key_length": len(alice_shared_key)
    }
    
    # Print the results
    print(f"Quantum Bit Error Rate (QBER): {qber:.4f}")
    print(f"Channel loss: {channel_loss:.4f}")
    print(f"Noise levels: {noise_levels:.4f}")
    print(f"Final key length: {len(alice_shared_key)}")

    return env_data