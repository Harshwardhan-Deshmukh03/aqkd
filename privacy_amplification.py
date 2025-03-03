import numpy as np
import hashlib
from utils.logger import get_logger

logger = get_logger(__name__)

def privacy_amplification(corrected_key, qber):
    """Perform privacy amplification using Toeplitz matrix hashing"""
    logger.info("Starting privacy amplification...")
    
    if not corrected_key:
        logger.error("No key to amplify")
        return []
    
    # Calculate the amount of privacy amplification needed
    # based on QBER (higher QBER requires more compression)
    compression_factor = calculate_compression_factor(qber)
    final_key_length = int(len(corrected_key) * compression_factor)
    
    if final_key_length < 1:
        logger.error("QBER too high for effective privacy amplification")
        return []
    
    logger.info(f"Compression factor: {compression_factor:.4f}, Target length: {final_key_length}")
    
    # Generate a Toeplitz matrix for hashing
    toeplitz_matrix = create_toeplitz_matrix(len(corrected_key), final_key_length)
    
    # Convert key to numpy array for matrix multiplication
    key_array = np.array(corrected_key)
    
    # Apply Toeplitz matrix to the key (matrix multiplication mod 2)
    amplified_key = np.dot(toeplitz_matrix, key_array) % 2
    amplified_key = amplified_key.tolist()
    
    logger.info(f"Privacy amplification complete. Final key length: {len(amplified_key)}")
    return amplified_key

def calculate_compression_factor(qber):
    """Calculate the optimal compression factor based on QBER"""
    # Simple model: higher QBER requires more compression
    if qber < 0.01:
        return 0.8
    elif qber < 0.05:
        return 0.6
    elif qber < 0.1:
        return 0.4
    else:
        return 0.2

# def create_toeplitz_matrix(input_length, output_length):
#     """Create a Toeplitz matrix for hashing"""
#     # Generate random bits for the Toeplitz matrix
#     # A Toeplitz matrix has constant diagonals
#     random_bits = np.random.randint(0, 2, input_length + output_length - 1)
    
#     toeplitz_matrix = np.zeros((output_length, input_length), dtype=int)
#     for i in range(output_length):
#         toeplitz_matrix[i, :] = random_bits[input_length-1-i:2*input_length-1-i]
    
#     return toeplitz_matrix

### for the above we have fixed the bug of the mismatched indices in the topelitz matrices generation
### the correct code is as follows:

def create_toeplitz_matrix(input_length, output_length):
    """Create a Toeplitz matrix for hashing"""
    # Generate random bits for the Toeplitz matrix
    # A Toeplitz matrix has constant diagonals
    random_bits = np.random.randint(0, 2, input_length + output_length - 1)
    
    toeplitz_matrix = np.zeros((output_length, input_length), dtype=int)
    for i in range(output_length):
        toeplitz_matrix[i, :] = random_bits[i:i + input_length]
    
    return toeplitz_matrix

def universal_hash(key, seed):
    """Apply a universal hash function to the key"""
    hash_input = ''.join(map(str, key)) + str(seed)
    hash_output = hashlib.sha256(hash_input.encode()).digest()
    
    # Convert hash to binary
    binary_hash = []
    for byte in hash_output:
        for bit in format(byte, '08b'):
            binary_hash.append(int(bit))
    
    return binary_hash[:len(key)//2]  # Reduce key length by half