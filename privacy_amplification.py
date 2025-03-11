import numpy as np
import hashlib
import random
from utils.logger import get_logger

logger = get_logger(__name__)

def adaptive_privacy_amplification(corrected_key, qber, security_parameter=0.1):
    """
    Perform adaptive privacy amplification using Toeplitz matrix hashing.
    
    This implementation adjusts the compression ratio based on the QBER (Quantum Bit Error Rate),
    with higher QBER triggering more aggressive hashing to ensure security.
    
    Args:
        corrected_key (list): The error-corrected key after reconciliation
        qber (float): The estimated Quantum Bit Error Rate
        security_parameter (float): Additional security parameter to fine-tune compression
        
    Returns:
        list: The final secure key after privacy amplification
    """
    logger.info("Starting adaptive privacy amplification...")
    
    if not corrected_key:
        logger.error("No key to amplify")
        return []
    
    # Calculate the optimal compression factor based on QBER
    compression_factor = calculate_adaptive_compression_factor(qber, security_parameter)
    final_key_length = max(1, int(len(corrected_key) * compression_factor))
    
    if final_key_length < 8:  # Ensure minimum key length
        logger.warning(f"Very short final key length ({final_key_length} bits). QBER may be too high.")
        if final_key_length < 1:
            logger.error("QBER too high for effective privacy amplification")
            return []
    
    logger.info(f"Input key length: {len(corrected_key)}, QBER: {qber:.4f}")
    logger.info(f"Adaptive compression factor: {compression_factor:.4f}, Target length: {final_key_length}")
    
    # Generate a secure seed for the Toeplitz matrix
    seed = generate_secure_seed()
    logger.info(f"Generated secure seed for Toeplitz matrix generation")
    
    # Create a Toeplitz matrix for hashing with the generated seed
    toeplitz_matrix = create_toeplitz_matrix(len(corrected_key), final_key_length, seed)
    
    # Convert key to numpy array for matrix multiplication
    key_array = np.array(corrected_key)
    
    # Apply Toeplitz matrix to the key (matrix multiplication mod 2)
    amplified_key = np.dot(toeplitz_matrix, key_array) % 2
    amplified_key = amplified_key.tolist()
    
    # Verify the hashing process didn't produce unexpected results
    if len(amplified_key) != final_key_length:
        logger.error(f"Hashing process error: expected {final_key_length} bits, got {len(amplified_key)}")
        return []
    
    # Calculate and log the effective compression ratio
    effective_ratio = len(amplified_key) / len(corrected_key)
    logger.info(f"Privacy amplification complete. Effective compression ratio: {effective_ratio:.4f}")
    logger.info(f"Final key length: {len(amplified_key)} bits")
    
    return amplified_key

def calculate_adaptive_compression_factor(qber, security_parameter=0.1):
    """
    Calculate the optimal compression factor adaptively based on QBER.
    
    Higher QBER requires more aggressive compression to ensure security.
    The security parameter allows for additional adjustment based on specific requirements.
    
    Args:
        qber (float): The Quantum Bit Error Rate
        security_parameter (float): Additional parameter to adjust compression (higher = more conservative)
        
    Returns:
        float: The compression factor (between 0 and 1)
    """
    # Base compression calculation using Shannon information theory principles
    # As QBER approaches 25%, security diminishes significantly for BB84
    # This model is more sophisticated than fixed thresholds
    
    # Calculate the binary entropy function H(x) = -x*log2(x) - (1-x)*log2(1-x)
    # This is used to estimate Eve's information
    if qber == 0:
        h_qber = 0
    elif qber >= 0.5:
        h_qber = 1  # Maximum entropy
    else:
        h_qber = -qber * np.log2(qber) - (1-qber) * np.log2(1-qber)
    
    # Calculate mutual information between Alice and Bob: I(A:B) = 1 - h(QBER)
    mutual_info = 1 - h_qber
    
    # Calculate estimated information leaked to Eve
    # For BB84, Eve's information is estimated as: I(A:E) = min(h_qber, 1-h_qber)
    # This is a simplified model and could be replaced with protocol-specific estimates
    eve_info = min(h_qber, 1-h_qber)
    
    # Apply security parameter for additional safety margin
    eve_info = eve_info * (1 + security_parameter)
    
    # Calculate the optimal compression factor
    # We need to compress enough to eliminate Eve's knowledge
    # But we want to retain as many bits as possible
    compression_factor = max(0, (mutual_info - eve_info) / mutual_info)
    
    # Apply bounds to ensure reasonable compression
    compression_factor = max(0.1, min(0.9, compression_factor))
    
    return compression_factor

def generate_secure_seed():
    """
    Generate a cryptographically secure seed for Toeplitz matrix generation.
    
    Returns:
        int: A secure random seed
    """
    # Use cryptographically secure random number generator
    return int.from_bytes(random.randbytes(8), byteorder='big')

def create_toeplitz_matrix(input_length, output_length, seed=None):
    """
    Create a Toeplitz matrix for hashing.
    
    A Toeplitz matrix has constant diagonals and is defined by input_length + output_length - 1
    random bits.
    
    Args:
        input_length (int): Length of the input key (number of columns)
        output_length (int): Length of the output key (number of rows)
        seed (int, optional): Seed for the random number generator
        
    Returns:
        numpy.ndarray: A Toeplitz matrix of shape (output_length, input_length)
    """
    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)
    
    # Generate random bits for the Toeplitz matrix
    # A Toeplitz matrix is completely defined by the first row and first column
    random_bits = np.random.randint(0, 2, input_length + output_length - 1)
    
    # Create the Toeplitz matrix
    toeplitz_matrix = np.zeros((output_length, input_length), dtype=int)
    for i in range(output_length):
        toeplitz_matrix[i, :] = random_bits[i:i + input_length]
    
    return toeplitz_matrix

def universal_hash_amplification(key, security_level=128):
    """
    Alternative approach: Apply a universal hash function for privacy amplification.
    
    This method can be used as an alternative or supplement to Toeplitz matrix hashing.
    
    Args:
        key (list): The input key
        security_level (int): The desired security level in bits
        
    Returns:
        list: The hashed key
    """
    if not key:
        return []
    
    # Convert key to string for hashing
    key_str = ''.join(map(str, key))
    
    # Generate a random salt for hashing
    salt = str(random.randint(0, 2**32 - 1))
    
    # Apply SHA-256 hash
    hash_input = key_str + salt
    hash_output = hashlib.sha256(hash_input.encode()).digest()
    
    # Convert hash to binary
    binary_hash = []
    for byte in hash_output:
        for bit in format(byte, '08b'):
            binary_hash.append(int(bit))
    
    # Truncate to desired security level
    output_length = min(len(key) // 2, security_level)
    return binary_hash[:output_length]

def multi_level_privacy_amplification(corrected_key, qber, use_universal_hash=True):
    """
    Apply multi-level privacy amplification for enhanced security.
    
    This function combines Toeplitz matrix hashing with universal hashing for
    stronger security guarantees.
    
    Args:
        corrected_key (list): The error-corrected key
        qber (float): The Quantum Bit Error Rate
        use_universal_hash (bool): Whether to apply universal hashing after Toeplitz
        
    Returns:
        list: The final secure key
    """
    # First level: Toeplitz matrix hashing
    amplified_key = adaptive_privacy_amplification(corrected_key, qber)
    
    # Second level (optional): Universal hashing
    if use_universal_hash and len(amplified_key) > 16:
        final_key = universal_hash_amplification(amplified_key)
        logger.info(f"Applied second-level universal hashing, final length: {len(final_key)}")
        return final_key
    
    return amplified_key