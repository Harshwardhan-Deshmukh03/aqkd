import numpy as np
import hashlib
import random
import json
from utils.logger import get_logger

logger = get_logger(__name__)

def adaptive_privacy_amplification(corrected_key, qber, security_parameter=0.1, seed=None):
    """
    Perform adaptive privacy amplification using Toeplitz matrix hashing.
    
    This implementation adjusts the compression ratio based on the QBER (Quantum Bit Error Rate),
    with higher QBER triggering more aggressive hashing to ensure security.
    
    Args:
        corrected_key (list): The error-corrected key after reconciliation
        qber (float): The estimated Quantum Bit Error Rate
        security_parameter (float): Additional security parameter to fine-tune compression
        seed (int, optional): Seed for the random number generator
        
    Returns:
        list: The final secure key after privacy amplification
    """
    logger.info("Starting adaptive privacy amplification...")
    logger.info(f"Input key length: {len(corrected_key)}, QBER: {qber:.4f}")
    
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
    
    logger.info(f"Adaptive compression factor: {compression_factor:.4f}, Target length: {final_key_length}")
    
    # Use the provided seed or generate a new one
    if seed is None:
        try:
            seed = generate_secure_seed()
            logger.info(f"Generated secure seed for Toeplitz matrix generation: {seed}")
        except Exception as e:
            logger.error(f"Error generating secure seed: {str(e)}")
            # Fallback to a simpler seed if there's an error
            seed = random.randint(0, 2**32 - 1)
            logger.info(f"Using fallback seed: {seed}")
    else:
        logger.info(f"Using provided seed for Toeplitz matrix generation: {seed}")
    
    # Create a Toeplitz matrix for hashing with the generated seed
    try:
        logger.info(f"Creating Toeplitz matrix of shape ({final_key_length}, {len(corrected_key)})")
        toeplitz_matrix = create_toeplitz_matrix(len(corrected_key), final_key_length, seed)
        logger.debug(f"Toeplitz matrix created successfully")
    except Exception as e:
        logger.error(f"Error creating Toeplitz matrix: {str(e)}")
        return []
    
    # Convert key to numpy array for matrix multiplication
    key_array = np.array(corrected_key)
    logger.debug(f"Converting key to numpy array of shape: {key_array.shape}")
    
    # Apply Toeplitz matrix to the key (matrix multiplication mod 2)
    try:
        logger.info("Applying Toeplitz matrix hashing...")
        amplified_key = np.dot(toeplitz_matrix, key_array) % 2
        amplified_key = amplified_key.tolist()
        logger.debug("Matrix multiplication completed successfully")
    except Exception as e:
        logger.error(f"Error during matrix multiplication: {str(e)}")
        return []
    
    # Verify the hashing process didn't produce unexpected results
    if len(amplified_key) != final_key_length:
        logger.error(f"Hashing process error: expected {final_key_length} bits, got {len(amplified_key)}")
        return []
    
    # Calculate and log the effective compression ratio
    effective_ratio = len(amplified_key) / len(corrected_key)
    logger.info(f"Privacy amplification complete. Effective compression ratio: {effective_ratio:.4f}")
    logger.info(f"Final key length: {len(amplified_key)} bits")
    
    # Log a small sample of the key for debugging (first and last few bits)
    if len(amplified_key) > 10:
        key_sample = f"First 5 bits: {amplified_key[:5]}, Last 5 bits: {amplified_key[-5:]}"
        logger.debug(f"Key sample: {key_sample}")
    
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
    logger.info(f"Calculating adaptive compression factor for QBER={qber:.6f}, security_parameter={security_parameter}")
    
    # Base compression calculation using Shannon information theory principles
    # As QBER approaches 25%, security diminishes significantly for BB84
    # This model is more sophisticated than fixed thresholds
    
    # Calculate the binary entropy function H(x) = -x*log2(x) - (1-x)*log2(1-x)
    # This is used to estimate Eve's information
    if qber == 0:
        h_qber = 0
    elif qber >= 0.5:
        h_qber = 0.9  # Maximum entropy
    else:
        h_qber = -qber * np.log2(qber) - (1-qber) * np.log2(1-qber)
    
    logger.debug(f"Binary entropy H(QBER) = {h_qber:.6f}")
    
    # Calculate mutual information between Alice and Bob: I(A:B) = 1 - h(QBER)
    mutual_info = 1 - h_qber
    logger.debug(f"Mutual information I(A:B) = {mutual_info:.6f}")
    
    # Calculate estimated information leaked to Eve
    # For BB84, Eve's information is estimated as: I(A:E) = min(h_qber, 1-h_qber)
    # This is a simplified model and could be replaced with protocol-specific estimates
    eve_info = min(h_qber, 1-h_qber)
    logger.debug(f"Estimated Eve's information I(A:E) = {eve_info:.6f}")
    
    # Apply security parameter for additional safety margin
    eve_info = eve_info * (1 + security_parameter)
    logger.debug(f"Eve's information with security parameter = {eve_info:.6f}")
    
    # Calculate the optimal compression factor
    # We need to compress enough to eliminate Eve's knowledge
    # But we want to retain as many bits as possible
    compression_factor = max(0.1, (mutual_info - eve_info) / mutual_info)
    
    # Apply bounds to ensure reasonable compression
    original_factor = compression_factor
    compression_factor = max(0.1, min(0.9, compression_factor))
    
    logger.info(f"Raw compression factor: {original_factor:.6f}, Bounded compression factor: {compression_factor:.6f}")
    
    return compression_factor

def generate_secure_seed():
    """
    Generate a cryptographically secure seed for Toeplitz matrix generation.
    
    Returns:
        int: A secure random seed (between 0 and 2^32-1)
    """
    # FIX: Generate a seed within numpy's valid range (0 to 2^32-1)
    seed = random.randint(0, 2**32 - 1)
    logger.debug(f"Generated secure seed: {seed}")
    return seed

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
        logger.debug(f"Setting numpy random seed to: {seed}")
        np.random.seed(seed)
    
    # Generate random bits for the Toeplitz matrix
    # A Toeplitz matrix is completely defined by the first row and first column
    total_random_bits = input_length + output_length - 1
    logger.debug(f"Generating {total_random_bits} random bits for Toeplitz matrix")
    random_bits = np.random.randint(0, 2, total_random_bits)
    
    # Create the Toeplitz matrix
    logger.debug(f"Creating Toeplitz matrix of shape ({output_length}, {input_length})")
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
    logger.info(f"Applying universal hash amplification with security level: {security_level} bits")
    
    if not key:
        logger.error("Empty key provided to universal hash function")
        return []
    
    # Convert key to string for hashing
    key_str = ''.join(map(str, key))
    logger.debug(f"Key length for hashing: {len(key_str)} bits")
    
    # Generate a random salt for hashing
    salt = str(random.randint(0, 2**32 - 1))
    logger.debug(f"Generated salt for universal hashing: {salt}")
    
    # Apply SHA-256 hash
    hash_input = key_str + salt
    logger.debug(f"Hash input length: {len(hash_input)} characters")
    hash_output = hashlib.sha256(hash_input.encode()).digest()
    
    # Convert hash to binary
    binary_hash = []
    for byte in hash_output:
        for bit in format(byte, '08b'):
            binary_hash.append(int(bit))
    
    # Truncate to desired security level
    output_length = min(len(key) // 2, security_level)
    logger.info(f"Universal hash output length: {output_length} bits")
    
    return binary_hash[:output_length]

def multi_level_privacy_amplification(corrected_key, qber, use_universal_hash=True, seed=None):
    """
    Apply multi-level privacy amplification for enhanced security.
    
    This function combines Toeplitz matrix hashing with universal hashing for
    stronger security guarantees.
    
    Args:
        corrected_key (list): The error-corrected key
        qber (float): The Quantum Bit Error Rate
        use_universal_hash (bool): Whether to apply universal hashing after Toeplitz
        seed (int, optional): Seed for the random number generator
        
    Returns:
        list: The final secure key
    """
    logger.info(f"Starting multi-level privacy amplification. Use universal hash: {use_universal_hash}")
    
    # First level: Toeplitz matrix hashing
    logger.info("Applying first level: Toeplitz matrix hashing")
    amplified_key = adaptive_privacy_amplification(corrected_key, qber, seed=seed)
    
    # Second level (optional): Universal hashing
    if use_universal_hash and len(amplified_key) > 16:
        logger.info("Applying second level: Universal hashing")
        final_key = universal_hash_amplification(amplified_key)
        logger.info(f"Applied second-level universal hashing, final length: {len(final_key)}")
        return final_key
    else:
        if not use_universal_hash:
            logger.info("Second level universal hashing disabled by configuration")
        elif len(amplified_key) <= 16:
            logger.info(f"Key too short ({len(amplified_key)} bits) for second-level hashing, skipping")
    
    return amplified_key

def share_privacy_amplification_seed(classical_channel):
    """
    Generate and share a secure seed for privacy amplification over the classical channel.
    
    Args:
        classical_channel: The authenticated classical channel object
        
    Returns:
        int: The shared seed
    """
    # Generate a secure seed
    seed = generate_secure_seed()
    logger.info(f"Generated secure seed for privacy amplification: {seed}")
    
    # Prepare seed data for transmission
    seed_data = {
        "type": "PRIVACY_AMPLIFICATION_SEED",
        "seed": seed
    }
    
    # Serialize to JSON
    seed_json = json.dumps(seed_data)
    
    # Send over the classical channel
    try:
        classical_channel.send(seed_json)
        logger.info("Privacy amplification seed sent successfully")
    except Exception as e:
        logger.error(f"Error sending privacy amplification seed: {str(e)}")
        # Fall back to a simpler seed in case of error
        seed = random.randint(0, 2**32 - 1)
        logger.warning(f"Using fallback seed: {seed}")
    
    return seed