import time
import random
import math
import hashlib
from utils.logger import get_logger
import os

logger = get_logger(__name__)

def time_execution(func):
    """Decorator to time the execution of a function"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        logger.info(f"Function {func.__name__} executed in {execution_time:.4f} seconds")
        return result, execution_time
    return wrapper

def estimate_entropy(data):
    """Estimate the entropy of a binary sequence"""
    if not data:
        return 0
    
    # Count occurrences of different bit patterns
    pattern_counts = {}
    for pattern_len in range(1, min(5, len(data)//10 + 1)):
        for i in range(len(data) - pattern_len + 1):
            pattern = tuple(data[i:i+pattern_len])
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
    
    # Calculate entropy based on pattern distribution
    total_patterns = len(data)
    entropy = 0
    for pattern_len in range(1, min(5, len(data)//10 + 1)):
        pattern_entropy = 0
        total_patterns_of_len = total_patterns - pattern_len + 1
        
        for pattern, count in pattern_counts.items():
            if len(pattern) == pattern_len:
                prob = count / total_patterns_of_len
                pattern_entropy -= prob * (prob and math.log2(prob))
        
        entropy += pattern_entropy / pattern_len
    
    # Normalize entropy (max is 1.0 for a perfect random sequence)
    normalized_entropy = min(1.0, entropy / 4.0)
    logger.info(f"Estimated entropy of data: {normalized_entropy:.4f}")
    return normalized_entropy

def generate_random_seed():
    """Generate a random seed for cryptographic operations"""
    # Combine system time, random bytes, and process info for better randomness
    seed_material = str(time.time()).encode()
    seed_material += os.urandom(32)
    seed_material += str(os.getpid()).encode()
    
    seed = hashlib.sha256(seed_material).digest()
    return seed

def check_min_entropy(data, threshold=0.8):
    """Check if the data has sufficient min-entropy"""
    # Simplified min-entropy check
    counts = {}
    for bit in data:
        counts[bit] = counts.get(bit, 0) + 1
    
    max_prob = max(count/len(data) for count in counts.values()) if counts else 0
    min_entropy = -math.log2(max_prob) if max_prob > 0 else 0
    
    # Normalize to [0,1] range
    normalized_min_entropy = min_entropy
    
    logger.info(f"Min-entropy check: {normalized_min_entropy:.4f} (threshold: {threshold})")
    return normalized_min_entropy >= threshold