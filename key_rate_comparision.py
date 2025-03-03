import math
import numpy as np
from utils.logger import get_logger

logger = get_logger(__name__)

def calculate_theoretical_key_rate(key_length, qber, efficiency=0.5, dimension=4):
    """Calculate theoretical key rate based on security parameters"""
    logger.info(f"Calculating theoretical key rate with QBER: {qber:.4f}, dimension: {dimension}")
    
    # Higher dimensions can increase key rate in some scenarios
    dimension_factor = math.log2(dimension) / 2
    
    # Basic rate calculation (simplified from more complex formulas)
    if qber < 0.11:  # BB84 threshold
        rate = efficiency * (1 - 2 * h2(qber)) * dimension_factor
    else:
        rate = 0  # Above threshold, no secure key possible
    
    # Ensure rate is not negative
    rate = max(0, rate)
    
    expected_bits = int(key_length * rate)
    logger.info(f"Theoretical key rate: {rate:.4f}, expected bits: {expected_bits}")
    
    return rate, expected_bits

def calculate_actual_key_rate(initial_length, final_length, time_taken=1.0):
    """Calculate actual key rate from initial and final key lengths"""
    if initial_length <= 0:
        return 0
    
    rate = final_length / initial_length
    bits_per_second = final_length / time_taken
    
    logger.info(f"Actual key rate: {rate:.4f}, Bits/second: {bits_per_second:.2f}")
    return rate, bits_per_second

def compare_key_rates(theoretical_rate, actual_rate):
    """Compare theoretical and actual key rates"""
    if theoretical_rate <= 0:
        efficiency = 0
    else:
        efficiency = actual_rate / theoretical_rate
    
    logger.info(f"Key rate efficiency: {efficiency:.2f}")
    return efficiency

def h2(p):
    """Binary entropy function"""
    if p == 0 or p == 1:
        return 0
    return -p * math.log2(p) - (1-p) * math.log2(1-p)

def optimize_parameters(env_data, current_dimension):
    """Optimize parameters for better key rate"""
    qber = env_data["QBER"]
    loss = env_data["channel_loss"]
    
    # Simple optimization rules
    if qber > 0.08 and current_dimension > 4:
        new_dimension = current_dimension // 2
        logger.info(f"High QBER detected, reducing dimension from {current_dimension} to {new_dimension}")
        return new_dimension, "Computational"
    
    if qber < 0.02 and loss < 0.2 and current_dimension < 8:
        new_dimension = current_dimension * 2
        logger.info(f"Low QBER detected, increasing dimension from {current_dimension} to {new_dimension}")
        return new_dimension, "Fourier"
    
    # No change needed
    return current_dimension, "Computational" if current_dimension <= 4 else "Fourier"