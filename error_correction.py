import random
import math
from utils.logger import get_logger

logger = get_logger(__name__)

def cascade_correction(classical_channel, sifted_key, qber):
    """Implement the Cascade error correction protocol"""
    logger.info(f"Starting Cascade error correction with QBER: {qber:.4f}")
    
    if not sifted_key:
        logger.error("No key to correct")
        return []
    
    # Determine optimal block size based on QBER
    if qber < 0.01:
        init_block_size = 32
    elif qber < 0.05:
        init_block_size = 16
    elif qber < 0.1:
        init_block_size = 8
    else:
        init_block_size = 4
    
    # Make a copy of the key that we'll correct
    corrected_key = sifted_key.copy()
    
    # Perform multiple rounds of correction
    for round_num in range(4):  # Typically 4 rounds of Cascade
        # Calculate block size for this round
        block_size = init_block_size * (2 ** round_num)
        if block_size > len(corrected_key):
            block_size = len(corrected_key)
        
        logger.info(f"Cascade round {round_num+1}, block size: {block_size}")
        
        # Divide the key into blocks
        num_blocks = math.ceil(len(corrected_key) / block_size)
        blocks = []
        for i in range(num_blocks):
            start = i * block_size
            end = min(start + block_size, len(corrected_key))
            blocks.append(corrected_key[start:end])
        
        # Perform parity check for each block
        for block_idx, block in enumerate(blocks):
            # Calculate parity
            parity = sum(block) % 2
            
            # Send parity to check against Alice's parity
            classical_channel.send(f"PARITY:{block_idx}:{parity}")
            
            # Simulate Alice's response with a random error
            # In a real system, Alice would calculate the parity of her corresponding block
            alice_parity = (parity + (1 if random.random() < qber else 0)) % 2
            
            # If parities don't match, perform binary search to find the error
            if parity != alice_parity:
                # Binary search to find the error
                start_idx = block_idx * block_size
                end_idx = min(start_idx + block_size, len(corrected_key)) - 1
                
                while start_idx <= end_idx:
                    mid_idx = (start_idx + end_idx) // 2
                    
                    # Calculate parity of first half
                    first_half_parity = sum(corrected_key[start_idx:mid_idx+1]) % 2
                    
                    # Send parity to check against Alice's parity
                    classical_channel.send(f"PARITY:{start_idx}-{mid_idx}:{first_half_parity}")
                    
                    # Simulate Alice's response
                    alice_first_half_parity = (first_half_parity + (1 if random.random() < qber and (end_idx - start_idx > 5) else 0)) % 2
                    
                    if first_half_parity != alice_first_half_parity:
                        # Error is in the first half
                        end_idx = mid_idx
                    else:
                        # Error is in the second half
                        start_idx = mid_idx + 1
                
                # After binary search, flip the bit at the identified position
                if start_idx < len(corrected_key):
                    corrected_key[start_idx] = 1 - corrected_key[start_idx]
                    logger.debug(f"Corrected bit at position {start_idx}")
    
    # Verify correction success
    error_rate = estimate_error_rate(sifted_key, corrected_key)
    logger.info(f"Error correction complete. Estimated error rate: {error_rate:.4f}")
    
    return corrected_key

def estimate_error_rate(original_key, corrected_key):
    """Estimate the error rate between original and corrected keys"""
    if not original_key or len(original_key) != len(corrected_key):
        return 0.0
    
    errors = sum(1 for a, b in zip(original_key, corrected_key) if a != b)
    return errors / len(original_key)