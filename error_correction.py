import random
import math
from utils.logger import get_logger

logger = get_logger(__name__)

# def cascade_correction(classical_channel, sifted_key, qber):
#     """Implement the Cascade error correction protocol"""
#     logger.info(f"Starting Cascade error correction with QBER: {qber:.4f}")
    
#     if not sifted_key:
#         logger.error("No key to correct")
#         return []
    
#     # Determine optimal block size based on QBER
#     if qber < 0.01:
#         init_block_size = 32
#     elif qber < 0.05:
#         init_block_size = 16
#     elif qber < 0.1:
#         init_block_size = 8
#     else:
#         init_block_size = 4
    
#     # Make a copy of the key that we'll correct
#     corrected_key = sifted_key.copy()
    
#     # Perform multiple rounds of correction
#     for round_num in range(4):  # Typically 4 rounds of Cascade
#         # Calculate block size for this round
#         block_size = init_block_size * (2 ** round_num)
#         if block_size > len(corrected_key):
#             block_size = len(corrected_key)
        
#         logger.info(f"Cascade round {round_num+1}, block size: {block_size}")
        
#         # Divide the key into blocks
#         num_blocks = math.ceil(len(corrected_key) / block_size)
#         blocks = []
#         for i in range(num_blocks):
#             start = i * block_size
#             end = min(start + block_size, len(corrected_key))
#             blocks.append(corrected_key[start:end])
        
#         # Perform parity check for each block
#         for block_idx, block in enumerate(blocks):
#             # Calculate parity
#             parity = sum(block) % 2
            
#             # Send parity to check against Alice's parity
#             classical_channel.send(f"PARITY:{block_idx}:{parity}")
            
#             # Simulate Alice's response with a random error
#             # In a real system, Alice would calculate the parity of her corresponding block
#             alice_parity = (parity + (1 if random.random() < qber else 0)) % 2
            
#             # If parities don't match, perform binary search to find the error
#             if parity != alice_parity:
#                 # Binary search to find the error
#                 start_idx = block_idx * block_size
#                 end_idx = min(start_idx + block_size, len(corrected_key)) - 1
                
#                 while start_idx <= end_idx:
#                     mid_idx = (start_idx + end_idx) // 2
                    
#                     # Calculate parity of first half
#                     first_half_parity = sum(corrected_key[start_idx:mid_idx+1]) % 2
                    
#                     # Send parity to check against Alice's parity
#                     classical_channel.send(f"PARITY:{start_idx}-{mid_idx}:{first_half_parity}")
                    
#                     # Simulate Alice's response
#                     alice_first_half_parity = (first_half_parity + (1 if random.random() < qber and (end_idx - start_idx > 5) else 0)) % 2
                    
#                     if first_half_parity != alice_first_half_parity:
#                         # Error is in the first half
#                         end_idx = mid_idx
#                     else:
#                         # Error is in the second half
#                         start_idx = mid_idx + 1
                
#                 # After binary search, flip the bit at the identified position
#                 if start_idx < len(corrected_key):
#                     corrected_key[start_idx] = 1 - corrected_key[start_idx]
#                     logger.debug(f"Corrected bit at position {start_idx}")
    
#     # Verify correction success
#     error_rate = estimate_error_rate(sifted_key, corrected_key)
#     logger.info(f"Error correction complete. Estimated error rate: {error_rate:.4f}")
    
#     return corrected_key


def cascade_correction(classical_channel, bob_key, alice_key=None, qber=None, max_rounds=4):
    """
    Implement the Cascade error correction protocol
    
    Parameters:
    -----------
    classical_channel : object
        Channel for classical communication between Alice and Bob
    bob_key : list
        Bob's sifted key to be corrected
    alice_key : list, optional
        Alice's sifted key (for simulation purposes)
    qber : float, optional
        Estimated Quantum Bit Error Rate
    max_rounds : int, optional
        Maximum number of Cascade rounds to perform
        
    Returns:
    --------
    list
        Error-corrected key
    """
    import logging
    import math
    import random
    
    logger = logging.getLogger(__name__)
    logger.info("Starting Cascade error correction protocol")
    
    if not bob_key:
        logger.error("No key to correct")
        return []
    
    # Estimate QBER if not provided
    if qber is None:
        if alice_key is not None:
            # Calculate actual QBER from sample bits if Alice's key is available
            sample_size = min(100, len(bob_key))
            sample_indices = random.sample(range(len(bob_key)), sample_size)
            errors = sum(bob_key[i] != alice_key[i] for i in sample_indices)
            qber = errors / sample_size
        else:
            # Default QBER estimate if no information is available
            qber = 0.05
    
    logger.info(f"Using QBER estimate: {qber:.4f}")
    
    # Determine optimal initial block size based on QBER
    # BBBSS 1992 recommendation: block size ≈ 0.73/qber
    if qber < 0.01:
        init_block_size = 73  # ~0.73/0.01
    elif qber < 0.05:
        init_block_size = 16  # ~0.73/0.05
    elif qber < 0.1:
        init_block_size = 8   # ~0.73/0.1
    else:
        init_block_size = 4   # For high error rates
    
    # Make a copy of the key that we'll correct
    corrected_key = bob_key.copy()
    
    # Track corrected bit positions to optimize subsequent rounds
    corrected_positions = set()
    
    # Perform multiple rounds of correction
    for round_num in range(max_rounds):
        # Calculate block size for this round (different permutation each round)
        if round_num == 0:
            block_size = init_block_size
            # First round: sequential blocks
            permutation = list(range(len(corrected_key)))
        else:
            # Subsequent rounds: Block size doubles, random permutation
            block_size = init_block_size * (2 ** round_num)
            if block_size > len(corrected_key):
                block_size = len(corrected_key)
            
            # Create a random permutation for this round
            permutation = list(range(len(corrected_key)))
            random.shuffle(permutation)
        
        logger.info(f"Cascade round {round_num+1}, block size: {block_size}")
        
        # Divide the permuted key into blocks
        num_blocks = math.ceil(len(corrected_key) / block_size)
        errors_found = 0
        
        for block_idx in range(num_blocks):
            start = block_idx * block_size
            end = min(start + block_size, len(corrected_key))
            
            # Get block indices using the permutation
            block_indices = [permutation[i] for i in range(start, end)]
            
            # Calculate block parity
            block_parity = sum(corrected_key[i] for i in block_indices) % 2
            
            # Communicate with Alice about this block's parity
            if alice_key is not None:
                # Simulation mode: calculate Alice's parity directly
                alice_block_parity = sum(alice_key[i] for i in block_indices) % 2
            else:
                # Real mode: send parity to Alice and receive her parity
                classical_channel.send(f"PARITY_CHECK:ROUND{round_num}:BLOCK{block_idx}:{block_parity}")
                response = classical_channel.receive()
                alice_block_parity = int(response.split(":")[-1])
            
            # If parities don't match, perform binary search to find the error
            if block_parity != alice_block_parity:
                # Binary search to find the error
                error_index = binary_search_error(
                    corrected_key, 
                    block_indices, 
                    classical_channel, 
                    alice_key
                )
                
                if error_index is not None:
                    # Flip the erroneous bit
                    corrected_key[error_index] = 1 - corrected_key[error_index]
                    corrected_positions.add(error_index)
                    errors_found += 1
                    
                    # Cascade: Check other blocks that contain this bit from previous rounds
                    if round_num > 0:
                        cascade_error_correction(
                            corrected_key, 
                            error_index, 
                            round_num, 
                            corrected_positions,
                            classical_channel, 
                            alice_key
                        )
        
        logger.info(f"Round {round_num+1} complete. Errors found and corrected: {errors_found}")
        
        # Check if we should terminate early
        if errors_found == 0 and round_num >= 1:
            logger.info(f"No errors found in round {round_num+1}, terminating Cascade")
            break
    
    # Verify overall success rate if Alice's key is available
    if alice_key is not None:
        remaining_errors = sum(corrected_key[i] != alice_key[i] for i in range(len(corrected_key)))
        final_error_rate = remaining_errors / len(corrected_key)
        logger.info(f"Cascade complete. Remaining error rate: {final_error_rate:.6f}")
        logger.info(f"Total bits corrected: {len(corrected_positions)}")
    
    return corrected_key

def binary_search_error(key, block_indices, classical_channel, alice_key=None):
    """
    Perform binary search to find an error bit within a block
    
    Parameters:
    -----------
    key : list
        Bob's current key
    block_indices : list
        Indices of the bits in the block where an error exists
    classical_channel : object
        Channel for classical communication
    alice_key : list, optional
        Alice's key (for simulation)
        
    Returns:
    --------
    int or None
        Index of the error bit if found, None otherwise
    """
    import logging
    
    logger = logging.getLogger(__name__)
    
    if len(block_indices) == 1:
        return block_indices[0]
    
    # Binary search
    start = 0
    end = len(block_indices) - 1
    
    while start <= end:
        if end - start <= 1:
            # Base case: only two bits left
            first_bit_idx = block_indices[start]
            
            # Check first bit's parity (as a singleton block)
            if alice_key is not None:
                # Simulation mode
                bob_parity = key[first_bit_idx] % 2
                alice_parity = alice_key[first_bit_idx] % 2
            else:
                # Real communication mode
                bob_parity = key[first_bit_idx] % 2
                classical_channel.send(f"PARITY_CHECK:BIT:{first_bit_idx}:{bob_parity}")
                response = classical_channel.receive()
                alice_parity = int(response.split(":")[-1])
            
            if bob_parity != alice_parity:
                return first_bit_idx
            else:
                return block_indices[end]
        
        # Find middle index
        mid = (start + end) // 2
        first_half_indices = block_indices[start:mid+1]
        
        # Calculate parity of first half
        first_half_parity = sum(key[idx] for idx in first_half_indices) % 2
        
        if alice_key is not None:
            # Simulation mode
            alice_first_half_parity = sum(alice_key[idx] for idx in first_half_indices) % 2
        else:
            # Real communication mode
            indices_str = ",".join(str(idx) for idx in first_half_indices)
            classical_channel.send(f"PARITY_CHECK:SUBSET:{indices_str}:{first_half_parity}")
            response = classical_channel.receive()
            alice_first_half_parity = int(response.split(":")[-1])
        
        if first_half_parity != alice_first_half_parity:
            # Error is in the first half
            end = mid
        else:
            # Error is in the second half
            start = mid + 1
    
    logger.warning("Binary search did not find an error bit")
    return None

def cascade_error_correction(key, error_index, current_round, corrected_positions, 
                           classical_channel, alice_key=None):
    """
    Perform cascade error correction based on a newly discovered error
    
    Parameters:
    -----------
    key : list
        Bob's current key
    error_index : int
        Index of the just-corrected error bit
    current_round : int
        Current round number
    corrected_positions : set
        Set of positions that have been corrected
    classical_channel : object
        Channel for classical communication
    alice_key : list, optional
        Alice's key (for simulation)
    """
    import logging
    
    logger = logging.getLogger(__name__)
    logger.debug(f"Cascading error correction from bit {error_index}")
    
    # In a real implementation, we would need to keep track of all block assignments
    # from previous rounds. Here we implement a simplified approach.
    # In practice, we would store all permutations and block assignments.
    
    # Placeholder for cascade implementation
    # Key insight: If a bit flips in the current round, it means that all blocks
    # containing this bit in previous rounds must now have correct parity
    
    # This would be implemented by:
    # 1. Tracking which bits are in which blocks for all rounds
    # 2. When a bit is flipped, check all blocks containing it from earlier rounds
    # 3. If any of those blocks now have incorrect parity, there must be another error
    # 4. Recursively apply the process to find and fix all correlated errors
    
    # Simple demonstration for the idea (not complete implementation):
    errors_corrected = 0
    
    # In a real implementation, for each previous round:
    for round_idx in range(current_round):
        # Get the blocks from that round that contained this bit
        # This would use stored permutation and block information
        
        # For simulation, we'll just log what would happen
        logger.debug(f"Would check blocks from round {round_idx+1} containing bit {error_index}")
        
        # If the block parity no longer matches Alice's:
        # 1. Binary search for the new error
        # 2. Correct it
        # 3. Add to corrected_positions
        # 4. Recursively cascade again
    
    return errors_corrected


def estimate_error_rate(original_key, corrected_key):
    """Estimate the error rate between original and corrected keys"""
    if not original_key or len(original_key) != len(corrected_key):
        return 0.0
    
    errors = sum(1 for a, b in zip(original_key, corrected_key) if a != b)
    return errors / len(original_key)