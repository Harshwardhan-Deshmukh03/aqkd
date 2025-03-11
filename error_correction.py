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


def cascade_correction(classical_channel, bob_key, alice_key=None, qber=None, max_rounds=4, error_threshold=0.001):
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
    error_threshold : float, optional
        Terminate when estimated error rate falls below this threshold
        
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
    
    # Current estimated error rate
    current_error_rate = qber
    
    # Store permutations and block assignments for each round
    # FIX 3: Store permutations for cascade effect
    round_permutations = []
    round_block_assignments = []
    
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
        
        # FIX 3: Store permutation for this round
        round_permutations.append(permutation)
        
        logger.info(f"Cascade round {round_num+1}, block size: {block_size}")
        
        # Divide the permuted key into blocks
        num_blocks = math.ceil(len(corrected_key) / block_size)
        errors_found = 0
        
        # FIX 3: Store block assignments for this round
        block_assignments = [[] for _ in range(num_blocks)]
        
        for block_idx in range(num_blocks):
            start = block_idx * block_size
            end = min(start + block_size, len(corrected_key))
            
            # Get block indices using the permutation
            block_indices = [permutation[i] for i in range(start, end)]
            
            # FIX 3: Store which indices belong to this block
            block_assignments[block_idx] = block_indices
            
            # FIX 4: Use XOR for parity calculation instead of sum
            block_parity = 0
            for i in block_indices:
                block_parity ^= corrected_key[i]
            
            # Communicate with Alice about this block's parity
            if alice_key is not None:
                # Simulation mode: calculate Alice's parity directly
                # FIX 4: Use XOR for Alice's parity calculation
                alice_block_parity = 0
                for i in block_indices:
                    alice_block_parity ^= alice_key[i]
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
                    
                    # FIX 2: Improved cascade with proper block tracking
                    if round_num > 0:
                        additional_errors = cascade_error_correction(
                            corrected_key, 
                            error_index, 
                            round_num, 
                            corrected_positions,
                            round_permutations,
                            round_block_assignments,
                            classical_channel, 
                            alice_key
                        )
                        errors_found += additional_errors
        
        # FIX 3: Store block assignments for this round
        round_block_assignments.append(block_assignments)
        
        logger.info(f"Round {round_num+1} complete. Errors found and corrected: {errors_found}")
        
        # Estimate the current error rate
        if alice_key is not None:
            # In simulation mode, calculate the actual error rate
            remaining_errors = sum(corrected_key[i] != alice_key[i] for i in range(len(corrected_key)))
            current_error_rate = remaining_errors / len(corrected_key)
            logger.info(f"Current estimated error rate: {current_error_rate:.6f}")
        else:
            # In real mode, estimate the error rate based on errors found and QBER
            # This is a simple heuristic - in practice, you would use more sophisticated methods
            if round_num == 0:
                # After first round, adjust based on errors found
                expected_errors = int(qber * len(corrected_key))
                if errors_found > 0:
                    # Update error estimate based on first round findings
                    correction_factor = min(1.0, errors_found / expected_errors)
                    current_error_rate = qber * correction_factor
            else:
                # For later rounds, estimate based on diminishing returns
                if errors_found > 0:
                    # If we're still finding errors, estimate remains higher
                    current_error_rate = max(current_error_rate * 0.5, 0.002)  # Assume at least halving the errors
                else:
                    # If no errors found, estimate a lower bound
                    current_error_rate = max(current_error_rate * 0.25, 0.0005)  # Assume significant reduction
            
            logger.info(f"Current estimated error rate: {current_error_rate:.6f}")
        
        # Check termination conditions
        if current_error_rate <= error_threshold:
            logger.info(f"Error rate below threshold ({error_threshold}), terminating Cascade")
            break
            
        # Check if we should terminate early due to no more errors
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
            second_bit_idx = block_indices[end]
            
            # Check first bit's parity (as a singleton block)
            if alice_key is not None:
                # Simulation mode
                bob_parity = key[first_bit_idx]
                alice_parity = alice_key[first_bit_idx]
            else:
                # Real communication mode
                bob_parity = key[first_bit_idx]
                classical_channel.send(f"PARITY_CHECK:BIT:{first_bit_idx}:{bob_parity}")
                response = classical_channel.receive()
                alice_parity = int(response.split(":")[-1])
            
            if bob_parity != alice_parity:
                return first_bit_idx
            else:
                # FIX 1: Verify the second bit actually has an error
                if alice_key is not None:
                    # Simulation mode
                    bob_parity = key[second_bit_idx]
                    alice_parity = alice_key[second_bit_idx]
                else:
                    # Real communication mode
                    bob_parity = key[second_bit_idx]
                    classical_channel.send(f"PARITY_CHECK:BIT:{second_bit_idx}:{bob_parity}")
                    response = classical_channel.receive()
                    alice_parity = int(response.split(":")[-1])
                
                if bob_parity != alice_parity:
                    return second_bit_idx
                else:
                    # If we get here, no error was found in either bit
                    # This shouldn't happen if the block has an odd number of errors
                    logger.warning("No error found in final two bits, despite parity mismatch")
                    return None
        
        # Find middle index
        mid = (start + end) // 2
        first_half_indices = block_indices[start:mid+1]
        
        # FIX 4: Use XOR for parity calculation
        first_half_parity = 0
        for idx in first_half_indices:
            first_half_parity ^= key[idx]
        
        if alice_key is not None:
            # Simulation mode - FIX 4: Use XOR for Alice's parity
            alice_first_half_parity = 0
            for idx in first_half_indices:
                alice_first_half_parity ^= alice_key[idx]
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
                           round_permutations, round_block_assignments,
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
    round_permutations : list
        List of permutations used in each round
    round_block_assignments : list
        List of block assignments for each round
    classical_channel : object
        Channel for classical communication
    alice_key : list, optional
        Alice's key (for simulation)
        
    Returns:
    --------
    int
        Number of additional errors corrected
    """
    import logging
    
    logger = logging.getLogger(__name__)
    logger.debug(f"Cascading error correction from bit {error_index}")
    
    # FIX 2: Implement proper cascade error correction
    errors_corrected = 0
    
    # Check all previous rounds
    for round_idx in range(current_round):
        # Find which block in this round contained the error bit
        permutation = round_permutations[round_idx]
        block_assignments = round_block_assignments[round_idx]
        
        # Find the reverse mapping: in which position of the permutation was this bit?
        # We need to find i such that permutation[i] == error_index
        for i, perm_idx in enumerate(permutation):
            if perm_idx == error_index:
                permuted_position = i
                break
        else:
            logger.error(f"Could not find error_index {error_index} in permutation for round {round_idx}")
            continue
        
        # Find which block this permuted position belonged to
        containing_block = None
        for block_idx, block_indices in enumerate(block_assignments):
            if permuted_position in block_indices:
                containing_block = block_idx
                break
        
        if containing_block is None:
            logger.error(f"Could not find block containing bit {permuted_position} in round {round_idx}")
            continue
            
        # Get all indices in this block
        block_indices = block_assignments[containing_block]
        
        # Calculate block parity after correction
        # FIX 4: Use XOR for parity calculation
        block_parity = 0
        for idx in block_indices:
            orig_bit_idx = permutation[idx]  # Get the original bit index
            block_parity ^= key[orig_bit_idx]
        
        # Get Alice's parity for this block
        if alice_key is not None:
            # Simulation mode
            alice_block_parity = 0
            for idx in block_indices:
                orig_bit_idx = permutation[idx]
                alice_block_parity ^= alice_key[orig_bit_idx]
        else:
            # Real communication mode
            # We need to tell Alice which block we're checking from which round
            block_id = f"ROUND{round_idx}:BLOCK{containing_block}"
            classical_channel.send(f"PARITY_CHECK:{block_id}:{block_parity}")
            response = classical_channel.receive()
            alice_block_parity = int(response.split(":")[-1])
        
        # If parities still don't match, there's another error in this block
        if block_parity != alice_block_parity:
            logger.debug(f"Found parity mismatch in round {round_idx} block {containing_block} after correction")
            
            # Map block indices back to original bit indices
            original_indices = [permutation[idx] for idx in block_indices]
            
            # We need to exclude the bit we just corrected
            original_indices = [idx for idx in original_indices if idx != error_index]
            
            if original_indices:
                # Use binary search to find the new error
                new_error_index = binary_search_error(
                    key,
                    original_indices,
                    classical_channel,
                    alice_key
                )
                
                if new_error_index is not None:
                    # Found another error, correct it
                    key[new_error_index] = 1 - key[new_error_index]
                    corrected_positions.add(new_error_index)
                    errors_corrected += 1
                    
                    # Recursively cascade from this new error
                    additional_errors = cascade_error_correction(
                        key,
                        new_error_index,
                        round_idx + 1,  # Only check rounds before this new error was found
                        corrected_positions,
                        round_permutations,
                        round_block_assignments,
                        classical_channel,
                        alice_key
                    )
                    
                    errors_corrected += additional_errors
    
    return errors_corrected


def estimate_error_rate(original_key, corrected_key):
    """Estimate the error rate between original and corrected keys"""
    if not original_key or len(original_key) != len(corrected_key):
        return 0.0
    
    errors = sum(1 for a, b in zip(original_key, corrected_key) if a != b)
    return errors / len(original_key)