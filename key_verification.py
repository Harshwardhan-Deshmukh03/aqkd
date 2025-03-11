import hashlib
# import falcon
import random
import struct
import json
from utils.logger import get_logger

logger = get_logger(__name__)

class UniversalHashFamily:
    """Implementation of a universal hash function family"""
    
    def __init__(self, prime=2**61 - 1):
        """Initialize with a large prime number"""
        self.prime = prime
        self.a = None
        self.b = None
    
    def select_function(self):
        """Select a random function from the family by choosing parameters a and b"""
        # Choose random coefficients for the hash function
        # a must be non-zero
        self.a = random.randint(1, self.prime - 1)
        self.b = random.randint(0, self.prime - 1)
        logger.info(f"Selected universal hash function with parameters a={self.a}, b={self.b}")
        return self.a, self.b
    
    def hash(self, data, a=None, b=None):
        """Compute the universal hash of the data using the selected function"""
        if a is None:
            a = self.a
        if b is None:
            b = self.b
        
        if a is None or b is None:
            raise ValueError("Hash function parameters not set. Call select_function() first.")
        
        # Convert bit array to chunks of integers
        chunks = []
        for i in range(0, len(data), 32):
            chunk = data[i:i+32]
            # Convert bit chunk to integer
            value = 0
            for bit in chunk:
                value = (value << 1) | bit
            chunks.append(value)
        
        # Apply universal hashing
        result = 0
        for x in chunks:
            result = (result + ((a * x + b) % self.prime)) % self.prime
            
        return result

def calculate_parity(bit_array, segment_size=8):
    """Calculate parity bits for segments of the key"""
    parity_bits = []
    
    for i in range(0, len(bit_array), segment_size):
        segment = bit_array[i:i+segment_size]
        # Even parity: count of 1s should be even
        # If odd number of 1s, parity bit is 1; if even, parity bit is 0
        parity_bit = sum(segment) % 2
        parity_bits.append(parity_bit)
    
    logger.info(f"Generated {len(parity_bits)} parity bits for {len(bit_array)} key bits")
    return parity_bits

def verify_parity(bit_array, parity_bits, segment_size=8):
    """Verify key integrity using parity bits"""
    errors = []
    
    for i, parity_bit in enumerate(parity_bits):
        start_idx = i * segment_size
        end_idx = min(start_idx + segment_size, len(bit_array))
        
        if start_idx >= len(bit_array):
            break
            
        segment = bit_array[start_idx:end_idx]
        calculated_parity = sum(segment) % 2
        
        if calculated_parity != parity_bit:
            errors.append(i)
            logger.warning(f"Parity error detected in segment {i} (bits {start_idx}-{end_idx-1})")
    
    return errors

def verify_key(classical_channel, final_key):
    """Verify the integrity of the final key using universal hash and parity checks"""
    logger.info("Starting key verification...")
    
    if not final_key:
        logger.error("No key to verify")
        return False
    
    # 1. Universal Hash Verification
    uhash = UniversalHashFamily()
    a, b = uhash.select_function()
    
    # Calculate hash of the key
    key_hash = uhash.hash(final_key)
    
    # Convert to strings for safe transmission
    hash_data = {
        "type": "HASH_DATA",
        "a": str(a),
        "b": str(b),
        "hash": str(key_hash)
    }
    
    # Serialize to JSON string for transmission
    hash_json = json.dumps(hash_data)
    
    # Send hash parameters and hash value
    classical_channel.send(hash_json)
    
    # Simulate receiving other party's hash parameters and value
    # In a real system, we would receive this from the other party
    # For simulation, we'll use our own data
    received_json = classical_channel.receive(hash_json)
    
    try:
        received_data = json.loads(received_json)
        other_a = int(received_data["a"])
        other_b = int(received_data["b"])
        other_hash = int(received_data["hash"])
        
        # Verify the hashes match
        hash_verified = (key_hash == other_hash)
        logger.info(f"Universal hash verification: {'Successful' if hash_verified else 'Failed'}")
        
        if not hash_verified:
            # 2. Parity Check for detailed error detection
            parity_bits = calculate_parity(final_key)
            
            # Convert parity bits to JSON
            parity_data = {
                "type": "PARITY_DATA",
                "parity": parity_bits
            }
            parity_json = json.dumps(parity_data)
            
            # Exchange parity bits
            classical_channel.send(parity_json)
            
            # Simulate receiving other party's parity bits
            received_parity_json = classical_channel.receive(parity_json)
            received_parity_data = json.loads(received_parity_json)
            other_parity = received_parity_data["parity"]
            
            # Identify errors using parity
            errors = verify_parity(final_key, other_parity)
            
            if errors:
                logger.error(f"Parity errors detected in segments: {errors}")
                
                # Request retransmission of specific segments
                retransmission_result = request_segment_retransmission(classical_channel, errors, final_key)
                return retransmission_result
            else:
                logger.error("Hash verification failed but no parity errors detected. Discarding key.")
                return False
        
        # Sign the verification result
        verify_msg = f"VERIFIED:{hash_verified}"
        signature = classical_channel.sign_message(verify_msg)
        
        # Send verification with signature
        classical_channel.send(verify_msg, signature)
        
        # Verification successful
        logger.info("Key verification successful")
        return True
        
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        logger.error(f"Error processing hash data: {str(e)}")
        return False

def request_segment_retransmission(classical_channel, error_segments, current_key, segment_size=8):
    """Request retransmission of specific key segments"""
    logger.info(f"Requesting retransmission of {len(error_segments)} segments")
    
    # Create retransmission request
    request_data = {
        "type": "RETRANSMIT_REQUEST",
        "segments": error_segments
    }
    request_json = json.dumps(request_data)
    
    # Send request
    classical_channel.send(request_json)
    
    # Simulate receiving retransmitted segments
    # For simulation, we'll create mock retransmitted segments
    retransmitted_segments = {}
    for segment_idx in error_segments:
        start_idx = segment_idx * segment_size
        end_idx = min(start_idx + segment_size, len(current_key))
        # In a real system, this data would come from the other party
        # We'll use the current key data as mock "corrected" data
        segment_bits = current_key[start_idx:end_idx]
        retransmitted_segments[str(segment_idx)] = segment_bits
    
    # Create response data
    response_data = {
        "type": "RETRANSMIT_RESPONSE",
        "segments": retransmitted_segments
    }
    response_json = json.dumps(response_data)
    
    # Simulate receiving the response
    received_response_json = classical_channel.receive(response_json)
    
    try:
        received_response = json.loads(received_response_json)
        
        if received_response["type"] != "RETRANSMIT_RESPONSE":
            logger.error("Invalid response type to retransmission request")
            return False
        
        # Process retransmitted segments
        segments_data = received_response["segments"]
        for segment_idx_str, segment_bits in segments_data.items():
            segment_idx = int(segment_idx_str)
            start_idx = segment_idx * segment_size
            end_idx = min(start_idx + segment_size, len(current_key))
            
            # Replace the segment in the key
            # Note: In a real implementation, you'd need to handle
            # the conversion from JSON back to your bit format
            current_key[start_idx:end_idx] = segment_bits
            
            logger.info(f"Updated segment {segment_idx} (bits {start_idx}-{end_idx-1})")
        
        # For simulation purposes, we'll just return success
        logger.info("Retransmission and update successful")
        return True
        
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        logger.error(f"Error processing retransmission data: {str(e)}")
        return False

def calculate_key_hash(key):
    """Calculate a cryptographic hash of the key (backup method)"""
    key_str = ''.join(map(str, key))
    hash_result = hashlib.sha256(key_str.encode()).hexdigest()
    return hash_result