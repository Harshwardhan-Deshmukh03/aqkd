import hashlib
# import falcon
from utils.logger import get_logger

logger = get_logger(__name__)

def verify_key(classical_channel, final_key):
    """Verify the integrity of the final key"""
    logger.info("Starting key verification...")
    
    if not final_key:
        logger.error("No key to verify")
        return False
    
    # Calculate hash of the key
    key_hash = calculate_key_hash(final_key)
    
    # Sign the hash with Falcon
    signature = classical_channel.sign_message(key_hash)
    
    # Send hash and signature
    classical_channel.send((key_hash, signature))
    
    # Simulation: verify the signature
    # In a real system, Alice would verify Bob's signature
    verification_result = classical_channel.verify_signature(key_hash, signature)
    
    if verification_result:
        logger.info("Key verification successful")
    else:
        logger.error("Key verification failed")
    
    return verification_result

def calculate_key_hash(key):
    """Calculate a hash of the key for verification"""
    key_str = ''.join(map(str, key))
    hash_result = hashlib.sha256(key_str.encode()).hexdigest()
    return hash_result

def sign_message(message, secret_key, max_retries=100):
    """
    Signs a message using the Falcon signature algorithm.
    Retries the signing process if the signature norm is too large.
    """
    message_bytes = message.encode('utf-8')  # Convert the message to bytes
    retry_count = 0
    error_message = None
    
    while True:
        try:
            signature = secret_key.sign(message_bytes)
            logger.info(f"Signature generated successfully after {retry_count} retries")
            return signature, error_message
        except ValueError as e:
            if "Squared norm of signature is too large" in str(e):
                retry_count += 1
                logger.warning(f"Signature norm too large. Retrying ({retry_count}/{max_retries})...")
                error_message = str(e)
                if retry_count >= max_retries:
                    logger.error("Maximum retry count reached for signature generation")
                    return None, error_message
            else:
                raise e

def verify_message(message, signature, public_key):
    """
    Verifies the signature of a message using the Falcon signature algorithm.
    """
    message_bytes = message.encode('utf-8')  # Convert the message to bytes
    try:
        result = public_key.verify(message_bytes, signature)
        logger.info(f"Signature verification result: {result}")
        return result
    except Exception as e:
        logger.error(f"Signature verification failed: {str(e)}")
        return False