import sys
sys.path.append('./falcon.py/')
import falcon

from dataclasses import dataclass
import time
from utils.logger import get_logger
from typing import Optional, List 

logger = get_logger(__name__)

@dataclass
class ClockData:
    local_time: float
    offset: float
    last_sync: float
    drift_rate: float

@dataclass
class Participant:
    name: str
    encoding_supported: List[str]
    bases: List[str] = None
    bits: List[str] = None
    corrected_key: List[str] = None

    sent_qubits: List[str] = None
    received_qubits: List[str] = None
    secret_key: Optional[falcon.SecretKey] = None
    public_key: Optional[falcon.PublicKey] = None
    authenticated: bool = False
    clock_data: ClockData = ClockData(
        local_time=time.time(),
        offset=0.0,
        last_sync=time.time(),
        drift_rate=0.0
    )
    

    def __post_init__(self):
        self.generate_falcon_keys()

    def generate_falcon_keys(self):
        n = 128  # Security parameter for Falcon
        self.secret_key = falcon.SecretKey(n)
        self.public_key = falcon.PublicKey(self.secret_key)
        logger.info(f"{self.name} generated Falcon keys")

    def sign_message(self, message, secret_key, max_retries=100):
        message_bytes = message.encode('utf-8')  # Convert the message to bytes
        retry_count = 0
        error_message = None
        while True:
            try:
                signature = secret_key.sign(message_bytes)
                print(f"Signature generated successfully after {retry_count} retries.")
                return signature
            except ValueError as e:
                if "Squared norm of signature is too large" in str(e):
                    retry_count += 1
                    print(f"Signature norm too large. Retrying ({retry_count}/{max_retries})...")
                    error_message = str(e)
                    if retry_count >= max_retries:
                        return None, error_message
                else:
                    raise e

    def verify_signature(self, message, signature, public_key):
        message_bytes = message.encode('utf-8')  # Convert the message to bytes
        return public_key.verify(message_bytes, signature)

    def authenticate(self):
        self.authenticated = True
        logger.info(f"{self.name} authenticated successfully")
        return self.authenticated

    def synchronize_clock(self, remote_timestamp: float):
        offset = self.measure_time_offset(remote_timestamp)
        success = self.adjust_clock(offset)
        if success:
            logger.info(f"{self.name}'s clock synchronized with offset: {offset}")
        else:
            logger.error(f"{self.name}'s clock synchronization failed")
        return success

    def measure_time_offset(self, remote_timestamp: float) -> float:
        local_time = time.time()
        round_trip_time = local_time - remote_timestamp
        offset = round_trip_time / 2
        logger.debug(f"{self.name} measured time offset: {offset}")
        return offset

    def adjust_clock(self, offset: float) -> bool:
        try:
            self.clock_data.offset = offset
            self.clock_data.last_sync = time.time()
            self.clock_data.local_time = time.time() + offset
            logger.debug(f"{self.name}'s clock adjusted by offset: {offset}")
            return True
        except Exception as e:
            logger.error(f"{self.name}'s clock adjustment failed: {str(e)}")
            return False

    def get_name(self) -> str:
        """Return the participant's name"""
        return self.name
    def get_list(self) -> str:
        """Return the participant's name"""
        return self.encoding_supported

    def get_key(self) -> str:
        """Return the participant's public key"""
        return str(self.public_key)

def create_participants():
    alice = Participant(name="Alice",encoding_supported=["BB84","SIX_STATE","EIGHT_STATE","DECOY_BB84","E91","THREE_PLUS_ONE"])
    bob = Participant(name="Bob",encoding_supported=["BB84","SIX_STATE","EIGHT_STATE","DECOY_BB84","E91"])
    return alice, bob