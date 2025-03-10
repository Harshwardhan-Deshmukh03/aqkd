from dataclasses import dataclass
import time
from utils.falcon import SecretKey, PublicKey
from utils.logger import get_logger
from Crypto.PublicKey import RSA
from Crypto.Signature import pkcs1_15
from Crypto.Hash import SHA256
from typing import Optional  # Import Optional from typing


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
    rsa_key: RSA.RsaKey = None
    public_key: Optional[PublicKey] = None
    authenticated: bool = False
    clock_data: ClockData = ClockData(
        local_time=time.time(),
        offset=0.0,
        last_sync=time.time(),
        drift_rate=0.0
    )

    def __post_init__(self):
        self.generate_rsa_keys()

    def generate_rsa_keys(self):
        self.rsa_key = RSA.generate(2048)
        logger.info(f"{self.name} generated RSA keys")

    def sign_message(self, message: str) -> bytes:
        h = SHA256.new(message.encode('utf-8'))
        signature = pkcs1_15.new(self.rsa_key).sign(h)
        logger.info(f"{self.name} signed a message")
        return signature

    def verify_signature(self, message: str, signature: bytes, public_key: RSA.RsaKey) -> bool:
        h = SHA256.new(message.encode('utf-8'))
        try:
            pkcs1_15.new(public_key).verify(h, signature)
            logger.info(f"{self.name} verified a signature successfully")
            return True
        except (ValueError, TypeError):
            logger.error(f"{self.name} failed to verify a signature")
            return False

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
    def get_key(self) -> str:
        """Return the participant's name"""
        return self.rsa_key

def create_participants():
    alice = Participant(name="Alice")
    bob = Participant(name="Bob")
    return alice, bob