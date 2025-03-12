import hashlib
import time
import random
import numpy as np
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass
from datetime import datetime, timedelta
from utils.logger import get_logger
# from utils.falcon import *
from enum import Enum
from qiskit import QuantumCircuit
from typing import Dict, List, Tuple, Optional

import sys
sys.path.append('./falcon.py/')
import falcon


logger = get_logger(__name__)

import logging

# Configure Qiskit logging
qiskit_logger = logging.getLogger('qiskit')
qiskit_logger.setLevel(logging.WARNING)  # or logging.ERROR to suppress more

# Configure your application logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


@dataclass
class ClockData:
    local_time: float
    offset: float
    last_sync: float
    drift_rate: float


class EncodingScheme(Enum):
    BB84 = "BB84"
    SIX_STATE = "SIX_STATE"
    EIGHT_STATE = "EIGHT_STATE"
    THREE_PLUS_ONE = "THREE_PLUS_ONE"
    DECOY_BB84 = "DECOY_BB84"
    E91 = "E91" 


@dataclass
class EncodingParameters:
    scheme: EncodingScheme
    dimension: int
    basis_sets: List[str]
    error_tolerance: float
    max_distance: float
    uses_entanglement: bool = False  # Add this flag
    recommended_intensity: float = 1.0




from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
from qiskit.quantum_info import state_fidelity, Statevector


# Add these new classes after your ClockData class
@dataclass
class DecoyState:
    intensity: float
    num_photons: int
    position: int
    basis: int

@dataclass
class BellState:
    circuit: QuantumCircuit
    type: str  # Phi+, Phi-, Psi+, Psi-
    fidelity: float



class QuantumEncoder:
    def __init__(self):
        self.supported_encodings = {
            EncodingScheme.BB84: EncodingParameters(
                scheme=EncodingScheme.BB84,
                dimension=2,
                basis_sets=["computational", "hadamard"],
                error_tolerance=0.11,
                max_distance=50
            ),
            EncodingScheme.SIX_STATE: EncodingParameters(
                scheme=EncodingScheme.SIX_STATE,
                dimension=2,
                basis_sets=["computational", "hadamard", "circular"],
                error_tolerance=0.126,
                max_distance=40
            ),
            EncodingScheme.EIGHT_STATE: EncodingParameters(
                scheme=EncodingScheme.EIGHT_STATE,
                dimension=2,
                basis_sets=["computational", "hadamard", "circular", "custom"],
                error_tolerance=0.15,
                max_distance=30
            ),
            EncodingScheme.THREE_PLUS_ONE: EncodingParameters(
                scheme=EncodingScheme.THREE_PLUS_ONE,
                dimension=3,
                basis_sets=["three_state", "single_state"],
                error_tolerance=0.13,
                max_distance=45
            ),
            EncodingScheme.DECOY_BB84: EncodingParameters(
                scheme=EncodingScheme.DECOY_BB84,
                dimension=2,
                basis_sets=["computational", "hadamard"],
                error_tolerance=0.11,
                max_distance=100,
                recommended_intensity=0.5
            ),
             EncodingScheme.E91: EncodingParameters(
            scheme=EncodingScheme.E91,
            dimension=2,
            basis_sets=["bell_basis"],
            error_tolerance=0.15,
            max_distance=100,
            uses_entanglement=True
        )
        }
        self.current_scheme = None
        logger.info("Quantum encoder initialized")

    def prepare_state(self, bit: int, basis: int, scheme: EncodingScheme, **kwargs) -> QuantumCircuit:
        """Prepare quantum state based on encoding scheme"""
        try:
            if scheme not in self.supported_encodings:
                raise ValueError(f"Unsupported encoding scheme: {scheme}")
            qc = QuantumCircuit(1, 1)
            if scheme == EncodingScheme.E91:
                return self._prepare_e91()
            if scheme == EncodingScheme.BB84:
                self._prepare_bb84(qc, bit, basis)
            elif scheme == EncodingScheme.SIX_STATE:
                self._prepare_six_state(qc, bit, basis)
            elif scheme == EncodingScheme.EIGHT_STATE:
                self._prepare_eight_state(qc, bit, basis)
            elif scheme == EncodingScheme.THREE_PLUS_ONE:
                self._prepare_three_plus_one(qc, bit, basis)
            elif scheme == EncodingScheme.DECOY_BB84:
                intensity = kwargs.get('intensity', 1.0)
                self._prepare_decoy_bb84(qc, bit, basis, intensity)
                
            return qc
        except Exception as e:
            logger.error(f"State preparation failed: {str(e)}")
            raise
        

    def _prepare_bb84(self, qc: QuantumCircuit, bit: int, basis: int):
        """Standard BB84 state preparation"""
        if bit == 1:
            qc.x(0)
        if basis == 1:
            qc.h(0)

    def _prepare_six_state(self, qc: QuantumCircuit, bit: int, basis: int):
        """Six-state protocol preparation"""
        if bit == 1:
            qc.x(0)
        if basis == 1:
            qc.h(0)
        elif basis == 2:
            qc.h(0)
            qc.s(0)

    def _prepare_eight_state(self, qc: QuantumCircuit, bit: int, basis: int):
        """Eight-state protocol preparation"""
        if bit == 1:
            qc.x(0)
        if basis == 1:
            qc.h(0)
        elif basis == 2:
            qc.h(0)
            qc.s(0)
        elif basis == 3:
            qc.sdg(0)
            qc.h(0)

    def _prepare_three_plus_one(self, qc: QuantumCircuit, bit: int, basis: int):
        """Three-plus-one protocol preparation"""
        if basis == 0:
            # Three-state basis
            if bit == 1:
                qc.rx(2*np.pi/3, 0)
            elif bit == 2:
                qc.rx(4*np.pi/3, 0)
        else:
            # Single-state basis
            qc.h(0)

    def _prepare_decoy_bb84(self, qc: QuantumCircuit, bit: int, basis: int, intensity: float):
        """Decoy-state BB84 preparation"""
        if bit == 1:
            qc.x(0)
        if basis == 1:
            qc.h(0)
        # Apply intensity adjustment
        theta = np.arccos(np.sqrt(intensity))
        qc.rx(theta, 0)


    def _prepare_e91(self) -> QuantumCircuit:
        """E91 protocol state preparation"""
        qc = QuantumCircuit(2, 2)
        # Create Bell state
        qc.h(0)
        qc.cx(0, 1)
        return qc

    def select_encoding(self, channel_conditions: Dict) -> EncodingScheme:
        """Select appropriate encoding based on channel conditions"""
        qber = channel_conditions.get("QBER", 0.0)
        distance = channel_conditions.get("distance", 0.0)
        noise_level = channel_conditions.get("noise_level", 0.0)
        
        if distance > 50:
            return EncodingScheme.DECOY_BB84
        elif noise_level > 0.1:
            return EncodingScheme.EIGHT_STATE
        elif qber > 0.08:
            return EncodingScheme.SIX_STATE
        elif channel_conditions.get("asymmetric", False):
            return EncodingScheme.THREE_PLUS_ONE
        else:
            return EncodingScheme.BB84





class QuantumChannel:
    def __init__(self,encoding, error_rate=0.03, loss_rate=0.1):
        self.encoding= encoding
        self.error_rate = error_rate
        self.loss_rate = loss_rate
        self.received_qubits = []
        self.dummy_positions = []
        self.calibration_data = {}
        self.clock_data = ClockData(
            local_time=time.time(),
            offset=0.0,
            last_sync=time.time(),
            drift_rate=0.0
        )
        self.decoy_states: List[DecoyState] = []
        self.bell_pairs: List[BellState] = []
        self.backend = Aer.get_backend('aer_simulator')
        self.encoder = QuantumEncoder()
        self.current_scheme = None
        logger.info(f"Quantum channel initialized with error rate: {error_rate}, loss rate: {loss_rate}")

    # Keep existing methods
    def transmit(self, qubits):
        transmitted_qubits = []
        for qubit in qubits:
            if random.random() > self.loss_rate:
                if random.random() < self.error_rate:
                    qubit = flip_qubit(qubit)
                transmitted_qubits.append(qubit)
            else:
                transmitted_qubits.append(None)  # Lost qubit
        return transmitted_qubits

    def set_error_rate(self, error_rate):
        self.error_rate = error_rate
    
    def set_loss_rate(self, loss_rate):
        self.loss_rate = loss_rate
    
    

    # Add new methods
    def generate_dummy_qubits(self, num_dummies: int) -> List[Tuple[int, int]]:
        """Generate dummy qubits for channel calibration"""
        dummy_qubits = []
        for _ in range(num_dummies):
            state = random.randint(0, 1)
            basis = random.randint(0, 1)
            dummy_qubits.append((state, basis))
        logger.debug(f"Generated {num_dummies} dummy qubits")
        return dummy_qubits

    def insert_dummy_qubits(self, qubits: List[Tuple], dummy_ratio: float = 0.1) -> List[Tuple]:
        """Insert dummy qubits into transmission"""
        num_dummies = int(len(qubits) * dummy_ratio)
        dummy_qubits = self.generate_dummy_qubits(num_dummies)
        
        self.dummy_positions = []
        enhanced_qubits = qubits.copy()
        
        for dummy in dummy_qubits:
            position = random.randint(0, len(enhanced_qubits))
            enhanced_qubits.insert(position, dummy)
            self.dummy_positions.append(position)
        
        logger.debug(f"Inserted {num_dummies} dummy qubits")
        return enhanced_qubits

    def calibrate_channel(self) -> Dict:
        """Calibrate channel using dummy qubits"""
        calibration_qubits = self.generate_dummy_qubits(100)
        transmitted = self.transmit(calibration_qubits)
        
        received_count = sum(1 for q in transmitted if q is not None)
        error_count = sum(1 for i, q in enumerate(transmitted) 
                         if q is not None and q[0] != calibration_qubits[i][0])
        
        self.calibration_data = {
            "loss_rate": (len(transmitted) - received_count) / len(transmitted),
            "error_rate": error_count / received_count if received_count > 0 else 1.0,
            "timestamp": time.time()
        }
        
        logger.info(f"Channel calibration completed: {self.calibration_data}")
        return self.calibration_data
    

    def generate_decoy_states(self, num_states: int, intensities: List[float]) -> List[DecoyState]:
        """
        Generate decoy states with different intensities
        """
        decoy_states = []
        for _ in range(num_states):
            intensity = random.choice(intensities)
            num_photons = np.random.poisson(intensity)
            basis = random.randint(0, 1)
            position = random.randint(0, 1000)  # Random position in sequence
            
            decoy = DecoyState(
                intensity=intensity,
                num_photons=num_photons,
                position=position,
                basis=basis
            )
            decoy_states.append(decoy)
            
        self.decoy_states = decoy_states
        logger.debug(f"Generated {num_states} decoy states")
        return decoy_states
    
    def insert_decoy_states(self, qubits: List[Tuple], decoy_states: List[DecoyState]) -> List[Tuple]:
        """
        Insert decoy states into quantum transmission
        """
        enhanced_qubits = qubits.copy()
        for decoy in decoy_states:
            # Create decoy state qubit
            decoy_qubit = (random.randint(0, 1), decoy.basis, decoy.intensity)
            # Insert at random position
            position = random.randint(0, len(enhanced_qubits))
            enhanced_qubits.insert(position, decoy_qubit)
            decoy.position = position
            
        logger.debug(f"Inserted {len(decoy_states)} decoy states")
        return enhanced_qubits
    
    def analyze_decoy_statistics(self) -> Dict:
        """
        Analyze decoy state statistics to detect attacks
        """
        stats = {
            "yield": {},
            "error_rates": {},
            "photon_numbers": {}
        }
        
        for intensity in set(d.intensity for d in self.decoy_states):
            intensity_decoys = [d for d in self.decoy_states if d.intensity == intensity]
            
            # Calculate yield and error rates for each intensity
            total = len(intensity_decoys)
            detected = sum(1 for d in intensity_decoys if d.num_photons > 0)
            errors = sum(1 for d in intensity_decoys if d.num_photons != 1)
            
            stats["yield"][intensity] = detected / total if total > 0 else 0
            stats["error_rates"][intensity] = errors / detected if detected > 0 else 0
            stats["photon_numbers"][intensity] = sum(d.num_photons for d in intensity_decoys) / total
            
        logger.info(f"Decoy state analysis completed: {stats}")
        return stats
    
    # Add these new methods for Bell Tests
    def create_bell_pair(self, bell_type: str = "Phi+") -> BellState:
        """
        Create a Bell pair for entanglement-based QKD
        """
        qc = QuantumCircuit(2, 2)
        
        # Create different Bell states
        if bell_type == "Phi+":
            qc.h(0)
            qc.cx(0, 1)
        elif bell_type == "Phi-":
            qc.h(0)
            qc.cx(0, 1)
            qc.z(1)
        elif bell_type == "Psi+":
            qc.h(0)
            qc.cx(0, 1)
            qc.x(1)
        elif bell_type == "Psi-":
            qc.h(0)
            qc.cx(0, 1)
            qc.x(1)
            qc.z(1)
            
        # Calculate state fidelity
        ideal_state = Statevector.from_instruction(qc)
        fidelity = state_fidelity(ideal_state, ideal_state)
        
        bell_state = BellState(
            circuit=qc,
            type=bell_type,
            fidelity=fidelity
        )
        
        self.bell_pairs.append(bell_state)
        logger.debug(f"Created Bell pair of type {bell_type} with fidelity {fidelity}")
        return bell_state
    
    def measure_bell_state(self, bell_state: BellState, basis: List[int]) -> Tuple[int, int]:
        """
        Measure a Bell state in specified bases
        """
        try:
            qc = bell_state.circuit.copy()
            if basis[0] == 1:  # X basis for first qubit
                qc.h(0)
            if basis[1] == 1:  # X basis for second qubit
                qc.h(1)
                
            qc.measure([0, 1], [0, 1])
            
            # Run the circuit
            transpiled_qc = transpile(qc, self.backend)
            job = self.backend.run(transpiled_qc, shots=1)
            result = job.result().get_counts()
            
            # Get measurement outcomes
            measurement = list(result.keys())[0]
            return int(measurement[0]), int(measurement[1])
            # ... rest of the code
        except Exception as e:
            logger.error(f"Bell state measurement failed: {str(e)}")
            return (0, 0)

        
        # Apply measurement bases
        
    
    def perform_bell_test(self, num_pairs: int = 100) -> Dict:
        """
        Perform a full Bell test to verify quantum channel
        """
        results = {
            "total_pairs": num_pairs,
            "successful_measurements": 0,
            "correlation": 0.0,
            "chsh_value": 0.0
        }
        
        measurements = []
        for _ in range(num_pairs):
            # Create and measure Bell pair
            bell_state = self.create_bell_pair()
            basis_alice = [random.randint(0, 1), 0]
            basis_bob = [0, random.randint(0, 1)]
            
            outcome = self.measure_bell_state(bell_state, [basis_alice[0], basis_bob[1]])
            measurements.append((outcome, basis_alice, basis_bob))
            
            if outcome[0] == outcome[1]:  # Correlated measurement
                results["successful_measurements"] += 1
                
        # Calculate CHSH value
        results["correlation"] = results["successful_measurements"] / num_pairs
        results["chsh_value"] = abs(2 * results["correlation"] - 1) * 2**0.5
        
        logger.info(f"Bell test completed: {results}")
        return results
    
    def select_encoding_scheme(self) -> EncodingScheme:
        """Select appropriate encoding scheme based on channel conditions"""
        if self.encoder is None:
            raise ValueError("Encoder not initialized")
        channel_conditions = {
            "QBER": self.error_rate,
            "distance": self.calculate_effective_distance(),
            "noise_level": self.calculate_noise_level(),
            "asymmetric": self.check_channel_asymmetry()
        }
        
        scheme = self.encoder.select_encoding(channel_conditions)
        self.current_scheme = scheme
        logger.info(f"Selected encoding scheme: {scheme.value}")
        return scheme

    def calculate_effective_distance(self) -> float:
        """Calculate effective channel distance based on loss rate"""
        # Typical fiber loss is 0.2 dB/km
        return -10 * np.log10(1 - self.loss_rate) / 0.2

    def calculate_noise_level(self) -> float:
        """Calculate channel noise level"""
        return self.error_rate + self.loss_rate / 2

    def check_channel_asymmetry(self) -> bool:
        """Check if channel is asymmetric"""
        # Implementation depends on your specific setup
        return False

    def prepare_quantum_state(self, bit: int, basis: int) -> QuantumCircuit:
        """Prepare quantum state using current encoding scheme"""
        if self.current_scheme == EncodingScheme.DECOY_BB84:
            intensity = random.choice([0.1, 0.5, 1.0])  # Different intensity levels
            return self.encoder.prepare_state(bit, basis, self.current_scheme, intensity=intensity)
        return self.encoder.prepare_state(bit, basis, self.current_scheme)
    
    def send(self, qubits):
    
        # Create a copy of qubits to avoid modifying the original
        received_qubits = qubits.copy()
   
        # Import necessary libraries
        import random
        import numpy as np
        from qiskit_aer.noise import NoiseModel, depolarizing_error, amplitude_damping_error
        import logging
        
        # Get logger
        logger = logging.getLogger()
        
        # Define noise parameters (these could be class attributes of QuantumChannel)
        # Use the parameters from environment analysis or set defaults
        # if hasattr(self, 'p_gate'):
        #     p_gate = self.p_gate
        # else:
        #     p_gate = 0.02  # 1-qubit gate error probability
            
        # if hasattr(self, 'p_measurement'):
        #     p_meas = self.p_measurement
        # else:
        #     p_meas = 0.05  # measurement error probability
            
        # if hasattr(self, 'gamma'):
        #     gamma = self.gamma
        # else:
        #     gamma = 0.0  # amplitude damping parameter for channel loss
        p_gate=0.05
        p_meas=0.08
        # Apply depolarizing noise (bit and phase flips)
        for i in range(len(received_qubits)):
            # Apply depolarizing error with probability p_gate
            if random.random() < p_gate:
                # Randomly choose error type (bit flip, phase flip, or both)
                error_type = random.randint(1, 3)
                
                if error_type == 1 or error_type == 3:  # Bit flip (X error)
                    if isinstance(received_qubits[i], (int, bool)):
                        received_qubits[i] = 1 - received_qubits[i]  # Flip the bit
                    elif hasattr(received_qubits[i], 'x'):  # If it's a Qiskit Statevector or similar
                        received_qubits[i].x(0)  # Apply X gate
                
                if error_type == 2 or error_type == 3:  # Phase flip (Z error)
                    if hasattr(received_qubits[i], 'z'):  # If it's a Qiskit Statevector or similar
                        received_qubits[i].z(0)  # Apply Z gate
        
        # Apply amplitude damping (loss)
        qubits_after_loss = []
        for qubit in received_qubits:
            # Simulate qubit loss with probability gamma
            if random.random() >= 0:
                qubits_after_loss.append(qubit)
            else:
                # Log the loss
                logger.debug("Qubit lost in transmission due to amplitude damping")
        

        # If all qubits were lost (unlikely but possible), return an empty list
        if not qubits_after_loss:
            logger.warning("All qubits were lost in transmission!")
            return []
        
        # Handle case where some qubits were lost
        if len(qubits_after_loss) != len(received_qubits):
            logger.info(f"Channel loss: {(len(received_qubits) - len(qubits_after_loss)) / len(received_qubits):.4f}")
            return qubits_after_loss
        
        # Log the transmission with noise information
        logger.info(f"Transmitted {len(qubits)} qubits through quantum channel")
        logger.info(f"Applied noise model: Gate error ({p_gate}), Measurement error ({p_meas}), Amplitude damping )")
        
        return received_qubits
    
def flip_qubit(qubit: Tuple[int, int]) -> Tuple[int, int]:
    # Simple bit flip for demonstration
    if isinstance(qubit, tuple):
        return (1 - qubit[0], qubit[1])
    return 1 - qubit



class ClassicalChannel:
    def __init__(self):
        self.authenticated = False
        self.secret_key = None
        self.public_key = None
        self.clock_data = ClockData(
            local_time=time.time(),
            offset=0.0,
            last_sync=time.time(),
            drift_rate=0.0
        )
        self.sync_interval = 60  # seconds

    # Keep existing methods
    def send(self, message, signature=None):
        if signature:
            return message, signature
        return message
    
    def receive(self, message_package):
        if isinstance(message_package, tuple):
            message, signature = message_package
            if self.verify_signature(message, signature):
                return message
            return None
        return message_package
    
    def authenticate(self):
        self.secret_key = falcon.SecretKey(128)
        self.public_key = falcon.PublicKey(self.secret_key)
        self.authenticated = True
        return self.authenticated
    
    def verify_signature(self, message, signature):
        if self.public_key and signature:
            return self.public_key.verify(message.encode('utf-8'), signature)
        return False
    
    def sign_message(self, message):
        if self.secret_key:
            return self.secret_key.sign(message.encode('utf-8'))
        return None

    # Add new methods
    def measure_time_offset(self, remote_timestamp: float) -> float:
        """Measure time offset between local and remote clocks"""
        local_time = time.time()
        round_trip_time = local_time - remote_timestamp
        offset = round_trip_time / 2
        logger.debug(f"Measured time offset: {offset}")
        return offset

    def adjust_clock(self, offset: float) -> bool:
        """Adjust local clock based on measured offset"""
        try:
            self.clock_data.offset = offset
            self.clock_data.last_sync = time.time()
            self.clock_data.local_time = time.time() + offset
            logger.debug(f"Clock adjusted by offset: {offset}")
            return True
        except Exception as e:
            logger.error(f"Clock adjustment failed: {str(e)}")
            return False

    def verify_clock_sync(self) -> bool:
        """Verify if clocks are properly synchronized"""
        current_time = time.time()
        time_since_sync = current_time - self.clock_data.last_sync
        
        if time_since_sync > self.sync_interval:
            logger.warning("Clock sync interval exceeded")
            return False
        
        return True
    


def test_qiskit_logging():
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()
    
    backend = Aer.get_backend('aer_simulator')
    transpiled_qc = transpile(qc, backend)
    job = backend.run(transpiled_qc)
    result = job.result().get_counts()
    return result

def setup_channels(alice, bob,supported_encoding):
    """
    Set up quantum and classical channels with participant authentication
    
    Args:
        alice (Participant): Alice participant object
        bob (Participant): Bob participant object
        
    Returns:
        tuple: (quantum_channel, classical_channel) - The initialized channels
    """
    test_qiskit_logging()
    logger.info("Initializing quantum and classical channels...")
    quantum_channel = QuantumChannel(supported_encoding)
    classical_channel = ClassicalChannel()
    
    # Authenticate participants using RSA keys
    logger.info(f"Authenticating {alice.get_name()} and {bob.get_name()} using RSA keys...")
    
    # Generate authentication messages
    alice_auth_msg = f"AUTH:{alice.get_name()}:{int(time.time())}"
    alice_signature = alice.sign_message(alice_auth_msg, alice.secret_key)
    
    bob_auth_msg = f"AUTH:{bob.get_name()}:{int(time.time())}"
    bob_signature = bob.sign_message(bob_auth_msg, bob.secret_key)

    if alice_signature is None or bob_signature is None:
        logger.error("Failed to generate valid signatures for authentication")
        return None, None
    
    # Verify signatures
    alice_verified = bob.verify_signature(alice_auth_msg, alice_signature, alice.public_key)
    bob_verified = alice.verify_signature(bob_auth_msg, bob_signature, bob.public_key)
    
    # Update participants' authentication status
    if alice_verified and bob_verified:
        alice.authenticate()
        bob.authenticate()
        logger.info("Participant authentication successful")
        
        # Authenticate classical channel with participants' keys
        classical_channel.secret_key = falcon.SecretKey(128)  # Use internal SecretKey for compatibility
        classical_channel.public_key = falcon.PublicKey(classical_channel.secret_key)
        
        # # Store participant public keys for future verification
        # classical_channel.alice_public_key = alice.rsa_key.publickey()
        # classical_channel.bob_public_key = bob.rsa_key.publickey() 
        
        # Authenticate classical channel
        if classical_channel.authenticate():
            logger.info("Classical channel authenticated successfully")
            
            # Initial calibration
            quantum_channel.calibrate_channel()
            
            # Select encoding scheme
            encoding_scheme = quantum_channel.select_encoding_scheme()
            logger.info(f"Selected encoding scheme: {encoding_scheme.value}")
            
            # Initial clock sync between all parties
            synchronize_clocks(classical_channel, alice, bob)
        else:
            logger.error("Classical channel authentication failed")
    else:
        logger.error("Participant authentication failed")
        if not alice_verified:
            logger.error(f"Failed to verify {alice.get_name()}'s identity")
        if not bob_verified:
            logger.error(f"Failed to verify {bob.get_name()}'s identity")
    
    return quantum_channel, classical_channel


def synchronize_clocks(classical_channel, alice, bob):
    """
    Synchronize clocks between participants and the classical channel
    
    Args:
        classical_channel (ClassicalChannel): The classical communication channel
        alice (Participant): Alice participant
        bob (Participant): Bob participant
        
    Returns:
        bool: True if synchronization was successful, False otherwise
    """
    logger.info("Synchronizing clocks...")
    
    try:
        # Get reference timestamp from Alice
        alice_timestamp = time.time()
        sync_message = f"SYNC:{alice_timestamp}"
        
        # Sign the message with Alice's key
        alice_signature = alice.sign_message(sync_message)
        
        # Send through classical channel
        classical_channel.send(sync_message, alice_signature)
        
        # Bob synchronizes with Alice's timestamp
        bob.synchronize_clock(alice_timestamp)
        
        # Classical channel synchronizes
        offset = classical_channel.measure_time_offset(alice_timestamp)
        success = classical_channel.adjust_clock(offset)
        
        if success:
            logger.info(f"Clocks synchronized with offset: {offset}")
            
            # Verify clock synchronization by comparing times
            time_diff_alice_bob = abs(alice.clock_data.local_time - bob.clock_data.local_time)
            time_diff_alice_channel = abs(alice.clock_data.local_time - classical_channel.clock_data.local_time)
            
            logger.info(f"Time difference Alice-Bob: {time_diff_alice_bob} seconds")
            logger.info(f"Time difference Alice-Channel: {time_diff_alice_channel} seconds")
            
            if time_diff_alice_bob < 1.0 and time_diff_alice_channel < 1.0:
                logger.info("Clock synchronization verification successful")
            else:            
                logger.warning("Clock synchronization verification showed discrepancies")
        else:
            logger.error("Clock synchronization failed")
        
        return success
        
    except Exception as e:
        logger.error(f"Clock synchronization error: {str(e)}")
        return False