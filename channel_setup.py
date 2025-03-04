import hashlib
import time
import random
import numpy as np
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass
from datetime import datetime, timedelta
from utils.logger import get_logger
from utils.falcon import *

logger = get_logger(__name__)

import logging
logging.getLogger('qiskit').setLevel(logging.WARNING)

@dataclass
class ClockData:
    local_time: float
    offset: float
    last_sync: float
    drift_rate: float




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


class QuantumChannel:
    def __init__(self, error_rate=0.03, loss_rate=0.1):
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
        super().__init__()
        self.decoy_states: List[DecoyState] = []
        self.bell_pairs: List[BellState] = []
        self.backend = Aer.get_backend('aer_simulator')
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
        qc = bell_state.circuit.copy()
        
        # Apply measurement bases
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
    









    
    
def flip_qubit(qubit):
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
        self.secret_key = SecretKey(1024)
        self.public_key = PublicKey(self.secret_key)
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
    

def setup_channels():
    logger.info("Initializing quantum and classical channels...")
    quantum_channel = QuantumChannel()
    classical_channel = ClassicalChannel()
    
    # Authenticate classical channel
    if classical_channel.authenticate():
        logger.info("Classical channel authenticated successfully")
        
        # Initial calibration
        quantum_channel.calibrate_channel()
        
        # Initial clock sync
        synchronize_clocks(classical_channel)
        
        # Setup decoy states
        decoy_intensities = [0.1, 0.2, 0.5]  # Different intensity levels
        decoy_states = quantum_channel.generate_decoy_states(100, decoy_intensities)
        
        # Perform initial Bell test
        bell_test_results = quantum_channel.perform_bell_test(100)
        if bell_test_results["chsh_value"] > 2:
            logger.info("Bell test passed - quantum channel verified")
        else:
            logger.warning("Bell test failed - possible classical channel simulation")
    else:
        logger.error("Classical channel authentication failed")
    
    return quantum_channel, classical_channel



def synchronize_clocks(classical_channel):
    logger.info("Synchronizing clocks...")
    
    # Send initial timestamp
    timestamp = time.time()
    message = f"SYNC:{timestamp}"
    signature = classical_channel.sign_message(message)
    classical_channel.send(message, signature)
    
    # Measure and adjust offset
    offset = classical_channel.measure_time_offset(timestamp)
    success = classical_channel.adjust_clock(offset)
    
    if success:
        logger.info(f"Clocks synchronized with offset: {offset}")
    else:
        logger.error("Clock synchronization failed")
    
    return success