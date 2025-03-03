# Adaptive Quantum Key Distribution (AQKD) System

This repository contains an implementation of an Adaptive Quantum Key Distribution system that combines quantum cryptography, classical post-processing, and machine learning for enhanced security and performance.

## Core Components

1. **Seven-Phase QKD Framework**
   - Combines quantum and classical safeguards
   - Machine learning-based adaptations
   - Flexible post-processing
   - Real-time QBER monitoring
   - Falcon-based quantum digital signatures

2. **Technology Stack**
   - Qiskit for quantum operations
   - Falcon for post-quantum digital signatures
   - Basic ANN for adaptive learning
   - Python-based implementation

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Basic usage:

```bash
python main.py
```

With options:

```bash
python main.py --key-length 2048 --decoy-states --debug
```

## Architecture

The system implements a seven-phase QKD framework:

1. **Channel Setup and Authentication**
2. **Environmental Analysis**
3. **Quantum Data Transmission**
4. **Measurement and Data Acquisition**
5. **Error Correction (Cascade Protocol)**
6. **Privacy Amplification**
7. **Key Verification**

## Testing

Run the tests with:

```bash
python -m unittest discover tests
```

## Security Features

- Post-quantum digital signatures using Falcon
- Adaptive encoding based on channel conditions
- Real-time QBER monitoring
- Cascade error correction protocol
- Toeplitz matrix-based privacy amplification

## Performance Considerations

The system optimizes key rate by:
- Adapting to channel conditions
- Monitoring and reacting to QBER in real-time
- Using machine learning for parameter optimization

## License

MIT License