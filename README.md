# Adaptive Quantum Key Distribution (AQKD) System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Qiskit](https://img.shields.io/badge/Qiskit-0.45%2B-673AB7.svg)](https://qiskit.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0-EE4C2C.svg)](https://pytorch.org/)
[![PennyLane](https://img.shields.io/badge/PennyLane-0.40.0-gray.svg)](https://pennylane.ai/)

This repository contains an advanced **Adaptive Quantum Key Distribution (AQKD)** system combining quantum cryptography, classical post-processing, and a **Quantum Neural Network (QNN) Machine Learning Model** for optimizing security dynamically.

Instead of statically selecting a QKD encoding, this framework actively evaluates environmental noise, Quantum Bit Error Rate (QBER), and channel loss, mapping them via a highly trained PennyLane-based Quantum Neural Network to the optimal encoding structure (such as BB84, Decoy-BB84, or Six-State).

---

## Table of Contents

- [Overview](#-overview)
- [System Architecture](#-system-architecture)
- [Adaptive Machine Learning Component](#-adaptive-machine-learning-component)
- [Security Features](#-security-features)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Usage & CLI Arguments](#-usage--cli-arguments)
- [Algorithm & Flow Details](#-algorithm--flow-details)

---

## Overview

The AQKD system securely connects an origin ("Alice") and destination ("Bob") through a simulated noisy quantum channel and a classical communication channel. It features an adaptive Block-Wise transmission mechanic:
If noise properties spike severely (QBER > `0.85`), the protocol retransmits or aborts; if subtle environmental shifts occur, it can swap the encoding mechanism (e.g., standard BB84 substituting for 6-State) strictly upon the guidance of the QNN Model on a **block-by-block** basis.

The execution handles everything from channel modeling (via `qiskit_aer.noise`) to privacy amplification algorithms using Toeplitz matrices and ends with classical key verification hashes.

---

## System Architecture

The protocol is explicitly broken out into **Seven Phases**:

1. **Channel Setup and Authentication:** Initiating a modeled noisy quantum channel and a classical authentication channel.
2. **Environmental Analysis:** Evaluating QBER, Gate Error Probability (`p_gate`), and Channel Decay rate (`gamma`) using Qiskit Aer simulators.
3. **Adaptive ML Engine:** PyTorch & PennyLane-based Neural Network infers the safest encoding method based on environmental variables.
4. **Quantum Data Transmission:** Qubits are transmitted in defined blocks with dynamically applied Decoy states if required.
5. **Measurement & Data Sifting:** Filtering incompatible basis configurations to produce the initial "Sifted Key".
6. **Error Correction (Cascade Protocol):** An interactive classical communication procedure to perfectly align keys through parity checksums.
7. **Privacy Amplification & Verification:** Shrinks the corrected key to mathematically destroy partial eavesdropped information (using Universal Hashing), validating the final output.

---

## Adaptive Machine Learning Component

A core innovation in this repository is the inclusion of a PyTorch + PennyLane hybrid model for live adaptation.

Located in `adaptive_encoding.py`, a `QNN-based Module` monitors environments specifically evaluating:
- Initial QBER checks
- Channel degradation estimates
- Hardware operation error probabilities

The QNN performs entangling matrix operations over 4 qubits before utilizing a classical linear layer and a softmax wrapper to rank the predicted efficiency of 6 encoding states: `["BB84", "DECOY_BB84", "SIX_STATE", "EIGHT_STATE", "E91", "THREE_PLUS_ONE"]`.

---

## Security Features

- **Post-Quantum Authentication Support:** Prepares integration for signature generation.
- **Dynamic Decoy States:** Adds random decoy signals with alternative intensities to detect Photon Number Splitting (PNS) attacks dynamically via `main.py` `--decoy-states` flag.
- **Cascade Error Correction Protocol:** Robust multi-pass error correction scheme resolving differing bits securely.
- **Toeplitz Matrix Privacy Amplification:** Guarantees absolute unpredictability on the resulting truncated key.

---

## Prerequisites

- **Python 3.8+**
- Git

Ensure you have a hardware setup or an OS capable of running standard Python ML operations.

---

## Installation 

1. **Clone the project:**
   ```bash
   git clone https://github.com/example/aqkd.git
   cd aqkd
   ```

2. **(Optional but Recommended) Setup a Virtual Environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate       # For Linux/Mac
   venv\Scripts\activate.bat      # For Windows
   ```

3. **Install Requirements:**
   ```bash
   pip install -r requirements.txt
   ```
   *Note: This will install massive compute libraries such as PyTorch, PennyLane, and Qiskit.*

---

## Usage & CLI Arguments

To boot into the default simulation process, simple use the `main.py` entrypoint.

### Basic Run
```bash
python main.py
```

### Advanced Usage

You can control key mechanics directly from CLI flags.

```bash
python main.py [OPTIONS]
```

| Flag | Type | Description | Default |
| :--- | :--- | :--- | :--- |
| `--key-length` | `int` | Total length of the requested quantum key. | 1024 |
| `--block-size` | `int` | Size of transmission blocks (for block-by-block adaptation). | 128 |
| `--test-size` | `int` | Number of test qubits checked during environment analysis. | 32 |
| `--decoy-states` | `flag`| Use protective Decoy states (varies intensity) on blocks. | False |
| `--debug` | `flag`| Triggers extensive DEBUG logging across the application. | False |
| `--mitm` | `flag`| Simulate a harsh Man-In-The-Middle attack dropping and measuring qubits.| False |

**Example configuration producing an adaptive, decoy-protected simulation targeting 2048 requested bits with debugging enabled:**
```bash
python main.py --key-length 2048 --block-size 256 --decoy-states --debug
```

### Analyzing Metrics
You can test multi-run visual analysis using the internal matplotlib tools:
```bash
python plot_graph.py 
```
This simulates runs scaled up to 2048 bits and visualizes the relation indicating bits preserved against loss mapping.

---

## Algorithm & Flow Details

The `main.py` control loop executes via these strict algorithmic bounds:

1. A `temp_results.json` actively caches live results.
2. Based on evaluated parameters, Qubits are encoded physically onto Qiskit computational circuits (`x` / `h` rotational gates defining diagonal/computational bases).
3. At the end of every block evaluation window (`--block-size`), if the `qber` threshold meets limits (`>0.80`), it aborts, if it exceeds base standards, it will invoke `select_encoding()`, utilizing the trained ML model (`models/qnn_model.pth`) mapped to (`models/qnn_scaler.pkl`) to decide if transitioning into states like "Six-State" guarantees a safer remainder of execution.
4. Alice and Bob compute their matching strings locally. Parity checking resolves discrepancies, Hashing slices away partial leaks, and keys are mathematically verified without direct exposure.

