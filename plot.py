import subprocess
import json
import matplotlib.pyplot as plt
from main import main

def run_main(key_length, mitm=False):
    """Runs main.py with specified key length and optional MITM attack"""
    cmd = ["python", "main.py", "--key-length", str(key_length)]
    if mitm:
        cmd.append("--mitm")
    
    # result = subprocess.run(cmd, capture_output=True, text=True)
    if mitm:
        qber = main(['--key-length', str(key_length), '--mitm'])
    else:
        qber = main(['--key-length', str(key_length)])

    print("Raw Output:", qber )  # Debugging line

    return qber


def plot_qber_vs_key_length():
    """Runs main.py with different key lengths, collects QBER values, and plots them."""
    key_lengths = [128, 256, 512, 1024]
    qber_mitm = []
    qber_no_mitm = []

    # Collect QBER values
    for key_length in key_lengths:
        print(f"Running AQKD for key length: {key_length}")
        qber_no_mitm.append(run_main(key_length, mitm=False))
        qber_mitm.append(run_main(key_length, mitm=True))


    # Plotting
    plt.figure(figsize=(8, 6))
    plt.plot(key_lengths, qber_mitm, marker='o', linestyle='-', color='r', label='With MITM')
    plt.plot(key_lengths, qber_no_mitm, marker='s', linestyle='--', color='b', label='Without MITM')

    plt.xlabel("Key Length (bits)")
    plt.ylabel("Quantum Bit Error Rate (QBER)")
    plt.title("QBER vs Key Length")
    plt.ylim(0, 1)  # QBER ranges from 0 to 1
    plt.xticks(key_lengths)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.show()

if __name__ == "__main__":
    plot_qber_vs_key_length()
