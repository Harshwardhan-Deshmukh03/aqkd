import json
from tabulate import tabulate
from main import main

def run_main(key_length, mitm=False):
    """Runs main.py with specified key length and optional MITM attack"""
    if mitm:
        result = main(['--key-length', str(key_length), '--mitm'])
    else:
        result = main(['--key-length', str(key_length)])
    return result

def generate_tables():
    """Generates and prints tables for QBER with and without MITM attack."""
    key_lengths = [256, 512, 1024, 2048]
    tries = 5

    data_no_mitm = []
    data_mitm = []
    
    for i in range(tries):
        row_no_mitm = [f'Try {i+1}']
        row_mitm = [f'Try {i+1}']
        
        for key_length in key_lengths:
            qber_no_mitm = run_main(key_length, mitm=False)
            qber_mitm = run_main(key_length, mitm=True)
            row_no_mitm.extend([key_length, f"{qber_no_mitm}%"])
            row_mitm.extend([key_length, f"{qber_mitm}%"])
        
        data_no_mitm.append(row_no_mitm)
        data_mitm.append(row_mitm)
    
    headers = ["No. Try"] + [f"Initial Qbits={k}\nLen. Key | QBER" for k in key_lengths]
    
    print("Table 1: The final key length and QBER in case of MitM attack")
    print(tabulate(data_no_mitm, headers=headers, tablefmt="grid"))
    print()
    print("Table 2: The final key length and QBER with the presence of MitM attack")
    print(tabulate(data_mitm, headers=headers, tablefmt="grid"))

if __name__ == "__main__":
    generate_tables()
