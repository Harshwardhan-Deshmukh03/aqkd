import json
import time
import os
import random
from tabulate import tabulate
from main import main

# File for storing results
RESULTS_FILE = "qkd_block_results.json"

def init_results_file():
    """Initialize the results file if it doesn't exist"""
    if not os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, 'w') as f:
            json.dump([], f)

def save_run_result(run_data):
    """Append a single run result to the file"""
    try:
        with open(RESULTS_FILE, 'r') as f:
            all_results = json.load(f)
        all_results.append(run_data)
        with open(RESULTS_FILE, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"Result saved: Try {run_data['try']}, Key Length {run_data['key_length']}, MITM: {run_data['mitm']}")
    except Exception as e:
        print(f"Error saving result to file: {e}")

def run_main(key_length, block_size, try_num, mitm=True, decoy_states=True):
    """Runs main.py with specified parameters and saves results"""
    args = [
        '--key-length', str(key_length),
        '--block-size', str(block_size),
        '--test-size', '32'
    ]
    if mitm:
        args.append('--mitm')
    if decoy_states:
        args.append('--decoy-states')
    
    random.seed(time.time() + try_num + key_length + (1000 if mitm else 0))
    
    print(f"Running: Try {try_num}, Key Length {key_length}, Block Size {block_size}, MITM: {mitm}, Decoy: {decoy_states}")
    
    start_time = time.time()
    try:
        qber = main(args)
        execution_time = time.time() - start_time
        
        run_data = {}
        try:
            with open('temp_results.json', 'r') as f:
                run_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        
        result = {
            'try': try_num,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'key_length': key_length,
            'block_size': block_size,
            'mitm': mitm,
            'decoy_states': decoy_states,
            'qber': qber,
            'execution_time': execution_time,
            **run_data
        }
        
        save_run_result(result)
        return result
    except Exception as e:
        print(f"Error in run: {e}")
        error_result = {
            'try': try_num,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'key_length': key_length,
            'block_size': block_size,
            'mitm': mitm,
            'decoy_states': decoy_states,
            'error': str(e),
            'execution_time': time.time() - start_time
        }
        save_run_result(error_result)
        return error_result

def load_results():
    """Load all stored results"""
    try:
        with open(RESULTS_FILE, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []

def generate_block_table():
    """Generate table of QBER and adaptive responses across blocks"""
    key_configs = [
        {'key_length': 512, 'block_size': 64},   # 512 / 64 = 8 blocks
        {'key_length': 1024, 'block_size': 128}  # 1024 / 128 = 8 blocks
    ]
    try_num = 1
    
    init_results_file()
    
    # Run simulations
    for config in key_configs:
        run_main(config['key_length'], config['block_size'], try_num, mitm=True, decoy_states=True)
    
    # Load results
    results = load_results()
    if not results:
        print("No results found in file.")
        return
    
    # Prepare table data
    table_data = []
    headers = ["Key Length", "Block Number", "QBER", "Encoding Method", "Action Taken"]
    
    for result in results:
        if 'error' in result:
            continue  # Skip failed runs
        key_length = result.get('key_length', 'N/A')
        block_logs = result.get('block_logs', [])
        
        for i in range(8):
            if i < len(block_logs):
                log = block_logs[i]
                row = [
                    str(key_length) if i == 0 else "",
                    log['block_number'],
                    f"{log['qber']:.4f}" if isinstance(log['qber'], (int, float)) else "N/A",
                    log['encoding_method'],
                    log['action']
                ]
            else:
                row = [str(key_length) if i == 0 else "", i + 1, "N/A", "N/A", "N/A"]
            table_data.append(row)
    
    # Print plain text table
    print(f"\nTable: QBER and Adaptive Responses Across Transmission Blocks (Try {try_num}, MITM, Decoy States)")
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    # Generate LaTeX table
    latex_table = """
\\begin{table}[h]
\\centering
\\caption{Table Z: QBER and Adaptive Responses Across Transmission Blocks. The table illustrates the protocol’s real-time adaptation to varying channel conditions.}
\\label{tab:qber_adaptive}
\\begin{tabular}{|c|c|c|l|c|}
\\hline
\\textbf{Key Length} & \\textbf{Block Number} & \\textbf{Measured QBER} & \\textbf{Selected Encoding Method} & \\textbf{Action Taken} \\\\
\\hline
"""
    for row in table_data:
        latex_table += f"{row[0]} & {row[1]} & {row[2]} & {row[3]} & {row[4]} \\\\\n\\hline\n"
    
    latex_table += """
\\end{tabular}
\\end{table}
"""
    
    with open('qber_adaptive_table.tex', 'w') as f:
        f.write(latex_table)
    
    print("\nLaTeX table saved to 'qber_adaptive_table.tex'")

if __name__ == "__main__":
    generate_block_table()