import json
import time
import os
import random
from tabulate import tabulate
from main import main

# File for storing results
RESULTS_FILE = "qkd_results2.json"

def init_results_file():
    """Initialize the results file if it doesn't exist"""
    if not os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, 'w') as f:
            json.dump([], f)

def save_run_result(run_data):
    """Append a single run result to the file"""
    try:
        # Read existing data
        with open(RESULTS_FILE, 'r') as f:
            all_results = json.load(f)
        
        # Append new data
        all_results.append(run_data)
        
        # Write back to file
        with open(RESULTS_FILE, 'w') as f:
            json.dump(all_results, f, indent=2)
            
        print(f"Result saved: Try {run_data['try']}, Key Length {run_data['initial_key_length']}, MITM: {run_data['mitm']}")
    except Exception as e:
        print(f"Error saving result to file: {e}")

def run_main(key_length, try_num, mitm=False, decoy_states=False):
    """Runs main.py with specified parameters and saves results"""
    args = ['--key-length', str(key_length)]
    if mitm:
        args.append('--mitm')
    if decoy_states:
        args.append('--decoy-states')
    
    # Ensure different randomization for each run
    random.seed(time.time() + try_num + key_length + (1000 if mitm else 0))
    
    print(f"Running: Try {try_num}, Key Length {key_length}, MITM: {mitm}, Decoy: {decoy_states}")
    
    start_time = time.time()
    try:
        qber = main(args)
        execution_time = time.time() - start_time
        
        # Read detailed results from the run
        run_data = {}
        try:
            with open('temp_results.json', 'r') as f:
                run_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            # If file doesn't exist or is invalid, create an empty dict
            pass
        
        # Build complete result dictionary
        result = {
            'try': try_num,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'initial_key_length': key_length,
            'mitm': mitm,
            'decoy_states': decoy_states,
            'qber': qber,
            'execution_time': execution_time,
            **run_data  # Include all data from the run
        }
        
        # Save to file
        save_run_result(result)
        
        return result
    except Exception as e:
        print(f"Error in run: {e}")
        # Save error information
        error_result = {
            'try': try_num,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'initial_key_length': key_length,
            'mitm': mitm,
            'decoy_states': decoy_states,
            'error': str(e),
            'execution_time': time.time() - start_time
        }
        save_run_result(error_result)
        return error_result

def format_cell(data):
    """Format cell data for table display"""
    if 'error' in data:
        return f"ERROR: {data['error'][:30]}..."
    
    # Calculate privacy amplification factor
    sifted_key_len = data.get('sifted_key_length', 'N/A')
    final_key_len = data.get('final_key_length', 'N/A')
    
    if isinstance(sifted_key_len, (int, float)) and isinstance(final_key_len, (int, float)) and sifted_key_len > 0:
        pa_factor = round(final_key_len / sifted_key_len, 2)
    else:
        pa_factor = 'N/A'
    
    # Format decoy info if present
    decoy_info = f", Decoys: {data.get('decoy_count', 'N/A')}" if data.get('decoy_states', False) else ""
    
    # Build cell content
    cell = (
        f"Init: {data.get('initial_key_length', 'N/A')}\n"
        f"Sifted: {data.get('sifted_key_length', 'N/A')}\n"
        f"Final: {data.get('final_key_length', 'N/A')}\n"
        f"QBER: {data.get('qber', 'N/A')}%\n"
        f"Method: {data.get('encoding_method', 'N/A')} {data.get('encoding_dimension', 'N/A')}D\n"
        f"PA Factor: {pa_factor}\n"
        f"Time: {data.get('execution_time', 'N/A'):.2f}s\n"
        f"Success: {data.get('verification_success', 'N/A')}{decoy_info}"
    )
    
    return cell

def load_results():
    """Load all stored results"""
    try:
        with open(RESULTS_FILE, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []

def generate_tables_from_file():
    """Generate tables from saved results file"""
    results = load_results()
    
    if not results:
        print("No results found in file.")
        return
    
    # Group by decoy states setting
    for decoy in [False, True]:
        decoy_results = [r for r in results if r.get('decoy_states', False) == decoy]
        if not decoy_results:
            continue
            
        decoy_text = "with decoy states" if decoy else "without decoy states"
        
        # Group by MITM setting
        for mitm in [False, True]:
            mitm_results = [r for r in decoy_results if r.get('mitm', False) == mitm]
            if not mitm_results:
                continue
                
            mitm_text = "with MITM" if mitm else "without MITM"
            
            # Get unique key lengths and try numbers
            key_lengths = sorted(set(r.get('initial_key_length', 0) for r in mitm_results))
            try_nums = sorted(set(r.get('try', 0) for r in mitm_results))
            
            # Create table data
            table_data = []
            for try_num in try_nums:
                row = [f'Try {try_num}']
                
                for key_length in key_lengths:
                    # Find matching result
                    matches = [r for r in mitm_results if r.get('try') == try_num and r.get('initial_key_length') == key_length]
                    if matches:
                        cell = format_cell(matches[0])
                    else:
                        cell = "No data"
                    
                    row.append(cell)
                
                table_data.append(row)
            
            # Create headers
            headers = ["No. Try"] + [f"Initial Qbits={k}" for k in key_lengths]
            
            # Print table
            print(f"\nTable: QKD Results {decoy_text} ({mitm_text})")
            print(tabulate(table_data, headers=headers, tablefmt="grid"))

def generate_tables():
    """Generates new data and prints tables for QKD results"""
    key_lengths = [256, 512, 1024, 2048]
    tries = 3
    use_decoy = [False, True]  # Run with and without decoy states
    
    # Initialize results file
    init_results_file()
    
    # Run all combinations
    for try_num in range(1, tries + 1):
        for decoy in use_decoy:
            for mitm in [False, True]:
                for key_length in key_lengths:
                    run_main(key_length, try_num, mitm, decoy)
    
    # Generate tables from all results
    generate_tables_from_file()

if __name__ == "__main__":
    # You can either generate new data:
    generate_tables()
    
    # Or just display existing data:
    # generate_tables_from_file()