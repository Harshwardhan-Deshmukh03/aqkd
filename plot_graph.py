import matplotlib.pyplot as plt
from main import main  # Assuming main.py is structured to allow this import

def run_aqkd_for_key_lengths(key_lengths):
    results = []
    for key_length in key_lengths:
        print(f"Running AQKD for key length: {key_length}")
        # Run the AQKD protocol with the specified key length
        final_key = main(['--key-length', str(key_length)])
        
        # Store the result (length of the final key)
        if final_key is not None:
            results.append(len(final_key))
        else:
            results.append(0)  # If key verification failed, store 0

    return results

def plot_results(key_lengths, key_lengths_results):
    plt.figure(figsize=(10, 6))
    plt.plot(key_lengths, key_lengths_results, marker='o')
    plt.title('Key Length vs. Number of Bits Transmitted')
    plt.xlabel('Number of Bits Transmitted')
    plt.ylabel('Key Length')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # Define a range of key lengths to test
    key_lengths = [128,256, 512, 1024, 2048]
    
    # Run the AQKD protocol for each key length
    key_lengths_results = run_aqkd_for_key_lengths(key_lengths)
    
    # Plot the results
    plot_results(key_lengths, key_lengths_results)