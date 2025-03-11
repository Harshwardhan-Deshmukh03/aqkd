import os
import torch
import numpy as np
import joblib
import pennylane as qml
import torch.nn as nn

# Recreate the QNN model architecture with 6 classes instead of 5
class QNN(nn.Module):
    def __init__(self):
        super(QNN, self).__init__()
        n_qubits = 4  # Keep 4 qubits for input features
        self.quantum_weights = nn.Parameter(torch.randn(3, n_qubits))
        self.post_processing = nn.Linear(n_qubits, 6)  # Changed to 6 output classes
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        # Quantum circuit definition (unchanged)
        def quantum_circuit(inputs, weights):
            dev = qml.device("default.qubit", wires=4)
            
            @qml.qnode(dev)
            def circuit(inputs, weights):
                # Encode input features
                for i in range(4):
                    qml.RY(inputs[i], wires=i)
                
                # Apply parameterized quantum circuit
                for i in range(4):
                    qml.RY(weights[0, i], wires=i)
                
                # Entangling layer
                for i in range(3):
                    qml.CNOT(wires=[i, i + 1])
                qml.CNOT(wires=[3, 0])  # Connect last qubit to first
                
                # Second layer of rotations
                for i in range(4):
                    qml.RY(weights[1, i], wires=i)
                
                # Entangling layer
                for i in range(3):
                    qml.CNOT(wires=[i, i + 1])
                qml.CNOT(wires=[3, 0])
                
                # Final rotation layer
                for i in range(4):
                    qml.RY(weights[2, i], wires=i)
                
                # Return expectation values
                return [qml.expval(qml.PauliZ(i)) for i in range(4)]
            
            return circuit(inputs, weights)
        
        # Apply quantum circuit to each input sample
        q_out = torch.zeros(x.shape[0], 4)
        for i, features in enumerate(x):
            q_out[i] = torch.tensor(quantum_circuit(features, self.quantum_weights))
        
        # Post-processing with classical layer
        out = self.post_processing(q_out)
        # Apply softmax to get probabilities
        return self.softmax(out)

def load_model_and_scaler(model_path='models/qnn_model.pth', 
                           scaler_path='models/qnn_scaler.pkl'):
    """
    Load trained model and scaler
    """
    # Initialize model
    model = QNN()
    
    # Load model weights
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set to evaluation mode
    
    # Load scaler
    scaler = joblib.load(scaler_path)
    
    return model, scaler

def predict(input_data, model=None, scaler=None):
    """
    Make predictions using the trained quantum model
    
    Args:
    - input_data: numpy array of input features
    - model: Trained QNN model (optional)
    - scaler: Feature scaler (optional)
    
    Returns:
    - Predicted probabilities
    - Predicted class
    """
    # Load model and scaler if not provided
    if model is None or scaler is None:
        model, scaler = load_model_and_scaler()
    
    # Ensure input is 2D
    if input_data.ndim == 1:
        input_data = input_data.reshape(1, -1)
    
    # Scale input data
    scaled_input = scaler.transform(input_data)
    input_tensor = torch.tensor(scaled_input, dtype=torch.float32)
    
    # Make prediction
    with torch.no_grad():
        output = model(input_tensor)
    
    # Convert to numpy
    probabilities = output.numpy()
    predicted_class = np.argmax(probabilities, axis=1)
    
    return probabilities, predicted_class

def main():
    # Example usage
    # Replace with your actual input data
    new_data = np.array([
        [0.1, 0.2, 0.3, 0.4],
        [0.5, 0.6, 0.7, 0.8]
    ])
    
    # Make predictions
    probabilities, predicted_classes = predict(new_data)
    
    # Print results
    for i, (prob, pred_class) in enumerate(zip(probabilities, predicted_classes)):
        print(f"\nInput {i+1}: {new_data[i]}")
        print(f"Predicted Probabilities: {prob}")
        print(f"Predicted Class: {pred_class}")

if __name__ == "__main__":
    main()