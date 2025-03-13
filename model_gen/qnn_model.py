import pennylane as qml
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import os
import joblib

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Define quantum device
n_qubits = 4  # One qubit per input feature
dev = qml.device("default.qubit", wires=n_qubits)

# Define the quantum circuit
@qml.qnode(dev)
def quantum_circuit(inputs, weights):
    # Encode input features
    for i in range(n_qubits):
        qml.RY(inputs[i], wires=i)
    
    # Apply parameterized quantum circuit
    # First layer of rotations
    for i in range(n_qubits):
        qml.RY(weights[0, i], wires=i)
    
    # Entangling layer
    for i in range(n_qubits - 1):
        qml.CNOT(wires=[i, i + 1])
    qml.CNOT(wires=[n_qubits - 1, 0])  # Connect last qubit to first
    
    # Second layer of rotations
    for i in range(n_qubits):
        qml.RY(weights[1, i], wires=i)
    
    # Entangling layer
    for i in range(n_qubits - 1):
        qml.CNOT(wires=[i, i + 1])
    qml.CNOT(wires=[n_qubits - 1, 0])
    
    # Final rotation layer
    for i in range(n_qubits):
        qml.RY(weights[2, i], wires=i)
    
    # Return expectation values
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

# Define Quantum Neural Network class for 6 classes
class QNN(nn.Module):
    def __init__(self):
        super(QNN, self).__init__()
        self.quantum_weights = nn.Parameter(torch.randn(3, n_qubits))
        self.post_processing = nn.Linear(n_qubits, 6)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        q_out = torch.zeros(x.shape[0], n_qubits)
        for i, features in enumerate(x):
            q_out[i] = torch.tensor(quantum_circuit(features, self.quantum_weights))
        
        out = self.post_processing(q_out)
        return self.softmax(out)

# Function to load and preprocess data
def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    
    if len(data.columns) < 5:  
        raise ValueError("CSV file should have at least 5 columns: 4 input features and 1 target")
    
    X = data.iloc[:, :4].values
    y = data.iloc[:, 4:].values
    
    if y.shape[1] == 1:
        y_encoded = np.zeros((y.shape[0], 6))  # Changed from 5 to 6 classes
        for i, val in enumerate(y.flatten()):
            y_encoded[i, int(val)] = 1
        y = y_encoded
    
    # Ensure we have 6 output classes
    if y.shape[1] != 6:
        raise ValueError(f"Target should have 6 classes, but found {y.shape[1]}")
    
    # Normalize inputs to range [0, π] for quantum circuit encoding
    scaler = MinMaxScaler(feature_range=(0, np.pi))
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler

# Function to train the model
def train_model(model, X_train, y_train, epochs=100, batch_size=32, lr=0.01):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    
    # Create DataLoader
    dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    losses = []
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_x, batch_y in dataloader:
            # Forward pass
            pred = model(batch_x)
            loss = loss_fn(pred, batch_y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    return losses

# Function to evaluate the model
def evaluate_model(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
        
        predictions = model(X_test_tensor)
        
        # Calculate accuracy for classification
        pred_classes = torch.argmax(predictions, dim=1)
        true_classes = torch.argmax(y_test_tensor, dim=1)
        accuracy = (pred_classes == true_classes).float().mean()
        
        # Calculate MSE loss
        loss_fn = nn.MSELoss()
        mse = loss_fn(predictions, y_test_tensor).item()
        
    return accuracy.item(), mse, predictions.numpy()

# Main function to run the entire pipeline
def main(csv_file, test_size=0.2, epochs=100, batch_size=32, lr=0.01):
    # Load and preprocess data
    print(f"Loading data from {csv_file}...")
    X, y, scaler = load_and_preprocess_data(csv_file)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples")
    
    # Initialize model
    model = QNN()
    
    # Train model
    print("Training model...")
    losses = train_model(model, X_train, y_train, epochs=epochs, batch_size=batch_size, lr=lr)
    
    # Evaluate model
    print("Evaluating model...")
    accuracy, mse, predictions = evaluate_model(model, X_test, y_test)
    
    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Test MSE: {mse:.4f}")
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Save model and scaler
    model_path = os.path.join('models', 'qnn_model.pth')
    scaler_path = os.path.join('models', 'qnn_scaler.pkl')
    
    # Save model
    torch.save(model.state_dict(), model_path)
    print(f"Model saved as {model_path}")
    
    # Save scaler using joblib
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved as {scaler_path}")
    
    # Example of using the model for prediction
    print("\nPrediction example:")
    sample_input = X_test[0:1]  # Take first test sample
    sample_input_tensor = torch.tensor(sample_input, dtype=torch.float32)
    with torch.no_grad():
        sample_output = model(sample_input_tensor)
    
    print(f"Input (scaled): {sample_input[0]}")
    print(f"Raw input would be: {scaler.inverse_transform(sample_input)[0]}")
    print(f"Predicted probabilities: {sample_output.numpy()[0]}")
    print(f"Predicted class: {torch.argmax(sample_output, dim=1).item()}")
    
    return model, scaler

if __name__ == "__main__":
    # Replace with your CSV file path
    csv_file = "cleaned_data.csv"
    
    # Train and evaluate model
    model, scaler = main(csv_file, epochs=100)