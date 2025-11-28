"""
Demo script for RNA 3D Geometry Prediction

This script demonstrates model inference with sample RNA sequences.
It shows how to:
1. Load a trained model
2. Encode RNA sequences
3. Predict 3D coordinates
4. Evaluate predictions

Sample Input: RNA sequences (strings of A, C, G, U)
Sample Output: 3D coordinates (x, y, z) for each nucleotide
"""

import os
import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple

# Model definitions (same as in notebook)
VOCAB = ["A", "C", "G", "U"]
stoi = {c: i for i, c in enumerate(VOCAB)}
PAD_IDX = 0
UNK_IDX = len(VOCAB)
VOCAB_SIZE = len(VOCAB) + 2
MAX_LEN = 256

def encode_sequence(seq: str, max_len: int = MAX_LEN) -> np.ndarray:
    """Encode RNA sequence to integer array"""
    arr = np.zeros(max_len, dtype=np.int64)
    for i, ch in enumerate(seq[:max_len]):
        idx = stoi.get(ch.upper(), UNK_IDX)
        arr[i] = idx + 1
    return arr

# Model architectures (simplified versions)
class BaselineCNN(nn.Module):
    def __init__(self, vocab_size, emb_dim=64, hidden_dim=128, n_targets=3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.cnn = nn.Sequential(
            nn.Conv1d(emb_dim, hidden_dim, 5, padding=2), nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, 5, padding=2), nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, 5, padding=2), nn.ReLU(),
        )
        self.head = nn.Conv1d(hidden_dim, n_targets, 1)

    def forward(self, x):
        emb = self.embedding(x).transpose(1, 2)
        feat = self.cnn(emb)
        out = self.head(feat).transpose(1, 2)
        return out

def load_model(model_path: str, device: str = "cpu"):
    """Load a trained model from checkpoint"""
    model = BaselineCNN(VOCAB_SIZE).to(device)
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        print(f"✓ Loaded model from {model_path}")
        return model
    else:
        print(f"⚠ Model file {model_path} not found. Using untrained model.")
        return model

def predict_coordinates(model: nn.Module, sequences: List[str], device: str = "cpu") -> np.ndarray:
    """
    Predict 3D coordinates for RNA sequences
    
    Args:
        model: Trained PyTorch model
        sequences: List of RNA sequence strings (e.g., ["ACGU", "GGGAAACCC"])
        device: Device to run inference on
    
    Returns:
        numpy array of shape (batch_size, max_len, 3) with predicted coordinates
    """
    model.eval()
    batch_encoded = []
    
    for seq in sequences:
        encoded = encode_sequence(seq)
        batch_encoded.append(encoded)
    
    batch_tensor = torch.from_numpy(np.array(batch_encoded)).long().to(device)
    
    with torch.no_grad():
        predictions = model(batch_tensor)
    
    return predictions.cpu().numpy()

def compute_rmsd(pred_coords: np.ndarray, true_coords: np.ndarray, seq_lengths: List[int]) -> List[float]:
    """
    Compute RMSD (Root Mean Square Deviation) for each sequence
    
    Args:
        pred_coords: Predicted coordinates (batch_size, max_len, 3)
        true_coords: True coordinates (batch_size, max_len, 3)
        seq_lengths: Actual length of each sequence
    
    Returns:
        List of RMSD values in Angstroms
    """
    rmsds = []
    for i, length in enumerate(seq_lengths):
        pred = pred_coords[i, :length, :]
        true = true_coords[i, :length, :]
        diff_sq = np.sum((pred - true) ** 2, axis=1)
        mse = np.mean(diff_sq)
        rmsd = np.sqrt(mse)
        rmsds.append(rmsd)
    return rmsds

def print_prediction_summary(sequences: List[str], predictions: np.ndarray, seq_lengths: List[int]):
    """Print formatted prediction summary"""
    print("\n" + "="*70)
    print("PREDICTION SUMMARY")
    print("="*70)
    
    for i, (seq, length) in enumerate(zip(sequences, seq_lengths)):
        print(f"\nSequence {i+1}: {seq}")
        print(f"Length: {length} nucleotides")
        print(f"Predicted coordinates (first 5 nucleotides):")
        print(f"{'Nucleotide':<12} {'X (Å)':<12} {'Y (Å)':<12} {'Z (Å)':<12}")
        print("-" * 50)
        
        for j in range(min(5, length)):
            nt = seq[j]
            x, y, z = predictions[i, j, :]
            print(f"{nt:<12} {x:>11.2f} {y:>11.2f} {z:>11.2f}")
        
        if length > 5:
            print(f"... ({length - 5} more nucleotides)")
    
    print("\n" + "="*70)

def main():
    """Main demo function"""
    print("RNA 3D Geometry Prediction - Demo Script")
    print("="*70)
    
    # Check for GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")
    
    # Sample RNA sequences
    sample_sequences = [
        "GGGUGCUCAGUACGAGAGGAACCGCACCC",  # Example from dataset
        "ACGUACGUACGU",                    # Simple repeating sequence
        "GGGAAACCC",                        # Short hairpin-like sequence
    ]
    
    print("Sample Input Sequences:")
    for i, seq in enumerate(sample_sequences, 1):
        print(f"  {i}. {seq} (length: {len(seq)})")
    
    # Try to load trained model
    model_path = "best_baseline_coords.pt"
    model = load_model(model_path, device)
    
    # Make predictions
    print(f"\n{'='*70}")
    print("Running Inference...")
    print(f"{'='*70}")
    
    predictions = predict_coordinates(model, sample_sequences, device)
    seq_lengths = [len(seq) for seq in sample_sequences]
    
    # Print results
    print_prediction_summary(sample_sequences, predictions, seq_lengths)
    
    # Statistics
    print("\n" + "="*70)
    print("STATISTICS")
    print("="*70)
    
    for i, (seq, length) in enumerate(zip(sample_sequences, seq_lengths)):
        coords = predictions[i, :length, :]
        print(f"\nSequence {i+1} ({seq[:20]}...):")
        print(f"  Coordinate ranges:")
        print(f"    X: [{coords[:, 0].min():.2f}, {coords[:, 0].max():.2f}] Å")
        print(f"    Y: [{coords[:, 1].min():.2f}, {coords[:, 1].max():.2f}] Å")
        print(f"    Z: [{coords[:, 2].min():.2f}, {coords[:, 2].max():.2f}] Å")
        print(f"  Mean coordinate: ({coords[:, 0].mean():.2f}, {coords[:, 1].mean():.2f}, {coords[:, 2].mean():.2f}) Å")
    
    print("\n" + "="*70)
    print("Demo completed!")
    print("="*70)
    print("\nNote: For accurate predictions, train the model first using code.ipynb")
    print("      The model weights will be saved as .pt files after training.")

if __name__ == "__main__":
    main()

