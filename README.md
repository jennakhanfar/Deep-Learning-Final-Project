# RNA 3D Geometry Prediction from Sequence

**Course Project – Deep Learning Pipeline for RNA Folding**

This project implements deep learning models to predict 3D coordinates of RNA backbone (C1' atom) from nucleotide sequences using the Stanford RNA 3D Folding dataset.

## Project Overview

**Goal:** Predict 3D coordinates (x, y, z) of the backbone C1' atom for each nucleotide in an RNA sequence using deep learning architectures.

**Models Implemented:**
1. **Baseline CNN** - Simple 3-layer 1D convolutional neural network
2. **ResNet-1D CNN** - Residual network with skip connections, batch normalization, and dropout
3. **Transformer** - Transformer encoder with multi-head attention
4. **Graph Neural Network (GNN)** - Message-passing network for modeling RNA structure

**Key Features:**
- PCA alignment for rotation-invariant coordinate prediction
- Hybrid loss function combining coordinate MSE and pairwise distance matrix loss
- Comprehensive evaluation metrics (MSE, MAE, RMSD, correlations)

## Repository Structure

```
.
├── code.ipynb                   # Main Jupyter notebook with complete pipeline
├── README.md                    # This file - project documentation
├── requirements.txt             # Python dependencies
├── demo.py                      # Demo script with sample inputs/outputs
└── AI702_Project_guidelines_Fall2025_v2 (1).pdf  # Project guidelines
```

### File Descriptions

- **`code.ipynb`**: Main project notebook containing:
  - Data loading and preprocessing
  - Exploratory data analysis (EDA)
  - Model architectures (Baseline CNN, ResNet, Transformer, GNN)
  - Training loops with validation
  - Evaluation metrics and visualizations
  - Model comparison and analysis

- **`README.md`**: Project documentation with setup instructions, usage guide, and component descriptions

- **`requirements.txt`**: List of all Python package dependencies with version specifications

- **`demo.py`**: Standalone demo script demonstrating model inference with sample inputs and outputs

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended) or CPU
- Jupyter Notebook or JupyterLab

### Install Dependencies

```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install torch>=2.0.0 pandas>=1.5.0 numpy>=1.23.0 matplotlib>=3.6.0 scikit-learn>=1.2.0 scipy>=1.10.0 jupyter>=1.0.0
```

### PyTorch Installation (GPU Support)

For GPU acceleration, install PyTorch with CUDA support:

```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CPU only
pip install torch torchvision torchaudio
```

## Dataset Download

The project uses the **Stanford RNA 3D Folding** dataset from Kaggle.

### Option 1: Running on Kaggle (Recommended - No Download Needed)

1. Go to [Kaggle Competition: Stanford RNA 3D Folding](https://www.kaggle.com/competitions/stanford-rna-3d-folding)
2. Click "Code" → "New Notebook"
3. Add the competition dataset: "Add Input" → search "stanford-rna-3d-folding"
4. Upload or copy `code.ipynb` to your Kaggle notebook
5. Run all cells - the data path is already configured!

The notebook is pre-configured with:
- Data path: `/kaggle/input/stanford-rna-3d-folding/`
- File names: `train_sequences.csv` and `train_labels.csv`

### Option 2: Running Locally

1. **Download the dataset from Kaggle:**
   - Visit: https://www.kaggle.com/competitions/stanford-rna-3d-folding/data
   - Sign in to Kaggle (create account if needed)
   - Download the following files:
     - `train_sequences.csv`
     - `train_labels.csv`
     - `validation_sequences.csv` (optional)
     - `validation_labels.csv` (optional)

2. **Extract and organize data:**
   ```bash
   mkdir -p data/stanford_rna
   # Move downloaded CSV files to data/stanford_rna/
   ```

3. **Update the notebook:**
   - Open `code.ipynb`
   - In Cell 3, comment out the Kaggle path:
     ```python
     # DATA_DIR = "/kaggle/input/stanford-rna-3d-folding/"  # Kaggle path
     DATA_DIR = "data/stanford_rna/"  # Local path
     ```

## Usage

### Running the Complete Pipeline

1. **Open the notebook:**
   ```bash
   jupyter notebook code.ipynb
   ```
   Or use JupyterLab:
   ```bash
   jupyter lab code.ipynb
   ```

2. **Run all cells sequentially:**
   - The notebook will automatically:
     - Load and process the dataset
     - Perform exploratory data analysis
     - Train all four models (Baseline CNN, ResNet, Transformer, GNN)
     - Generate evaluation plots and metrics
     - Compare model performance

3. **Expected outputs:**
   - Training/validation loss curves
   - Model checkpoint files (`.pt` files)
   - Evaluation visualizations (coordinate plots, RMSD distributions)
   - Model comparison tables

### Running the Demo Script

For a quick demonstration with sample inputs:

```bash
python demo.py
```

This will:
- Load a pre-trained model (if available)
- Run inference on sample RNA sequences
- Display predicted 3D coordinates
- Show evaluation metrics

**Note:** The demo script requires trained model weights. Train the models first using the notebook, or download pre-trained weights if available.

## Model Architectures

### 1. Baseline CNN
- **Architecture**: 3-layer 1D CNN with embedding layer
- **Parameters**: ~206K
- **Features**: Simple convolutional layers with ReLU activation

### 2. ResNet-1D CNN
- **Architecture**: Residual blocks with skip connections
- **Parameters**: ~865K
- **Features**: 
  - Batch normalization
  - Dropout for regularization
  - Learning rate scheduling

### 3. Transformer
- **Architecture**: Transformer encoder with multi-head attention
- **Parameters**: ~3.2M
- **Features**:
  - Learnable positional embeddings
  - 6 encoder layers, 8 attention heads
  - Feed-forward dimension: 512

### 4. Graph Neural Network (GNN)
- **Architecture**: Message-passing graph layers
- **Parameters**: ~68K
- **Features**:
  - Backbone adjacency matrix (connects adjacent nucleotides)
  - Layer normalization
  - 4 graph layers

## Key Components

### Data Processing
- **Sequence Encoding**: Maps RNA sequences (A, C, G, U) to integer indices
- **Coordinate Extraction**: Converts per-residue labels to per-sequence coordinate arrays
- **PCA Alignment**: Aligns structures to canonical orientation (rotation-invariant)
- **Padding/Truncation**: Handles variable-length sequences (MAX_LEN = 256)

### Loss Functions
- **Hybrid Loss**: Combines coordinate MSE and pairwise distance matrix loss
  - Coordinate MSE: Standard error between predicted and true (x, y, z)
  - Distance Loss: Rotation-invariant loss on pairwise distances
  - Formula: `Loss = MSE_coords + 0.5 * MSE_distances`

### Evaluation Metrics
- **MSE (Mean Squared Error)**: Coordinate-wise error
- **MAE (Mean Absolute Error)**: Average absolute coordinate error
- **RMSD (Root Mean Square Deviation)**: 3D Euclidean distance error in Angstroms
- **Pearson Correlation**: Per-coordinate correlation between predicted and true values

## Training Configuration

Default hyperparameters (can be modified in the notebook):
- **Batch Size**: 32
- **Max Epochs**: 30-50
- **Learning Rate**: 1e-3 to 3e-4 (varies by model)
- **Validation Split**: 10%
- **Random Seed**: 42
- **Max Sequence Length**: 256 nucleotides
- **Early Stopping**: Patience of 20-50 epochs

## Output Files

After training, the following files are generated:
- `best_baseline_coords.pt`: Baseline CNN model weights
- `best_resnet_coords.pt`: ResNet-1D CNN model weights
- `best_transformer_coords.pt`: Transformer model weights
- `best_gnn_coords.pt`: GNN model weights

## Results

Based on validation set performance:
- **Transformer**: Best performance (~10.32 Å RMSD)
- **ResNet CNN**: ~12.53 Å RMSD
- **GNN**: ~12.82 Å RMSD
- **Baseline CNN**: ~15.41 Å RMSD

## Credits and Acknowledgments

### Dataset
- **Stanford RNA 3D Folding Dataset**: Provided by Kaggle competition
  - Source: https://www.kaggle.com/competitions/stanford-rna-3d-folding
  - Dataset contains RNA sequences and corresponding 3D coordinate structures

### Codebase
This project is built from scratch for the course assignment. The implementation includes:
- Custom model architectures (CNN, ResNet, Transformer, GNN)
- Data preprocessing and alignment techniques
- Training and evaluation pipelines

### Libraries and Frameworks
- **PyTorch**: Deep learning framework
- **NumPy**: Numerical computations
- **Pandas**: Data manipulation
- **Matplotlib**: Visualization
- **scikit-learn**: Data splitting and utilities
- **SciPy**: Scientific computing

### References
- Transformer architecture based on "Attention Is All You Need" (Vaswani et al., 2017)
- ResNet architecture inspired by "Deep Residual Learning for Image Recognition" (He et al., 2016)
- Graph Neural Networks based on message-passing frameworks

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory:**
   - Reduce `BATCH_SIZE` in the notebook
   - Reduce `MAX_LEN` (e.g., from 256 to 128)
   - Use CPU instead: `DEVICE = "cpu"`

2. **File Not Found Errors:**
   - Ensure data files are in the correct directory
   - Check `DATA_DIR` path in Cell 3 of the notebook

3. **Import Errors:**
   - Verify all dependencies are installed: `pip install -r requirements.txt`
   - Check Python version (3.8+ required)

4. **Model Loading Errors:**
   - Ensure model weights (`.pt` files) exist before loading
   - Train models first using the notebook

## License

This project is for educational purposes as part of a course assignment.

## Contact

For questions or issues, please refer to the course instructor or create an issue in the repository.
