# Penrose Diffusion

**Learning to generate aperiodic Penrose tilings (and hexagonal grids) using class-conditional diffusion models.**

Penrose Diffusion is a geometric **Denoising Diffusion Probabilistic Model (DDPM/DDIM)** that generates aperiodic tilings (Penrose P3) (and periodic grids (Hexagonal)) constrained by semantic class labels.


**Symmetries:**
- **5-fold Penrose P3 tilings** - Aperiodic rhombus patterns (Thin/Fat rhombuses)
- **6-fold hexagonal tilings** - Regular tessellations (Dark/Lite rhombuses)

**Training Dataset:**
The model is trained on a manufacturered dataset of tile arrangements, algorithmically designed to mimic the silhouettes from **MPEG-7 Core Experiment Shape-1 Part B** 
- 70 distinct classes (e.g., apple, fish, bat, chopper, etc.).
- For Penrose tilings the arrangement of Fatt and Thin tiles must obey Penrose rules
-- 1.612:1 ratio of Thin to Fatt
- For hexagonal mesh, no two colored tiles must be touching 
-- 2:1 ratio of uncolored to colored

### Key Features

#### 1. Infinite Data Generation

We do **not** train on a fixed dataset of cached tessellatios. Instead, we implement an **infinite data loader** that generates unique samples on the fly.

- **Dual Rotation**: For every training step, we apply random rotations to both the underlying tile canvas and the target silhouette mask independently.

- **Result**: The model never sees the exact same arrangement of tiles twice, preventing memorization and encouraging robust geometric generalization.

#### 2. Uniform Sample Complexity

A major challenge in geometric modeling is varying density amongst silhouettes. Some are dense some are light. We solve this with a **dynamic scaling algorithm**:

- Regardless of the class (e.g., a thin pencil vs. a bulky elephant), we automatically scale the silhouette so that it is filled by a **fixed, target-number of tiles** (e.g., 768 tiles).
- This ensures consistent tensor shapes `(Batch, Num_Tiles, 4)` and stable training dynamics across all classes.
- 4 ≈ (x, y, orientation, color)

#### 3. Advanced Permutation-Invariant Losses

Standard loss functions (like MSE) fail for sets of tiles because the order of tiles doesn't matter (permutation invariance). We implement advanced losses to handle this:

- **Linear Sum Assignment (LSA)**: We use the Hungarian Algorithm (solved via scipy or custom CUDA kernels) to find the optimal matching between predicted tiles and ground truth tiles before calculating loss.
- **Geometric Constraints**: Custom loss terms enforce valid unit-circle properties for orientation parameters (sin θ, cos θ).

#### 4. Vector Graphics Engine

We provide a **complex, customizable SVG engine** to visualize outputs. Unlike pixel plots, our scripts generate resolution-independent `.svg` files, allowing you to inspect individual tile placements, angles, and types (Thin/Fat rhombuses).

## Architecture

### Model Components

- **Denoiser**: Transformer encoder that predicts noise or samples
- **Diffuser**: DDIM/DDPM diffusion process manager (1000 timesteps)
- **Representation**: Tiles as (x, y, sin θ, cos θ) with color labels
- **Conditioning**: Class embeddings + time embeddings + color embeddings

### Hardware Compatibility

The codebase automatically detects and adapts to available hardware:

- **TPU (v4/v5e/v6e)**: Uses `torch_xla` with PJRT runtime for distributed training
- **GPU (CUDA)**: Standard PyTorch with CUDA acceleration
- **CPU**: Fallback mode for development and testing

### Loss Functions

The model supports multiple training objectives (selected via config):

1. **Noise Prediction Loss (NPL)**: Standard DDIM, predicts added noise
2. **Sample Prediction Loss (SPL)**: Directly predicts clean samples
3. **Sample + Angle Loss (SAL)**: SPL with circle regularization
4. **LSA Loss (Serial/Parallel)**: Linear Sum Assignment for permutation-invariant learning

### Vector Graphics Output

All samples are exported as **resolution-independent SVG files**:

- **Inspect individual tiles**: Each tile is a separate SVG path element
- **Customizable styling**: Colors, stroke width, margins configurable
- **Type visualization**: Distinguish Thin/Fatt rhombuses or Colored and uncolored hexagons
- **Scalable**: View at any resolution without quality loss
- **Analysis-friendly**: Easy to measure tile positions, angles, and coverage

## Installation

```bash
# Clone repository
git clone https://github.com/rakeshvar/penrose_diffusion
cd penrose_diffusion

# Install dependencies
pip install torch numpy tqdm scipy

# For TPU support (optional)
pip install torch-xla[tpu] -f https://storage.googleapis.com/libtpu-releases/index.html
```

## Usage

### 1. Create Dataset

Generate training data from MPEG7 shape masks:

```bash
python create_dataset.py <symmetry> <num_tiles> <num_copies> [unit_side]
```

**Examples:**
```bash
# Hexagonal tilings with 96 tiles
python create_dataset.py 6 96 100 0.18

# Penrose tilings with 512 tiles  
python create_dataset.py 5 512 50 0.1
```

This creates an `.npz` file in `datasets/` containing:
- `xya`: Tile positions and orientation angles. (B, N, 3)
- `colors`: Binary colors (0 or 1). (B, N)
- `labels`: Shape class IDs (0-69). (B,)
- Metadata: symmetry, side length, class lookup table 

### 2. Train Model

```bash
python train.py <dataset.npz> [config_group] [checkpoint.pt] [options]
```

**Examples:**
```bash
# Train from scratch with default config
python train.py datasets/hex_t096_c100_u18.npz

# Resume from checkpoint
python train.py checkpoint.pt

# Use small model config
python train.py datasets/hex_t096_c100_u18.npz small

# Override training parameters
python train.py datasets/hex_t096_c100_u18.npz -t lr=0.0005 -t batch_size=32 -t num_epochs=99
```

**Training produces:**
- Checkpoints saved to `checkpoints/` (keeps latest 2)
- Sample SVGs saved to `samples/` each epoch

### 3. Generate Samples

Interactive sampling from trained checkpoint:

```bash
python sample.py <checkpoint.pt>
```

The sampler will prompt for class selection.


## Configuration

Model and training settings are defined in `configs.yaml`:

```yaml
# Model architectures
default_denoiser:
  num_classes: 70
  d_model: 64
  num_heads: 4
  num_layers: 4
  dropout: 0.1

# Training settings
default_train:
  lr: 0.001
  batch_size: 64
  num_epochs: 100
  loss: 'Noise'
```

**Config groups:** `default`, `toy`, `small`, `large`

**Override options:**
- `-t key=value`: Override training config
- `-d key=value`: Override denoiser config

**Loss options:** `noise`, `sample`, `sampleangle`, `lsaserial`, `lsaparallel`

## TPU Training

For Google Cloud TPU v6e, see [`v6e_setup_guide.md`](v6e_setup_guide.md) for complete setup instructions.


## Project Structure

```
code/
├── model/
│   ├── ddim.py           # DDIM diffusion + Transformer denoiser
│   ├── losses.py         # Loss function implementations
│   ├── augment.py        # Geometric data augmentation
│   └── sampler.py        # Sampling and SVG generation
├── data/
│   ├── generator.py      # Dataset generation from images
│   ├── imageset.py       # MPEG7 image loading
│   ├── load.py          # PyTorch dataset wrapper
│   └── create.py        # Dataset creation utilities
├── hex/
│   ├── base.py          # Hexagonal tiling logic
│   └── svg.py           # Hexagon SVG rendering
├── pen/
│   ├── base.py          # Penrose tiling logic
│   └── svg.py           # Penrose SVG rendering
├── config.py            # Configuration management
├── utils.py             # Utility functions
└── compatibility.py     # TPU/GPU/CPU compatibility layer

train.py                 # Main training script
sample.py               # Interactive sampling
create_dataset.py       # Dataset creation
configs.yaml            # Model configurations
```

## How It Works

### 1. Infinite Data Generation

Unlike traditional approaches that cache a fixed dataset, our generator creates unique samples dynamically:

- **Load shape masks** from MPEG7 dataset (70 classes)
- **Generate tile grids** covering the target area at appropriate density
- **Dynamic scaling**: Scale each silhouette to contain exactly `num_tiles` tiles regardless of shape complexity
- **Dual rotation**: Independently rotate both the tile canvas and the shape mask
- **Random translation**: Position the mask randomly over the tile grid
- **Sample tiles**: Select tiles that overlap with the shape mask (coverage-based sampling)
- **Store as tensors**: (x, y, angle, color) tuples with consistent shape `(Batch, Num_Tiles, 4)`

**Result**: Every training iteration sees a completely unique arrangement, preventing memorization.

### 2. Training

- **Convert angles** to (sin θ, cos θ) for continuity on the unit circle
- **Apply geometric augmentation** each epoch (additional rotation + translation)
- **Forward diffusion**: Gradually add Gaussian noise: x_t = √ᾱ_t x_0 + √(1-ᾱ_t) ε
- **Train transformer** to predict noise or clean samples
- **Loss computation**:
  - **Standard losses** (NPL, SPL): Direct MSE on predicted vs. ground truth
  - **LSA losses**: Solve optimal tile matching via Hungarian algorithm before computing MSE
  - **Geometric losses**: Additional circle constraints for angle representations

### 3. Sampling

- Start from pure Gaussian noise
- Iteratively denoise using DDIM (50 steps)
- Condition on shape class embedding
- Normalize angles to unit circle at final step
- Export as SVG

## Technical Details

**Tile Representation:**
- Position: (x, y) coordinates
- Orientation: (sin θ, cos θ) for continuity on circle
- Color: Binary label (0 or 1) for 2-coloring
- Total: 4D representation per tile

**Diffusion Process:**
- Forward: x_t = √ᾱ_t x_0 + √(1-ᾱ_t) ε
- Reverse: DDIM with configurable η (deterministic when η=0)
- Timesteps: 1000 (training), 50 (sampling)

**Linear Sum Assignment (LSA):**
- **Problem**: Tiles form an unordered set - standard MSE fails because it's order-dependent
- **Solution**: Use the **Hungarian Algorithm** to find optimal tile-to-tile matching
- **Implementation**: 
  - Compute pairwise distance matrix between predicted and ground truth tiles
  - Solve assignment problem using `scipy.optimize.linear_sum_assignment` or custom CUDA kernels
  - Apply MSE loss only on optimally matched pairs
- **Constraint**: Separate matching per color to respect the 2-coloring constraint
- **Result**: Permutation-invariant loss that learns tile arrangements regardless of ordering

## Requirements

- Python 3.8+
- PyTorch 2.0+
- NumPy
- PyYAML
- Pillow
- tqdm
- scipy

**Optional:**
- `torch-xla` for TPU support
- `torch-linear-assignment` for faster LSA on GPU with CUDA

## Citation

```bibtex
@software{penrose_diffusion,
  author = {Rakeshvara},
  title = {Penrose Diffusion: Conditional Tiling Generation with Diffusion Models},
  year = {2025},
  url = {https://github.com/rakeshvar/penrose_diffusion}
}
```

## License

MIT License