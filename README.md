# Penrose Diffusion

This is a **Denoising Diffusion Probabilistic Model (DDPM/DDIM)** that learns to generate tilings to form different shapes.
Tiling are of two types:

- Periodic Hexagonal tilings with *6-fold symmetry*
- Aperiodic Penrose P3 Rhombus tilings with *5-fold symmetry*

## Data
### Base Data
The guiding shapes come from silhouettes of the **MPEG-7** dataset (*Core Experiment Shape-1 Part B*).

- 70 distinct classes (e.g., apple, fish, bat, chopper, etc.)
- Each class has *20* different sihouttes
- Total 1400 unique silhouttes to base of off

![Sample GIFs of each class](reference/images/gifs_1.png)

### Data Generation

Mimicing every silhoutte, tiles are ‘cut out’ of a mother canvas of rhombuses or hexagons.
- **Endless** amount of tessellations can be generated to mimic one silhoutte by snipping from different parts of the canvas and by rotating it at different angles

- **Dual Rotation** for every training step, we apply random rotations to both the underlying tile canvas and the target silhouette mask independently.

- This acts as **Regularization** as the model never sees the exact same arrangement of tiles twice, preventing memorization and instead forcing geometric generalization.

### Uniform Sample Complexity

A major challenge in geometric modeling is varying density amongst silhouettes. 

- Some are dense some are light. We solve this with a **dynamic scaling algorithm**:
- Regardless of the class (e.g., a thin pencil vs. a bulky elephant), we automatically scale the silhouette so that it is filled by a **fixed, target-number of tiles** (e.g., 768 tiles).
- This ensures we have the exact same dimension for every data sample.

### Representation
There are many ways to represent the sets of polygons — be it hexagons with six-way symmetry or rhombuses with five-way symmetry. Canonically...
- A single sample is represented as N (say 768) polygons.
- Each polygon is represented as a center and an orientation
- One sample is N x 4 matrix of (x, y, angle and color)

  * `(x, y)` have zero-mean and unit variance
  * `angle` $\in [-\pi, \pi]$
  * `color` $\in \{0, 1\}$

#### Color Constraint
Each pentagon also has a **binary** color property
- For *6-fold* symmetry

  * `Uncolored` tiles are 0 (2/3)
  * `Colored` tiles are 1 (1/3)
  * Colored tiles should not be touching each other, in our problem
![Hexagonal Horse](reference/images/horse-07.svg)

- For *5-fold* symmetry

  * `Fat` tiles are 0 (61.8%)
  * `Thin` tiles are 1 (38.2%)
  * Together they should obey the Penrose P3 rules
![Pentagonal Bird](reference/images/bird-15.svg)

## Architectures

The project is designed in a highly modular code base, allowing one to plug-in a wide variety of Generative AI models. Currently Supported Models include...

### 1. Direct Diffusion (`direct`)
Operates directly on the tile coordinates `(x, y)` and angles using **Transformer**s. Standard self-attention over the set of tiles.

### 2. Direct Diffusion with ISAB (`isab`)
**ISAB (Induced Set Attention Block)**: Similar to the above but the transformer has an information bottleneck, which enocourages latent factor learning. Also reduces complexity from $O(N^2)$ to $O(NM)$ using inducing points for better scaling.

### 3. Latent Diffusion (`latent`)
Encodes the geometry into a compressed latent space before diffusion.
- **Set Encoder**: Compresses the unordered set of tiles into a latent vector $z$ using attention pooling.
- **Latent Diffuser**: Performs DDIM diffusion in the latent space.
- **Perceiver Decoder**: Reconstructs the set of tiles from the noisy latent $z$, conditioned on tile colors.

### 4. Language Model (`llm`)
Treats the tiling as a sequence of discrete tokens.
- **Quantization**: Converts continuous `(x, y)` coordinates into discrete integer grid coordinates.
- **Autoregressive**: Generates tiles sequentially (next-token prediction) using a GPT-style decoder.
- **Discretization**:
  - The hexagons are naturally converted into a `q, r, s` space on the hexagonal grid
  - The rhombuses have a hierarchical representation based on the generating tesselation process.

### Model Components
The Diffusion models contain two basic working horses: (plus additional components based on the specific architecture.)
- **Denoiser**: Transformer based encoder that predicts noise or samples by conditiontioning on:
  - Class
  - Time
  - Colors of tiles
- **Diffuser**: DDIM/DDPM diffusion process manager (1000 timesteps)
  - Forward process adds noise to `x, y, angle`
  - Reverse process denoises using the `Denoiser`
- **Augmenter**: Perturbs the data a bit by translation and rotation during training


### Hardware Compatibility

The codebase automatically detects and adapts to available hardware:

- **TPU (v4/v5e/v6e)**: `torch_xla` with PJRT runtime for distributed training
- **GPU**: Standard PyTorch with CUDA acceleration
- **CPU**: Fallback mode for development and testing

## Loss Functions

The model supports multiple training objectives:

### Standard Losses

- **Noise Prediction Loss (`npl`)**: Standard DDIM, predicts added noise
- **Sample Prediction Loss (`spl`)**: Directly predicts clean samples
- **Velocity-Prediction (`vpl`)**: Predicts velocity $v$ (useful for distillation).

### Permutation-Invariant Losses

The above loss functions do not take into account the permutation invariance for sets of tiles, where the order of tiles doesn't matter. These soft assignment losses borrow ideas from `Optimal Transport` to make the model learn permutation invariance:

- **Permutation Invariant Loss (`pil`)**: Computes a soft-assignment between predicted tiles and ground truth to calculate loss.
- **Sinkhorn Loss (`shl`)**: Uses the Sinkhorn-Knopp algorithm to enforce a doubly-stochastic match (permutation matrix) between prediction and truth.


Advanced `Linear Sum Assignment (LSA)` losses handle this. We use the Hungarian Algorithm to find the optimal matching between predicted tiles and ground truth tiles before calculating loss.

- **LSA Serial Loss (LSL)**: Sample is predicted and the minimum loss to any permutation of the original sample is considered.
- **LSA Parallel Loss (LPL)**: Sample is recovered from predicted Noise. Loss is the distance of this recovered sample to a permutation of data that is closed to the original data sample itself. *Note: This permutation of the orignial data can be calculated in parallel on the CPU, while the `Denoiser` is running, at no additional cost.*

### Auxillary Losses
**Geometric Constraints**: Auxillary loss terms can enforce:
-  valid unit-circle properties for orientation parameters (sin θ, cos θ) 
- valid distance between the tiles

## Output

We provide a comprehensive **SVG** engine to visualize outputs. Unlike pixel plots, our scripts generate resolution-independent `svg` files, allowing you to inspect individual tile placements, angles, and types. This keeps the view scalable and customizable.

## Installation

```bash
# Clone repository
git clone https://github.com/rakeshvar/penrose_diffusion
cd penrose_diffusion

# Install dependencies
pip install torch numpy tqdm scipy wandb

# For TPU support (optional)
pip install torch-xla[tpu] -f https://storage.googleapis.com/libtpu-releases/index.html
```

## Usage

### 1. Create Dataset [Optional]

Generate training data from MPEG7 shape masks:

```bash
python -m scripts.create_dataset <symmetry> <num_tiles> <num_copies> [unit_side]
```

**Examples:**
```bash
# Hexagonal tilings with 96 tiles, 100 copies per silhoutte
python -m scripts.create_dataset 6 96 100 0.18

# Penrose tilings with 512 tiles, 50 copies per silhoutte
python -m scripts.create_dataset 5 512 50 0.1
```

This creates an `.npz` file in `datasets/` containing:
- `xya`: Tile positions and orientation angles. (B, N, 3)
- `colors`: Binary colors (0 or 1). (B, N)
- `labels`: Shape class IDs (0-69). (B,)
- Metadata
  - `symmetry`: 5=Penrose, 6=Hexagons
  - `side_length`: of each rhombus or hexagon (defaults to a value that leads to unit variance for `x`, `y`)
  - `class_lookup_table`

where:
 - `B` = 70 * 20 * num_copies
 - `N` = number of tiles in each sample

### 2. Train Model

```bash
python train.py [dataset.npz] [config_group] [checkpoint.pt] [options]
```

**Examples:**
```bash
# Train from scratch with default config (direct)
python dd128 train.py datasets/hex_t096_c100_u18.npz

# Resume from checkpoint (using the same dataset)
python train.py checkpoint.pt

# ISAB
python dd128 isab train.py datasets/hex_t096_c100_u18.npz

# LLM
python dd128 llm train.py datasets/hexqr_t128_c64_u16.npz

# latent diffusion
python ld128 train.py datasets/hex_t096_c100_u18.npz 

# Override training parameters
python train.py datasets/hex_t096_c100_u18.npz -t lr=0.0005 -t batch_size=32 -t num_epochs=99 -t loss=lsaserial -d num_layers=4 -w enable=False
```

**Training produces:**
- Checkpoints saved to `checkpoints/` (keeps latest 2)
- Sample SVGs saved to `samples/` each epoch

### 3. Generate Samples

Interactive sampling from trained checkpoint:

```bash
python -m scripts.sample <checkpoint.pt>
```

The sampler will prompt for class selection, and creates a series of svg images starting from random noise to a created image of that class.


## Configuration

Model and training settings are defined in `configs.yaml`:

```yaml
# Denoiser settings
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

**Config groups:** `default`, `toy`, `small`, `large`, etc.

**Override options:**
- `-d key=value`: Override denoiser config
- `-t key=value`: Override training config

**Loss options:** `noise`, `sample`, `sampleangle`, `lsaserial`, `lsaparallel`

## TPU Training

For Google Cloud TPU v6e, see [`v6e_setup_guide.md`](v6e_setup_guide.md) for complete setup instructions.


## Project Structure
## Project Structure

```text
code/
├── models/
│   ├── directdiff/       # Direct Diffusion (Transformer & ISAB)
│   ├── latentdiff/       # Latent Diffusion (Set Encoder/Decoder)
│   ├── llm/              # Autoregressive Transformer (GPT-style)
│   ├── diffuser.py       # DDIM/DDPM logic
│   └── sinusoidal.py     # Time embeddings
├── data/
│   ├── generator.py      # Tiling generation logic (Hex/Penrose)
│   ├── imageset.py       # MPEG-7 silhouette loading
│   └── load.py           # Dataset loaders (.npz)
├── hex/                  # Hexagonal tiling logic & SVG
├── pen/                  # Penrose tiling logic & SVG
├── utils/
│   ├── lossy.py          # Geometric & set-matching losses
│   ├── qrs.py            # Coordinate systems (QRS <-> XY)
│   └── registry.py       # Model/Loss registry
├── config.py             # Configuration management
├── wandblog.py           # WandB logging wrapper
└── compatibility.py      # TPU/GPU/CPU compatibility layer
scripts/
├── sample.py             # Interactive sampling
├── create_dataset.py     # Create training .npz from shapes
└── passthru.py           # Model pass-through testing
train.py                  # Main training entry point
configs.yaml              # Hyperparameters & Experiment groups
```

## How It Works (Details)

### 1. Data Generation

Unlike traditional approaches that cache a fixed dataset, our generator creates unique samples dynamically:

- **Load shape masks** from MPEG7 dataset (70 classes)
- **Generate tile canvas** (of 5/6-fold symmetry) covering the target area at appropriate density
- **Dynamic scaling**: Scale each silhouette to contain exactly `num_tiles` tiles regardless of shape complexity
- **Dual rotation**: Independently rotate both the tile canvas and the shape mask
- **Random translation**: Position the mask randomly over the tile grid
- **Sample tiles**: Select tiles that overlap with the shape mask (**complicated coverage-based sampling**)
- Center each sample around the origin
- **Store as tensors**: (x, y, angle, color) tuples with consistent shape `(B, N, 4)`

#### Exact Coverage
We want each sample to have exactly `N` hexagons:
- We scale the silhoutte up by the proper value inversely proportional to its density.
  - The sparser the silhoutte, the more we have to scale it up
  - and viceversa
- Now we collect all the pentagons that have all the four ‘pseudo’ vertices within mask
- That is usually not enough then we add more pentagons that have three ‘pseudo’ vertices within mask
- If that is not enough — two. These are exactly on the border
- Rarely do we have those with only one within the mask


This is well-vectorized, but we do it only once on the CPU apriori and save it to an `npz` file, as it could be a speed bottle neck to do this coverage algorithm for every single data sample.

### 2. Training

- **Convert angles** to (sin θ, cos θ) for continuity on the unit circle
- **Apply geometric augmentation** each epoch (additional rotation + translation)
- **Forward diffusion**: Gradually add Gaussian noise ($x$ represents the `N x 4` matrix of $(x, y, sin θ, cos θ)$): $$x_t = \sqrtᾱ_t ~ x_0 + \sqrt(1-ᾱ_t) ~ ε $$
- **Train transformer** to predict noise $ \hat ε $ or clean samples $\hat x_0 $
  - Conditioned on `t`, the array of `colors` and the `class label`
- **Loss computation**: as above
- Timesteps: 1000 (training)

### 3. Sampling

- Start from pure Gaussian noise
- Iteratively denoise using DDIM/DDPM (50 steps)
   - Using the Transformer Denoiser
   - Condition on time, class shape and colors
   - DDPM with configurable η (DDIM when η=0)
- Normalize angles to unit circle at final step
- Export as SVG

### Hardware & Scaling
- **Multi-Backend**: Seamless switching between CPU, GPU (CUDA), and TPU (PJRT).
- **TPU v4/v5e/v6e Support**: Optimized for Google Cloud TPUs with `torch_xla` and distributed sampling.
- **WandB Integration**: Automatic logging of loss curves, gradients, and **SVG samples** directly to the dashboard.

## Requirements

- Python 3.8+
- PyTorch 2.0+
- numpy
- tqdm
- scipy

**Optional:**
- `torch-xla` for TPU support
- `torch-linear-assignment` for faster LSA on GPU with CUDA
