# Penrose Diffusion

We probe the limits of class-conditioned structure discovery using *Permutation-Invariant DDPM/DDIM models on Sets*. We train Deep Neural Networks to learn to sample chaotic, aperiodic tilings on zero-measure fractal manifolds without repetition.

A Unified "Plug-and-Play" Architecture for the latest generative frameworks in Gen AI:
- Transformer Denoisers: self-attention to manage long-range spatial dependencies.
- Latent Space Learning: cross-attention and VAE information bottlenecks to compress complex geometric structures.
- Discrete Diffusion: on the categorical realm of tile slots.
- Auto-regressive GPTs: LLM-style generation of a sequence of tokenized tiles.

We experiment with two distinct spatial regimes:
- Periodic Hexagonal Tilings that have a predictable 6-fold symmetry and translational invariance that repeats.
- Aperiodic Penrose (P3) Rhombus Tilings, which obey a chaotic/fractal 5-fold symmetry and a "forbidden" quasi-crystalline structures that never repeat globally.

## Data Generation

As there is no ready-made dataset for this problem, we set out to make one ourself. First we need some shapes.

### Base Shapes
The guiding shapes come from silhouettes of the **MPEG-7** dataset (*Core Experiment Shape-1 Part B*).

- 70 distinct classes (e.g., apple, fish, bat, chopper, etc.)
- Each class has *20* different sihouttes
- Total 1400 unique silhouttes to base of off

![Sample GIFs of each class](reference/images/gifs_1.png)

### Tile Generation

Mimicing every silhoutte, tiles are ‘cut out’ of a mother canvas of rhombuses or hexagons.
- A **Mother Canvas** of 10-20K tiles of hexagons or P3 Rhombuses is generated via tesselation, with the correct color pattern
- **Endless** amount of shapes can be generated to mimic one silhoutte by snipping from different parts of the canvas and by rotating it at different angles
- For **Uniform Sample Complexity**, i.e. to get samples of the same size regardless of the density of the original silhoutte, be it a pencil of an elephant, we need to appropriately scale them first.


### Representation
Canonically... A single sample is represented as N polygons. Each polygon is represented as a center and an orientation. One sample is N x 4 matrix of (x, y, angle and color)
  * `(x, y)` have zero-mean and unit variance
  * `angle` $\in [-\pi, \pi]$
  * `color` $\in \{0, 1\}$

### Color Constraint
Each polygon also has a **binary** color property that has to follow a strict pattern:
- For *6-fold* symmetry

  * 2/3rds of the tiles are `Uncolored`
  * Rest are `Colored`
  * Colored tiles should *not* be touching each other
![Hexagonal Horse](reference/images/horse-07.svg)

- For *5-fold* symmetry

  * $\phi-1 = \frac{1}{\phi} = \frac{\sqrt{5}-1}{2} = 61.8$% `Fat` tiles
  * $2 - \phi = 1- \frac{1}{\phi} = \frac{3-\sqrt{5}}{2} = 38.2$% `Thin` tiles
  * Together they should obey the Penrose P3 rules
![Pentagonal Bird](reference/images/bird-15.svg)

## Architectures


The project is designed as a highly modular code base, allowing one to plug-in a wide variety of Generative AI models. The Diffusion models contain two main components, the forward and the reverse processes. In general, we have:

- **Data Augmenter**: Perturbs the data a bit by translation and rotation during training
-  Forward process adds noise to `x, y, angle` via the *Diffuser*, a DDIM/DDPM diffusion process manager (1000 timesteps)
- Reverse process denoises using the *Denoiser*: Usually a Transformer based model that predicts added noise by conditiontioning on: `class`, `time` and `colors`

Currently Supported Models include...

### 1. Direct Diffusion (`direct`)
Diffuses forward and backward directly on the tile coordinates and angles using denoising *Transformer*s — standard self-attention over the set of tiles. There are 5% global `CLS` tokens/tiles are appended to learn the latent structure.

### 2. Direct Diffusion with ISAB (`isab`)
*Induced Set Attention Block*: Diffusion is similar to the above but the denoising transformer has an information bottleneck, which enocourages latent factor learning.

### 3. Latent Diffusion (`latent`)
Encodes the geometry into a compressed latent space before diffusion. It has three main components:
- *Set Encoder*: Compresses the unordered set of tiles into a latent vector $z$ using attention pooling.
- *Latent Diffuser*: Performs DDIM diffusion in the latent space.
- *Perceiver Decoder*: Reconstructs the set of tiles from the noisy latent $z$, conditioned on tile colors using cross-attention.

### 4. Masked Discrete Diffusion (`discrete`)
Here the mother canvas from which the shapes are cut is known and fixed. Each sample is then specified as a set of $N$ indices from the mother canvas. 
- Forward diffusion masks more and more of these indices 
- Reverse process trains a Neural Network to predict the index of the masked tokens 

### 5. Auto-Regressigve Language Model (`llm`)
This does not employ diffusion. Instead it treats the data as integer indicies on a discrete grid.
- The hexagons are naturally converted into a *q, r, s* space on the hexagonal grid
- The rhombuses have a hierarchical representation based on the generating tesselation process.

Then...
- *Tokenization* converts indexing integers to tokens
- *Autoregression* generates tiles sequentially using a GPT-style decoder.


## Loss Functions

In additon to the *array* of models above, the model supports multiple training objectives leaving us with a rich *matrix* of configurations to choose from. Couple this with support for various file systems (cloud and local) and hardware accelerators (XLA for TPUs and CUDA for GPUs) we are indeed left with a *tensor* of configurations.


#### Diffusion Equations

Variance Preserving Transformation. The signal $x_0$ and noise $\epsilon$ are combined so that the variance of $x_t$ is preserved across $t$.


$$
\begin{bmatrix}
x_t \\
v
\end{bmatrix} =
\begin{bmatrix}
\sqrt{\alpha_t} & \sqrt{1-\alpha_t} \\
-\sqrt{1-\alpha_t} & \sqrt{\alpha_t}
\end{bmatrix}
\begin{bmatrix}
x_0 \\
\epsilon
\end{bmatrix}
$$

### Standard Losses

- **Noise Prediction Loss (`npl`)**: Standard DDIM, predicts added noise $\epsilon$
- **Sample Prediction Loss (`spl`)**: Directly predicts clean samples $x_0$
- **Velocity-Prediction (`vpl`)**: Predicts velocity $v$ a combination of sample $x_0$ and noise $\epsilon$, that is orthogonal to the noised data $x_t$


### Permutation-Invariant Losses

The above loss functions do not take into account the permutation invariance for sets of tiles, where the order of tiles doesn't matter. These soft assignment losses borrow ideas from `Optimal Transport` to make the model learn permutation invariance:

- **Permutation Invariant Loss (`pil`)**: Computes a soft-assignment between predicted tiles and ground truth to calculate loss. The soft-assignement matrix is row-stochastic. It is simple, but there is a risk of all tiles wanting to collapse to a single target tile.
- **Sinkhorn Loss (`shl`)**: Uses the Sinkhorn-Knopp algorithm to enforce a doubly-stochastic match (permutation matrix) between prediction and truth. This is still a soft-assignment.

- **LSA Parallel Loss (`lpl`)**: Advanced `Linear Sum Assignment (LSA)` losses handle permuation-invariance more head on using the Hungarian Algorithm to find the optimal matching $\Pi^\star$ between *noised* tiles $x_t$ and ground truth tiles $x_0$ before calculating loss. This permutation $\Pi^\star$  can be calculated in parallel on the CPU, while the `Denoiser` is prediction $\hat\epsilon$.


$$
L = \| \Pi^\star(x_o) - (x_t - \hatϵ) \|^2 
$$

where

$$
\Pi^* = \arg\min_\Pi \| \Pi(x_o) - x_t \|^2
$$


- **LSA Serial Loss (`lsl`)**: Sample is predicted and the minimum loss to any permutation of the original sample is considered.


$$
L = \min_\Pi \| \Pi(x_o) - (x_t - \hatϵ) \|^2 \\
$$

### Auxillary Losses
In addition to set level permutation invariance, there are additional constraints, like the polygons do not overlap. i.e. the tokens in a sequence of the memebers of a set do not repeat! Auxillary loss terms can enforce such **geometric constraints**.

- **Lattice Loss** The distance to the nearest neighbour should be
  - exactly $\sqrt{3}$ unit side of the hexagon, or
  - within a given range for the P3 Rhombuses.

Overlaps can be penalized heavily using Itakura-Saito loss over the distance to the nearest neighbour $d^\star$.

$$
L_{lattice} = \sum_i d^\star_i - 1 -\log d^\star_i
$$

- **Stability-Margin Loss (`margin_lambda`)**: A port of the attention-prior regularizer from [*Support Tokens, Stability Margins, and a New Foundation for Robust LLMs*](https://arxiv.org/abs/2602.22271) to the set-diffusion setting. Viewing set attention over the noisy tiles as a latent-noise generator $u_i = \mu_i(u) + \epsilon_i$, the exact log-density contains a log-Jacobian term for the residual map $e_i = u_i - \mu_i(u)$. Because attention weights depend on $u_i$ through the query, the diagonal Jacobian block is

$$
\frac{\partial e_i}{\partial u_i} = I - \Sigma_i A, \qquad A = W_K^\top W_Q / \sqrt{p},
$$

  where $\Sigma_i$ is the attention-weighted covariance of the attended tile embeddings. The penalty is the log-barrier

$$
L_{margin} = -\frac{1}{N}\sum_i \log\left|\det\left(I - \Sigma_i A\right)\right|,
$$

  which diverges as any tile's attention geometry approaches the degeneracy boundary $\det(I - \Sigma_i A) = 0$. In diffusion terms, each reverse step $x_{t-1} = \mu_\theta(x_t) + \sigma_t \epsilon$ is exactly the latent-noise generative rule analyzed in the paper, so the barrier keeps the per-step denoising map well-conditioned. The prior is a single lightweight attention stage over the denoiser's input embeddings (projected to `margin_prior_dim` dimensions) and is enabled via `margin_lambda` (paper suggests 0.02–0.05):

```bash
python train.py is128 datasets/hex.npz -m margin_lambda=0.05
```

  Tiles with the largest barrier contribution are the *support tiles* — the set elements whose local context geometry constrains stability, analogous to support vectors in SVMs.

## Output

We provide a comprehensive **SVG** engine to visualize outputs. Unlike pixel plots, our scripts generate resolution-independent `svg` files, allowing us to inspect individual tile placements, angles, and types. This keeps the view scalable and customizable.

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
 - `B` = 70 × 20 × num_copies
 - `N` = number of tiles in each sample

### 2. Train Model

```bash
python train.py [dataset.npz] [config_group] [checkpoint.pt] [options]
```

**Examples:**
```bash
# Help
python train.py -h

# Train from scratch

# Direct diffusion
python dd128 train.py datasets/hex_t096_c100_u18.npz

# ISAB
python dd128 isab train.py datasets/hex_t096_c100_u18.npz

# Latent diffusion
python ld128 train.py datasets/hex_t096_c100_u18.npz

# LLM
python dd128 llm train.py datasets/hexqr_t128_c64_u16.npz

# Override training parameters
python train.py datasets/hex_t096_c100_u18.npz \
  -t lr=0.0005 -t batch_size=32 -t num_epochs=99 \                    # training
  -m model=latent -m loss=lsaserial -m num_layers=4 -m d_model=256 \  # model
  -w enable=False -w project=test                                     # WANDB

# Resume from checkpoint (using the same dataset)
python train.py checkpoint.pt
```

**Training produces:**
- Checkpoints saved to `checkpoints/` (keeps latest 2)
- Sample SVGs saved to `samples/` each epoch
- Losses, etc. are logged to WANDB.ai

### 3. Generate Samples

Interactive sampling from trained checkpoint:

```bash
python -m scripts.sample <checkpoint.pt>
```

The sampler will prompt for class selection, and creates a series of svg images starting from random noise to a created image of that class.


## Configuration
Given the tensor of options available, we configure everything via dictionaries. Model and training settings are defined in `configs/*.yaml`. Some examples might look like:

```yaml
dd64:
  model:              # Model settings
    model: 'direct'
    num_classes: 70
    d_model: 64
    num_layers: 4
    loss: 'shl'

  train:              # Training settings
    lr: 0.001
    batch_size: 64
    num_epochs: 100
  
  wandb:              # WANDB setting
    enable: true
    project: 'penrose-train'
    run_name: null        # Auto-generated if null
    run_id: null          # Specify to resume a specific run

```

**Override options:**
- `-m key=value`: Override denoiser model config
- `-t key=value`: Override training config
- `-w key=value`: Override WANDB config

## TPU Training

For Google Cloud TPU v6e, see [`v6e_setup_guide.md`](v6e_setup_guide.md) for complete setup instructions.


## Project Structure

```text
code/
├── models/
│   ├── directdiff/       # Direct Diffusion (Transformer & ISAB)
│   ├── latentdiff/       # Latent Diffusion (Set Encoder/Decoder)
│   ├── llm/              # Autoregressive Transformer (GPT-style)
│   └── diffuser.py       # DDIM/DDPM logic
├── data/
│   ├── generator.py
│   ├── imageset.py       # MPEG-7 silhouette loading
│   └── load.py
├── hex/                  # Hexagonal tiling logic & SVG
├── pen/                  # Penrose tiling logic & SVG
├── utils/
├── config.py             # Configuration management
├── wandblog.py
├── filesystem.py         # Enables local and cloud storage and retrieval
└── compatibility.py      # TPU/GPU/CPU compatibility layer

configs.yaml              # Hyperparameters & Experiment groups

train.py                  # Main training entry point

scripts/
├── sample.py             # Interactive sampling
├── create_dataset.py     # Create training .npz from shapes
└── passthru.py           # Model pass-through for sanity check
```

# Details

### 1. Dynamic Data Generation

Unlike traditional approaches that cache a fixed dataset, our generator creates unique samples **dynamically**. **Dual Rotation** for every training step, we apply random rotations to both the underlying tile canvas and the target silhouette mask independently. This acts as **Regularization** as the model never sees the exact same arrangement of tiles twice, preventing memorization and instead forcing geometric generalization.

The Process:

- **Load shape masks** from MPEG7 dataset (70 classes)
- **Generate tile canvas** (of 5/6-fold symmetry) covering the target area at appropriate density
- **Dynamic scaling**: Scale each silhouette to contain exactly `N` tiles regardless of shape complexity
- **Dual rotation**: Independently rotate both the tile canvas and the shape mask
- **Random translation**: Position the mask randomly over the tile grid
- **Sample tiles**: Select tiles that overlap with the shape mask (**complicated coverage-based sampling**)
- **Center** each sample around the origin
- **Store as tensors**: (x, y, angle, color) with consistent shape `(..., N, 4)`

### 2. Uniform Sample Complexity

A major challenge in geometric modeling is varying density amongst silhouettes. Some are dense some are light (e.g., a thin pencil vs. a bulky elephant). We solve this with a **dynamic scaling algorithm**, which ensures we have exactly `N` tiles for every data sample:

- We scale the silhoutte up by the proper value inversely proportional to its density. The sparser the silhoutte, the more we have to scale it up, and viceversa
- Now we collect all the pentagons that have all the four ‘pseudo’ vertices within mask
- That is usually not enough then we add more pentagons that have 3/4 ‘pseudo’ vertices within mask
- If that is not enough we add those with 2/4. These are exactly on the border
- Rarely do we have those with only 1/4 within the mask


This is well-vectorized, but we do it only once on the CPU apriori and save it to an `npz` file, as it could be a speed bottle neck to do this coverage algorithm for every single data sample.

### 3. Training

- **Apply geometric augmentation** each epoch add additional rotation + translation to $(x, y, θ)$
- **Convert angles**, to get unit variance on the orientation dimension, depending on the model, we either
  - Convert θ → (sin θ, cos θ)
  - Scale θ →  $\frac{\sqrt{3}}{\pi}$ θ
- **Forward diffusion**: Gradually add Gaussian noise to the sample. In diffusion lingo uncorrupted data is represented as $x_0$, which in our case is the `N x 3` matrix of $(x, y, θ)$. And $x_t$ is the forward diffused value with noise variance, $var(ϵ)$ is monotonic in $t$. The variance preserving transform is:

$$x_t = \sqrtᾱ_t ~ x_0 + \sqrt{1-ᾱ_t} ~ ε $$

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

### 4. Hardware & Scaling
- **Multi-Backend**: Seamless switching. The codebase automatically detects and adapts to available hardware:
  - **TPU (v4/v5e/v6e)**: `torch_xla` with PJRT runtime for distributed training
  - **GPU**: Standard PyTorch with CUDA acceleration
  - **CPU**: Fallback mode for development and testing

- **Google Cloud Storage Support**: Optimized for Google Cloud TPUs with `torch_xla` and distributed sampling and storage.

- **WandB Integration**: Automatic logging of loss curves, gradients, and **SVG samples** directly to the dashboard.



## Requirements
 `torch` `numpy` `scipy`
**Optional:** `torch-xla` (for TPU) `torch-linear-assignment` (on CPU/GPU)
