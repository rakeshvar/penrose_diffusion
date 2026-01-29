import sys
import numpy as np
import torch
from code.models.diffuser import Diffuser  # Assumes ddim.py is in the same folder

NBAR = 85

def print_stats(sample, text):
    g_min = sample.min(axis=0)
    g_mean = sample.mean(axis=0)
    g_max = sample.max(axis=0)
    g_std  = sample.std(axis=0)
    g_var  = sample.var(axis=0)

    print("\n", text, " ", sample.shape[:-1], " tiles")
    print(f"{'Dim':<10} {'Min':<15} {'Mean':<15} {'Max':<15} {'Std Dev':<15} {'Vari':<15}")
    print("-" * NBAR)
    var_names = ['x', 'y', 'angle', 'color']
    for i, name in enumerate(var_names):
        print(f"{name:<10} {g_min[i]:<15.4f} {g_mean[i]:<15.4f} {g_max[i]:<15.4f} {g_std[i]:<15.4f} {g_var[i]:<15.4f}")

    print("="*NBAR)

def analyze_dataset(data_path):
    print(f"Loading {data_path}...")
    dataset = np.load(data_path)
    xya = dataset['xya']
    colors = dataset['colors']
    xyac = np.concatenate([xya, colors[..., None]], axis=-1)
    print(f"Dataset Shape: {xyac.shape}")

    flat_all = xyac.reshape(-1, 4)
    print_stats(flat_all, "Global")

    # ---------------------------------------------------------
    for i in range(10):
        j = np.random.randint(0, xyac.shape[0])
        print_stats(xyac[j], f"{i:<4}) Sample {j}")

    # 4. Diffusion Process Check (Variance Preserving?)
    # ---------------------------------------------------------
    diffuser = Diffuser(num_timesteps=1000)

    for i in range(5):
        print("\n" + "="*50)
        print(f"DIFFUSION NOISE CHECK (Sample {i})")
        x_start = torch.from_numpy(xyac[i:i+1]).clone() # Shape [1, N, 4]

        # Metrics to track:
        # Mean (should stay ~0)
        # Var (should stay ~1 if VP schedule works)
        # MeanSq (Sum of Squares/N, usually = Var + Mean^2)
        # Header with grouped columns
        header_x = f"{'x_mean':<8} {'x_var':<8} {'x_msq':<8}"
        header_y = f"{'y_mean':<8} {'y_var':<8} {'y_msq':<8}"
        header_a = f"{'a_mean':<8} {'a_var':<8} {'a_msq':<8}"

        print(f"{'t':<4} | {header_x} | {header_y} | {header_a}")
        print("-" * 100)

        steps = list(range(0, 1000, 100)) + [999]

        for t_val in steps:
            t_tensor = torch.tensor([t_val]).long()

            # Generate noisy sample
            noisy_sample, _ = diffuser.q_sample(x_start, t_tensor)

            # Flatten to [Batch * N, 3] to separate dimensions
            flat_noisy = noisy_sample[..., :3].reshape(-1, 3)

            # Calculate stats per column (dim=0 collapses the batch, preserving x/y/a)
            c_mean = flat_noisy.mean(dim=0)          # shape [3]
            c_var  = flat_noisy.var(dim=0)           # shape [3]
            c_msq  = (flat_noisy ** 2).mean(dim=0)   # shape [3]

            # Format the row strings
            # x stats (index 0)
            str_x = f"{c_mean[0]:<8.4f} {c_var[0]:<8.4f} {c_msq[0]:<8.4f}"
            # y stats (index 1)
            str_y = f"{c_mean[1]:<8.4f} {c_var[1]:<8.4f} {c_msq[1]:<8.4f}"
            # a stats (index 2)
            str_a = f"{c_mean[2]:<8.4f} {c_var[2]:<8.4f} {c_msq[2]:<8.4f}"

            print(f"{t_val:<4} | {str_x} | {str_y} | {str_a}")


if __name__ == "__main__":
    analyze_dataset(sys.argv[1])