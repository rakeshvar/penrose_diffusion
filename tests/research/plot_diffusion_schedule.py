import numpy as np
import matplotlib.pyplot as plt

def plot_diffusion_schedule(num_timesteps=1000):
    # ---------------------------------------------------------
    # 1. Calculate Buffers (Exact logic from your DDIMDiffusion class)
    # ---------------------------------------------------------

    # Base schedule
    betas = np.linspace(1e-4, 0.02, num_timesteps)
    alphas = 1. - betas
    alphas_cumprod = np.cumprod(alphas)
    one_minus_alphas_cumprod = 1. - alphas_cumprod

    # Forward process calculations q(x_t | x_0)
    sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = np.sqrt(1. - alphas_cumprod)

    # ---------------------------------------------------------
    # 2. Plotting
    # ---------------------------------------------------------

    fig = plt.figure(figsize=(12, 14))
    plt.suptitle(f"DDIM Diffusion Schedule for {num_timesteps} steps", fontsize=16)

    # Helper to generate x-axis
    T = np.arange(num_timesteps)

    NROWS = 5
    NCOLS = 2

    # --- Row 1: The Base Schedule (Alpha & Beta) ---
    ax1 = plt.subplot(NROWS, NCOLS, 2)
    ax1.plot(T, betas, color='tab:red')
    ax1.set_title(r"$\beta_t$")
    ax1.set_ylabel("Value")
    ax1.grid(True, alpha=0.3)

    ax2 = plt.subplot(NROWS, NCOLS, 1)
    ax2.plot(T, alphas, color='tab:blue')
    ax2.set_title(r"$\alpha_t$")
    ax2.grid(True, alpha=0.3)

    # --- Row 2: Cumulative Product ---
    ax3 = plt.subplot(NROWS, NCOLS, 3)
    ax3.plot(T, alphas_cumprod, color='tab:red')
    ax3.set_title(r"$\bar{\alpha}_t$")
    ax3.set_ylabel("Remaining Signal Variance")
    ax3.grid(True, alpha=0.3)

    ax3 = plt.subplot(NROWS, NCOLS, 4)
    ax3.plot(T, one_minus_alphas_cumprod, color='tab:blue')
    ax3.set_title(r"$1-\bar{\alpha}_t$")
    ax3.set_ylabel("Added Noise Variance")
    ax3.grid(True, alpha=0.3)

    # --- Row 3: Forward Process Scaling Factors ---
    ax4 = plt.subplot(NROWS, NCOLS, 5)
    ax4.plot(T, sqrt_alphas_cumprod, color='tab:red')
    ax4.set_title(r"$\sqrt{\bar{\alpha}_t}$")
    ax4.set_ylabel("Signal std.")
    ax4.grid(True, alpha=0.3)

    ax5 = plt.subplot(NROWS, NCOLS, 6)
    ax5.plot(T, sqrt_one_minus_alphas_cumprod, color='tab:blue')
    ax5.set_title(r"$\sqrt{1 - \bar{\alpha}_t}$")
    ax5.set_ylabel("Noise std.")
    ax5.grid(True, alpha=0.3)

    t = np.linspace(0, 1, num_timesteps)
    ax6 = plt.subplot(NROWS, NCOLS, 7)
    ax6.plot(T, 1/(1+t), color='tab:red')
    ax6.set_title(r"$\frac{1}{1+t}$")
    ax6.set_ylabel("Sig Var (non VP)")
    ax6.grid(True, alpha=0.3)

    ax5 = plt.subplot(NROWS, NCOLS, 8)
    ax5.plot(T, t/(1+t), color='tab:blue')
    ax5.set_title(r"$\frac{t}{1+t}$")
    ax5.grid(True, alpha=0.3)

    # Add plot for (1−ᾱt)/ᾱt
    ax6 = plt.subplot(NROWS, NCOLS, 9)
    ax6.plot(T, one_minus_alphas_cumprod / alphas_cumprod, color='tab:purple')
    ax6.set_title(r"$\frac{1-\bar{\alpha}_t}{\bar{\alpha}_t}$")
    ax6.set_ylabel("Weightage in SPL")
    ax6.grid(True, alpha=0.3)
	

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust for suptitle
    plt.show()

if __name__ == "__main__":
    plot_diffusion_schedule()