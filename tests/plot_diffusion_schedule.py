import numpy as np
import matplotlib.pyplot as plt
import math

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
    plt.suptitle(f"DDIM Diffusion Schedule Buffers (Steps={num_timesteps})", fontsize=16)
    
    # Helper to generate x-axis
    t = np.arange(num_timesteps)

    # --- Row 1: The Base Schedule (Alpha & Beta) ---
    ax1 = plt.subplot(3, 2, 2)
    ax1.plot(t, betas, color='tab:red')
    ax1.set_title(r"betas ($\beta_t$)")
    ax1.set_ylabel("Value")
    ax1.grid(True, alpha=0.3)

    ax2 = plt.subplot(3, 2, 1)
    ax2.plot(t, alphas, color='tab:blue')
    ax2.set_title(r"alphas ($\alpha_t$)")
    ax2.grid(True, alpha=0.3)

    # --- Row 2: Cumulative Product ---
    ax3 = plt.subplot(3, 2, 3) 
    ax3.plot(t, alphas_cumprod, color='tab:purple')
    ax3.set_title(r"alphas_cumprod ($\bar{\alpha}_t$)")
    ax3.set_ylabel("Signal Variance Remaining")
    ax3.grid(True, alpha=0.3)

    ax3 = plt.subplot(3, 2, 4) 
    ax3.plot(t, one_minus_alphas_cumprod, color='tab:purple')
    ax3.set_title(r"1 - alphas_cumprod ($1-\bar{\alpha}_t$)")
    ax3.set_ylabel("Noise Variance Remaining")
    ax3.grid(True, alpha=0.3)

    # --- Row 3: Forward Process Scaling Factors ---
    ax4 = plt.subplot(3, 2, 5)
    ax4.plot(t, sqrt_alphas_cumprod, color='tab:green')
    ax4.set_title(r"sqrt_alphas_cumprod ($\sqrt{\bar{\alpha}_t}$)")
    ax4.set_xlabel("Timestep t")
    ax4.grid(True, alpha=0.3)

    ax5 = plt.subplot(3, 2, 6)
    ax5.plot(t, sqrt_one_minus_alphas_cumprod, color='tab:orange')
    ax5.set_title(r"sqrt_one_minus_alphas_cumprod ($\sqrt{1 - \bar{\alpha}_t}$)")
    ax5.set_xlabel("Timestep t")
    ax5.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust for suptitle
    plt.show()

if __name__ == "__main__":
    plot_diffusion_schedule()