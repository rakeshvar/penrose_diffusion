import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

class SinkhornLoss:
    def compute_loss(self, xysc_0, xysc_t, noise, noise_hat, colors, t):
        B = xysc_t.shape[0]
        σₓ = self.diffuser.rᾱ[t]       # B, 1, 1     # type: ignore
        σₑ = self.diffuser.r1mᾱ[t]                    # type: ignore
        twoσₑsqrd = 2.*(σₑ**2.)         # 2 *(1-ᾱₜ)
        total_loss = 0.

        σₓxysc0 = σₓ * xysc_0

        for b in range(B):
            noise_target = torch.zeros_like(xysc_t[0])

            for col in (0, 1):
                this_color = (colors[b] == col).nonzero(as_tuple=False).squeeze(1)

                xyscₜ_bc = xysc_t[b, this_color, :]
                σₓxysc0_bc = σₓxysc0[b, this_color, :]
                diff = xyscₜ_bc[:, None, :] - σₓxysc0_bc[None, :, :]   # N₀, N₀, d
                cost = (diff ** 2).sum(dim=-1)                         # N₀, N₀

                soft_assignment = sinkhorn_permutation(cost/twoσₑsqrd[b])
                σₓxysc0_post = soft_assignment @ σₓxysc0_bc

                noise_target[this_color] = (xyscₜ_bc - σₓxysc0_post) / σₑ[b]

            total_loss += F.mse_loss(noise_hat[b], noise_target)

        return total_loss / B


def sinkhorn_permutation(sq_dist_scaled, n_iters=7):
    log_K = -sq_dist_scaled
    log_u = torch.zeros_like(sq_dist_scaled[0])
    log_v = torch.zeros_like(sq_dist_scaled[0])

    for _ in range(n_iters):
        log_u = - torch.logsumexp(log_K + log_v[None, :], dim=1)
        log_v = - torch.logsumexp(log_K + log_u[:, None], dim=0)

    log_P = log_K + log_u[:, None] + log_v[None, :]
    return torch.exp(log_P)



def sinkhorn_permutation_iter(sq_dist, scaling, n_iters):
    N = sq_dist.shape[0]
    device = sq_dist.device

    log_K = -sq_dist / scaling
    log_u = torch.zeros(N, device=device)
    log_v = torch.zeros(N, device=device)
    log_a = torch.zeros(N, device=device)
    log_b = torch.zeros(N, device=device)

    P_history = []
    
    for i in range(n_iters):
        log_P = log_K + log_u[:, None] + log_v[None, :]
        P = torch.exp(log_P)
        P_history.append(P.cpu().numpy())
        
        log_u = log_a - torch.logsumexp(log_K + log_v[None, :], dim=1)
        log_v = log_b - torch.logsumexp(log_K + log_u[:, None], dim=0)
    
    return P_history

def compute_axis_limits(x0, xt):
    all_points = np.vstack([x0.numpy(), xt.numpy()])
    margin = 0.05
    abs_max = max(abs(all_points.max()), abs(all_points.min())) * (1 + margin)
    return -abs_max, abs_max


def draw_arrow(ax, start, end, color='k', alpha=1.0, width=1.0, with_head=True):
    style = '->' if with_head else '-'
    arrow = FancyArrowPatch(start, end, 
                           color=color, alpha=alpha,
                           arrowstyle=style, mutation_scale=15,
                           linewidth=width)
    ax.add_patch(arrow)


def pretty_print_P(P, iteration):
    print(f"\n=== Iteration {iteration} ===")
    P_int = (P * 100).astype(int)
    row_sums = P_int.sum(axis=1)
    col_sums = P_int.sum(axis=0)
    N = P.shape[0]
    minrow, mincol = row_sums.min(), col_sums.min()
    
    col_width = 4
    print(" ".rjust(col_width), end="")
    for j in range(N):
        print(f"T{j}".rjust(col_width), end="")
    print()

    for i in range(N):
        print(f"E{i}".rjust(col_width), end="")
        for j in range(N):
            print(f"{P_int[i, j]}".rjust(col_width), end="")
        print(f"{row_sums[i]}".rjust(col_width), "*" if row_sums[i] == minrow else "")
    
    print(" ".rjust(col_width), end="")
    for j in range(N):
        print(f"{'*' if col_sums[j] == mincol else ''}{col_sums[j]}".rjust(col_width), end="")
    print()
    

def compute_errors(P_history):
    P_true = np.eye(P_history[0].shape[0])
    errors = [np.linalg.norm(P - P_true, 'fro') for P in P_history]
    return errors


def draw_frame(ax_main, ax_error, x0, xt, colors, P, iteration, total_iters, limits, t, errors, max_error):
    ax_main.clear()
    
    lim_min, lim_max = limits
    ax_main.set_xlim(lim_min, lim_max)
    ax_main.set_ylim(lim_min, lim_max)
    ax_main.set_aspect('equal')
    ax_main.grid(True, alpha=0.3)
    ax_main.set_title(f'Sinkhorn Iteration {iteration}/{total_iters}, t={t:.2f} (Press up/down to navigate)', fontsize=14)

    # Assignments and probabilities
    N = P.shape[0]
    if N <= 12:
      Pr = (100*P).astype(int)
      for i in range(N):
        for j in range(N):
            alpha, alphar = P[i, j], Pr[i, j]
            if alphar > 0:
                alpha_vis = np.sqrt(alpha)
                draw_arrow(ax_main, xt[i].numpy(), x0[j].numpy(), 
                          color='red', alpha=alpha_vis, width=3.0)
                
                mid_x = (xt[i, 0].item() + x0[j, 0].item()) / 2
                mid_y = (xt[i, 1].item() + x0[j, 1].item()) / 2
                ax_main.text(mid_x, mid_y, alphar, fontsize=7, 
                        ha='center', va='center', color='darkred', weight='bold',
                        bbox=dict(boxstyle='round,pad=0.2', 
                                  facecolor='white', 
                                  alpha=0.7, 
                                  edgecolor='none'),
                        zorder=5)

    
    # Into truth
    col_sums = (100*P.sum(axis=0)).astype(int)
    for i in range(len(x0)):
        ax_main.text(x0[i, 0], x0[i, 1], str(i), 
                     fontsize=10, ha='center', va='center', 
                     color='white', zorder=4)
        ax_main.text(x0[i, 0]+.1, x0[i, 1], col_sums[i], 
                     fontsize=10, ha='center', va='top', 
                     color='green', weight='bold', zorder=4)
    
    # Out of estimate
    row_sums = (100*P.sum(axis=1)).astype(int)
    for i in range(len(xt)):
        ax_main.text(xt[i, 0], xt[i, 1], str(i), 
                     fontsize=10, ha='center', va='center', 
                     color='white', zorder=4)
        ax_main.text(xt[i, 0]+.1, xt[i, 1], row_sums[i], 
                     fontsize=10, ha='center', va='bottom', 
                     color='orchid', weight='bold', zorder=4)

    # Movement
    for i in range(len(x0)):
        draw_arrow(ax_main, x0[i].numpy(), xt[i].numpy(), color='gray', alpha=0.4, width=1.5, with_head=False)
    
    # truth and estimate
    ax_main.scatter(x0[colors==0, 0], x0[colors==0, 1], marker='o', c='darkgreen', s=100, label='x0', zorder=3)
    ax_main.scatter(x0[colors==1, 0], x0[colors==1, 1], marker='o', c='purple', s=100, label='x0', zorder=3)
    ax_main.scatter(xt[colors==0, 0], xt[colors==0, 1], marker='8', c='lightgreen', s=100, label='xt', zorder=3)
    ax_main.scatter(xt[colors==1, 0], xt[colors==1, 1], marker='8', c='orchid', s=100, label='xt', zorder=3)
    ax_main.legend(loc='upper right')

    iteration += 1
    # Error plot
    ax_error.clear()
    ax_error.plot(range(iteration), errors[:iteration], 'b-o', linewidth=2, markersize=4)
    ax_error.set_xlabel('Iteration', fontsize=12)
    ax_error.set_ylabel('||P - P*|| (Frobenius norm)', fontsize=12)
    ax_error.set_title('Convergence to Final P', fontsize=12)
    ax_error.grid(True, alpha=0.3)
    # ax_error.set_yscale('log')
    for i in range(iteration):
        ax_error.text(i, errors[i], f'{errors[i]:.2f}', 
                         fontsize=8, ha='center', va='bottom')
    
    plt.draw()


import sys
def main():
    N = 96 if len(sys.argv) < 2 else int(sys.argv[1])
    var = 1. if len(sys.argv) < 3 else float(sys.argv[2])
    n_iters = 12 if len(sys.argv) < 4 else int(sys.argv[3])
    
    x0 = torch.randn(N, 2)
    xt = x0 + torch.randn(N, 2) * np.sqrt(var)
    colors = torch.arange(N) % 2
    sq_dist = torch.cdist(xt, x0, p=2) ** 2
    print((100*sq_dist).round().int())
    sq_dist = torch.where(colors[None, :] == colors[:, None], sq_dist, 100*var)
    print((100*sq_dist).round().int())
    P_history = sinkhorn_permutation_iter(sq_dist, 2*var, n_iters)
    
    errors = compute_errors(P_history)
    max_error = max(errors)
    
    fig = plt.figure(figsize=(16, 8))
    ax_main = plt.subplot(1, 2, 1)
    ax_error = plt.subplot(1, 2, 2)
    
    limits = compute_axis_limits(x0, xt)
    
    current_frame = 0
    
    def on_key(event):
        nonlocal current_frame
        
        new_frame = current_frame
        if event.key in ('enter', 'up'):
            new_frame = (current_frame+1) % len(P_history)
        elif event.key in ('backspace', 'down'):
            new_frame = (current_frame-1) % len(P_history)        
        elif event.key == 'home':
            new_frame = 0
        elif event.key == 'end':
            new_frame = len(P_history)-1
        
        if new_frame != current_frame:
            current_frame = new_frame
            P = P_history[current_frame]
            pretty_print_P(P, current_frame)
            draw_frame(ax_main, ax_error, x0, xt, colors, P, current_frame, len(P_history), limits, var, errors, max_error)
    
    fig.canvas.mpl_connect('key_press_event', on_key)
    print(f"{N} tiles, {var} variance, {n_iters} iterations, {len(P_history)} frames")
    
    P = P_history[0]
    pretty_print_P(P, "None")
    draw_frame(ax_main, ax_error, x0, xt, colors, P, current_frame, len(P_history), limits, var, errors, max_error)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()