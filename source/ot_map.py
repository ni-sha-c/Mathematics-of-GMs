import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def plot_ot_map(alpha, b=0.5, ax=None):
    """
    Plot the optimal transport map from p_data to q_theta* = N(0, b^2).

    p_data: discrete on {-1, 1} with P(-1) = alpha, P(1) = 1 - alpha
    q_theta*: Gaussian N(0, b^2)

    In 1D OT, the map is given by the monotone rearrangement:
    - Mass alpha at x=-1 maps to the interval [Q(0), Q(alpha)] of the Gaussian
    - Mass (1-alpha) at x=1 maps to the interval [Q(alpha), Q(1)] of the Gaussian
    where Q is the quantile function of N(0, b^2).
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    # Quantile boundaries
    q_lower = norm.ppf(1e-4, loc=0, scale=b)  # approximate Q(0)
    q_alpha = norm.ppf(alpha, loc=0, scale=b)  # Q(alpha)
    q_upper = norm.ppf(1 - 1e-4, loc=0, scale=b)  # approximate Q(1)

    # Plot p_data (impulses at -1 and 1)
    ax.vlines(-1, 0, alpha, colors='navy', linewidth=3, label=r'$p_{\mathrm{data}}$')
    ax.plot(-1, alpha, 'o', color='navy', markersize=8)
    ax.vlines(1, 0, 1 - alpha, colors='navy', linewidth=3)
    ax.plot(1, 1 - alpha, 'o', color='navy', markersize=8)

    # Plot q_theta (Gaussian)
    x = np.linspace(-2, 2, 500)
    y = norm.pdf(x, loc=0, scale=b)
    ax.plot(x, y, 'maroon', linewidth=2, label=r'$q_{\theta^*} = \mathcal{N}(0, b^2)$')

    # Shade regions showing where mass is transported
    x_left = np.linspace(q_lower, q_alpha, 200)
    y_left = norm.pdf(x_left, loc=0, scale=b)
    ax.fill_between(x_left, y_left, alpha=0.3, color='blue',
                    label=f'Mass from $x=-1$ (area={alpha:.2f})')

    x_right = np.linspace(q_alpha, q_upper, 200)
    y_right = norm.pdf(x_right, loc=0, scale=b)
    ax.fill_between(x_right, y_right, alpha=0.3, color='red',
                    label=f'Mass from $x=1$ (area={1-alpha:.2f})')

    # Draw transport arrows
    arrow_y = max(alpha, 1 - alpha) + 0.1
    ax.annotate('', xy=(q_alpha/2 + q_lower/2, 0.15), xytext=(-1, alpha),
                arrowprops=dict(arrowstyle='->', color='blue', lw=1.5))
    ax.annotate('', xy=(q_alpha/2 + q_upper/2, 0.15), xytext=(1, 1 - alpha),
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5))

    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-0.05, max(1, norm.pdf(0, loc=0, scale=b)) + 0.2)
    ax.set_xlabel('$x$', fontsize=14)
    ax.set_title(r'Optimal Transport Map', fontsize=16)
    ax.tick_params(axis='both', labelsize=12)
    ax.legend(loc='upper right', fontsize=9)

    return ax


def plot_ot_map_reverse(alpha, i=1, b=0.5, ax=None):
    """
    Plot the optimal transport map from q_theta* = N(0, b^2) to p_data.

    q_theta*: Gaussian N(0, b^2)
    p_data: discrete on {-1, 1} with P(-1) = alpha, P(1) = 1 - alpha

    The OT map T: R -> {-1, 1} is given by:
    - T(x) = -1 if x <= Q(alpha)  (maps to mass alpha at -1)
    - T(x) = 1  if x > Q(alpha)   (maps to mass 1-alpha at 1)
    where Q is the quantile function of N(0, b^2).
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    # Quantile boundary
    q_alpha = norm.ppf(alpha, loc=0, scale=b)

    # Plot the OT map T(x)
    x_left = np.linspace(-2.5, q_alpha, 200)
    x_right = np.linspace(q_alpha, 2.5, 200)

    ax.plot(x_left, -np.ones_like(x_left), 'navy', linewidth=2.5)
    ax.plot(x_right, np.ones_like(x_right), 'maroon', linewidth=2.5)

    # Mark the transition point
    ax.axvline(q_alpha, color='gray', linestyle='--', linewidth=1.5)
    ax.plot(q_alpha, -1, 'o', color='navy', markersize=8)  # closed at -1
    ax.plot(q_alpha, 1, 'o', color='maroon', markersize=8, fillstyle='none')  # open at 1
    ax.plot(q_alpha, 0, 'k|', markersize=10)  # tick mark on x-axis
    ax.text(q_alpha, -0.25, r'$x(\alpha)$', ha='center', fontsize=12)

    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlabel('$x$', fontsize=14)
    ax.set_ylabel('$T(x)$', fontsize=14)
    ax.set_yticks([-1, 0, 1])
    ax.set_title(rf'$\alpha = \alpha_{i}$', fontsize=16)
    ax.tick_params(axis='both', labelsize=12)
    ax.axis("scaled")
    return ax


def plot_transport_plan(alpha, i=1, b=0.5, ax=None):
    """
    Plot the optimal transport plan from q_theta* = N(0, b^2) to p_data as a 2D contour.

    The transport plan gamma(x, y) is concentrated on the graph of the OT map:
    - gamma(x, -1) = q_theta*(x) for x <= Q(alpha)
    - gamma(x, 1) = q_theta*(x) for x > Q(alpha)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    # Quantile boundary
    q_alpha = norm.ppf(alpha, loc=0, scale=b)

    # Create 2D grid
    x = np.linspace(-2.5, 2.5, 500)
    y = np.linspace(-1.5, 1.5, 300)
    X, Y = np.meshgrid(x, y)

    # Transport plan: density concentrated near y=-1 for x<=q_alpha, y=1 for x>q_alpha
    # Use a narrow Gaussian in y to visualize the concentration
    sigma_y = 0.08  # width of the "smeared" delta function

    # Source density
    source_density = norm.pdf(x, loc=0, scale=b)

    # Transport plan as 2D density
    plan = np.zeros_like(X)
    for j, xj in enumerate(x):
        if xj <= q_alpha:
            # Mass goes to y = -1
            plan[:, j] = source_density[j] * norm.pdf(y, loc=-1, scale=sigma_y)
        else:
            # Mass goes to y = 1
            plan[:, j] = source_density[j] * norm.pdf(y, loc=1, scale=sigma_y)

    # Plot contour
    levels = np.linspace(0, plan.max(), 20)
    contour = ax.contourf(X, Y, plan, levels=levels, cmap='RdBu_r')
    #plt.colorbar(contour, ax=ax, label='Transport plan density')

    # Mark the transition point
    ax.axvline(q_alpha, color='white', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.text(q_alpha + 0.1, 0, r'$x(\alpha)$', color='white', fontsize=12, va='center')

    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlabel(r'Source $x \sim q_{\theta^*}$', fontsize=14)
    ax.set_ylabel(r'Target $y \in \{-1, 1\}$', fontsize=14)
    ax.set_yticks([-1, 0, 1])
    ax.set_title(rf'($\alpha = \alpha_{i}$)', fontsize=16)
    ax.tick_params(axis='both', labelsize=12)

    return ax


if __name__ == "__main__":
    alphas = [0.2, 0.9]

    # Plot OT map from q_theta* to p_data
    fig, axes = plt.subplots(1, 2)
    for i, (ax, alpha) in enumerate(zip(axes.flatten(), alphas), start=1):
        plot_ot_map_reverse(alpha, i=i, b=0.5, ax=ax)
    plt.tight_layout()
    plt.savefig('ot_map_qtheta_to_pdata.pdf', bbox_inches='tight')
    plt.savefig('ot_map_qtheta_to_pdata.png', dpi=150, bbox_inches='tight')

    # Plot transport plan as 2D contour
    alphas = [0.9, 0.2, 0.7, 0.01]

    fig2, axes2 = plt.subplots(1, 4, figsize=(12, 5))
    for i, (ax, alpha) in enumerate(zip(axes2.flatten(), alphas), start=1):
        plot_transport_plan(alpha, i=i, b=0.5, ax=ax)
    plt.tight_layout()
    plt.savefig('transport_plan_contour.pdf', bbox_inches='tight')
    plt.savefig('transport_plan_contour.png', dpi=150, bbox_inches='tight')

    plt.show()
    print("Saved plots to ot_map_qtheta_to_pdata.pdf and transport_plan_contour.pdf")
