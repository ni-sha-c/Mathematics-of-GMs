import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import torch
import torch.nn as nn
from torchcfm.conditional_flow_matching import ConditionalFlowMatcher, VariancePreservingConditionalFlowMatcher


def sample_two_moons(n, noise=0.1, rng=None):
    """
    Sample n points from a two moons distribution.

    Parameters
    ----------
    n : int
        Number of samples.
    noise : float
        Standard deviation of Gaussian noise added to the samples.
    rng : np.random.Generator, optional
        Random number generator for reproducibility.

    Returns
    -------
    X : np.ndarray of shape (n, 2)
        Samples from the two moons distribution.
    """
    if rng is None:
        rng = np.random.default_rng()

    n0 = n // 2
    n1 = n - n0

    # Upper moon
    theta0 = rng.uniform(0, np.pi, size=n0)
    X0 = np.stack([np.cos(theta0), np.sin(theta0)], axis=1)

    # Lower moon (shifted right and down)
    theta1 = rng.uniform(0, np.pi, size=n1)
    X1 = np.stack([1 - np.cos(theta1), 1 - np.sin(theta1) - 0.5], axis=1)

    X = np.concatenate([X0, X1], axis=0)
    X += rng.normal(scale=noise, size=X.shape)
    return X


def plot_two_moons_kde(X, ax=None, resolution=200):
    """
    Plot a 2D KDE of samples from the two moons distribution.

    Parameters
    ----------
    X : np.ndarray of shape (n, 2)
        Samples, e.g. from sample_two_moons.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, a new figure is created.
    resolution : int
        Number of grid points per axis for the KDE evaluation.

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots()

    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution),
    )
    grid = np.stack([xx.ravel(), yy.ravel()])

    kde = gaussian_kde(X.T)
    zz = kde(grid).reshape(xx.shape)

    ax.contourf(xx, yy, zz, levels=20, cmap="Blues")
    ax.set_aspect("equal")
    ax.set_xlabel("$x_1$", fontsize=20)
    ax.set_ylabel("$x_2$", fontsize=20)
    ax.tick_params(axis="both", labelsize=20)
    #ax.set_title("Two moons KDE")

    return ax


class _VectorField(nn.Module):
    """Simple MLP that parameterizes the time-dependent vector field v_theta(t, x)."""

    def __init__(self, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden_dim),   # input: (t, x1, x2)
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2),   # output: velocity in R^2
        )

    def forward(self, t, x):
        # t: (batch,) or scalar; x: (batch, 2)
        if t.dim() == 0:
            t = t.expand(x.shape[0])
        tx = torch.cat([t.unsqueeze(-1), x], dim=-1)
        return self.net(tx)


def train_otcfm(n_samples=5000, n_epochs=2000, batch_size=256,
                lr=1e-3, hidden_dim=256, sigma=0.0, rng=None):
    """
    Train a stochastic interpolant model from N(0, I) to the two moons distribution.

    Uses VariancePreservingConditionalFlowMatcher from torchcfm, which implements
    the variance-preserving stochastic interpolant:
        x_t = sin(pi/2 * t) x_1 + cos(pi/2 * t) x_0 + sigma * eps

    Parameters
    ----------
    n_samples : int
        Number of target samples pre-drawn from two moons.
    n_epochs : int
        Number of training iterations.
    batch_size : int
        Mini-batch size.
    lr : float
        Adam learning rate.
    hidden_dim : int
        Hidden layer width of the vector field MLP.
    sigma : float
        Noise level for the conditional flow.
    rng : np.random.Generator, optional
        RNG for two moons sampling.

    Returns
    -------
    model : _VectorField
        Trained vector field model.
    losses : list of float
        Training loss at each epoch.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Pre-sample target data
    X1_all = torch.tensor(
        sample_two_moons(n_samples, rng=rng), dtype=torch.float32
    )

    FM = VariancePreservingConditionalFlowMatcher(sigma=sigma)
    model = _VectorField(hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    losses = []
    for epoch in range(n_epochs):
        # Sample a mini-batch from the target
        idx = torch.randperm(n_samples)[:batch_size]
        x1 = X1_all[idx].to(device)

        # Sample source from N(0, I) independently
        x0 = torch.randn(batch_size, 2, device=device)

        # CFM: compute interpolated point and target velocity
        t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)

        # Predict vector field and compute MSE loss
        vt = model(t, xt)
        loss = ((vt - ut) ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        if (epoch + 1) % 200 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}  loss: {loss.item():.4f}")

    return model, losses


def sample_otcfm(model, n=1000, n_steps=100):
    """
    Generate samples by integrating the learned OT-CFM vector field.

    Starts from x0 ~ N(0, I) and applies Euler integration from t=0 to t=1.

    Parameters
    ----------
    model : _VectorField
        Trained vector field returned by train_otcfm.
    n : int
        Number of samples to generate.
    n_steps : int
        Number of Euler integration steps.

    Returns
    -------
    X : np.ndarray of shape (n, 2)
        Generated samples.
    """
    device = next(model.parameters()).device
    dt = 1.0 / n_steps

    model.eval()
    with torch.no_grad():
        x = torch.randn(n, 2, device=device)
        for i in range(n_steps):
            t = torch.full((n,), i * dt, device=device)
            x = x + model(t, x) * dt

    return x.cpu().numpy()


def plot_trajectories(model, n_trajectories=50, n_steps=100, axes=None,
                      x_range=(-2.5, 2.5), y_range=(-2.0, 2.0)):
    """
    Plot OT-CFM trajectories alongside linear interpolation paths.

    Two subplots are produced side by side using the same set of start points:
      - Left:  learned OT-CFM trajectories (Euler integration of v_theta)
      - Right: straight-line interpolations x_t = (1-t)*x0 + t*x1 between
               the same x0 and the corresponding OT-CFM endpoints x1.

    Parameters
    ----------
    model : _VectorField
        Trained vector field returned by train_otcfm.
    n_trajectories : int
        Number of trajectories to draw.
    n_steps : int
        Number of integration / interpolation steps.
    axes : array-like of two matplotlib.axes.Axes, optional
        If None, a new figure with two subplots is created.
    x_range, y_range : tuple of (float, float)
        Axis limits for both subplots.

    Returns
    -------
    axes : np.ndarray of two matplotlib.axes.Axes
    """
    device = next(model.parameters()).device
    dt = 1.0 / n_steps

    if axes is None:
        _, ax_cfm = plt.subplots(1, 1, figsize=(12, 5))
    

    # --- Integrate learned vector field ---
    model.eval()
    with torch.no_grad():
        x0 = torch.randn(n_trajectories, 2, device=device)
        x = x0.clone()
        path = [x.cpu().numpy().copy()]

        for i in range(n_steps):
            t = torch.full((n_trajectories,), i * dt, device=device)
            x = x + model(t, x) * dt
            path.append(x.cpu().numpy().copy())

    path = np.stack(path, axis=1)   # (n_trajectories, n_steps+1, 2)
    x0_np = path[:, 0, :]           # starting points
    x1_np = path[:, -1, :]          # OT-CFM endpoints

    # --- Left: OT-CFM trajectories ---
    for k in range(n_trajectories):
        ax_cfm.plot(path[k, :, 0], path[k, :, 1], color="b", linewidth=0.8, alpha=0.7)
        ax_cfm.plot(*x0_np[k], "s", color="k", markersize=6)
        ax_cfm.plot(*x1_np[k], "o", color="r", markersize=6)

    ax_cfm.set_xlim(*x_range)
    ax_cfm.set_ylim(*y_range)
    ax_cfm.set_aspect("equal")
    ax_cfm.set_xlabel("$x_1$", fontsize=20)
    ax_cfm.set_ylabel("$x_2$", fontsize=20)
    ax_cfm.tick_params(axis="both", labelsize=20)
    #ax_cfm.set_title("OT-CFM trajectories", fontsize=20)

    """
    # --- Right: linear interpolation between same x0 and x1 ---
    ts = np.linspace(0, 1, n_steps + 1)
    for k in range(n_trajectories):
        lin_path = (1 - ts[:, None]) * x0_np[k] + ts[:, None] * x1_np[k]
        ax_lin.plot(lin_path[:, 0], lin_path[:, 1], color="b", linewidth=0.8, alpha=0.7)
        ax_lin.plot(*x0_np[k], "s", color="k", markersize=6)
        ax_lin.plot(*x1_np[k], "o", color="r", markersize=6)

    ax_lin.set_xlim(*x_range)
    ax_lin.set_ylim(*y_range)
    ax_lin.set_aspect("equal")
    ax_lin.set_xlabel("$x_1$", fontsize=20)
    ax_lin.set_ylabel("$x_2$", fontsize=20)
    ax_lin.tick_params(axis="both", labelsize=20)
    #ax_lin.set_title("Linear interpolation", fontsize=20)
    """
    return axes


def plot_vector_field(model, times, resolution=20, ax=None, x_range=(-2.5, 2.5), y_range=(-2.0, 2.0)):
    """
    Plot the learned vector field v_theta(t, x) on a 2D grid at specified times.

    Parameters
    ----------
    model : _VectorField
        Trained vector field returned by train_otcfm.
    times : list of float
        Values of t in [0, 1] at which to plot the vector field.
    resolution : int
        Number of grid points per axis.
    ax : array of matplotlib.axes.Axes, optional
        One Axes per entry in times. If None, a new figure is created.
    x_range, y_range : tuple of (float, float)
        Spatial extent of the grid.

    Returns
    -------
    axes : np.ndarray of matplotlib.axes.Axes
    """
    device = next(model.parameters()).device

    if ax is None:
        fig, axes = plt.subplots(1, len(times), figsize=(4 * len(times), 4))
        if len(times) == 1:
            axes = np.array([axes])
    else:
        axes = np.asarray(ax).ravel()

    xs = np.linspace(*x_range, resolution)
    ys = np.linspace(*y_range, resolution)
    xx, yy = np.meshgrid(xs, ys)
    grid = np.stack([xx.ravel(), yy.ravel()], axis=1)  # (resolution^2, 2)
    grid_t = torch.tensor(grid, dtype=torch.float32, device=device)

    model.eval()
    with torch.no_grad():
        for ax_i, t_val in zip(axes, times):
            t = torch.full((grid_t.shape[0],), t_val, device=device)
            v = model(t, grid_t).cpu().numpy()  # (resolution^2, 2)

            vx = v[:, 0].reshape(xx.shape)
            vy = v[:, 1].reshape(yy.shape)
            speed = np.sqrt(vx**2 + vy**2)

            s = ax_i.contourf(xx, yy, speed, cmap="RdBu_r")
            ax_i.set_xlim(*x_range)
            ax_i.set_ylim(*y_range)
            ax_i.set_aspect("equal")
            ax_i.set_xlabel("$x_1$", fontsize=20)
            ax_i.set_ylabel("$x_2$", fontsize=20)
            ax_i.tick_params(axis="both", labelsize=20)
            cbar = plt.colorbar(s, ax=ax_i)
            cbar.ax.tick_params(labelsize=20)
            plt.tight_layout()
            ax_i.set_title(f"$t = {t_val:.2f}$", fontsize=20)

    return axes


if __name__ == "__main__":
    import os

    rng = np.random.default_rng(42)
    model_path = "cfm_two_moons.pt"

    if os.path.exists(model_path):
        print(f"Loading model from {model_path}...")
        model = _VectorField(hidden_dim=256)
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.eval()
    else:
        print("Training OT-CFM...")
        model, losses = train_otcfm(n_samples=5000, n_epochs=2000, batch_size=256, rng=rng)
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

    X_data = sample_two_moons(5000, rng=rng)
    ax = plot_two_moons_kde(X_data)
    plt.tight_layout()
    plt.savefig("two_moons_kde.pdf", bbox_inches="tight")
    plt.savefig("two_moons_kde.png", dpi=150, bbox_inches="tight")


    print("Plotting vector field...")
    times = [0.0, 0.5, 1.0]
    fig, axes = plt.subplots(1, len(times), figsize=(4 * len(times), 4))
    plot_vector_field(model, times, resolution=20, ax=axes)
    plt.tight_layout()
    plt.savefig("cfm_vector_field.pdf", bbox_inches="tight")
    plt.savefig("cfm_vector_field.png", dpi=150, bbox_inches="tight")

    print("Plotting trajectories...")
    plot_trajectories(model, n_trajectories=50, n_steps=100)
    plt.tight_layout()
    plt.savefig("cfm_trajectories.pdf", bbox_inches="tight")
    plt.savefig("cfm_trajectories.png", dpi=150, bbox_inches="tight")

    plt.show()
    print("Saved plots to otcfm_two_moons_kde.pdf, otcfm_vector_field.pdf, and otcfm_trajectories.pdf")
