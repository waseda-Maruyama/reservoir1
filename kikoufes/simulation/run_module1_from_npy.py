import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from module1 import simulate_module1_step31


def main():
    root = Path(__file__).resolve().parent
    trajectory_path = root / "H_trajectory_920_sine.npy"

    if not trajectory_path.exists():
        raise FileNotFoundError(f"Trajectory file not found: {trajectory_path}")

    H_trajectory = np.load(trajectory_path)
    H_trajectory = np.asarray(H_trajectory, dtype=float).ravel()

    # ここは物理モデルのパラメータに合わせて調整する
    params = {
        "H_c2": 170.0,
        "Gamma_C_TC": 0.02,
        "Gamma_TC_C": 0.008,
        "gamma_0": 0.1,
        "m": 3.0,
        "nu_0": 1.0,
        "delta_E0": 0.5,
        "kappa": 4.0,
        "k_B_T": 0.05,
        "H_min": 25.0,
        "H_max": 120.0,
        "dt": 1.0,
    }

    n_obs, mz_obs, phi_C_obs, phi_TC_obs, phi_Sk_obs = simulate_module1_step31(
        H_trajectory, params, return_all=True
    )

    print("Loaded H trajectory:")
    print(f"  shape = {H_trajectory.shape}")
    print(f"  min = {H_trajectory.min():.3f}")
    print(f"  max = {H_trajectory.max():.3f}")
    print()
    print("Simulation output:")
    print(f"  n_obs shape = {n_obs.shape}")
    print(f"  mz_obs shape = {mz_obs.shape}")
    print(f"  phi_C_obs shape = {phi_C_obs.shape}")
    print(f"  phi_TC_obs shape = {phi_TC_obs.shape}")
    print(f"  phi_Sk_obs shape = {phi_Sk_obs.shape}")
    print(f"  n_obs first 10 = {n_obs[:10]}")
    print(f"  mz_obs first 10 = {mz_obs[:10]}")
    print(f"  phi_C_obs first 10 = {phi_C_obs[:10]}")
    print(f"  phi_TC_obs first 10 = {phi_TC_obs[:10]}")
    print(f"  phi_Sk_obs first 10 = {phi_Sk_obs[:10]}")

    np.savez(
        root / "module1_output.npz",
        n_obs=n_obs,
        mz_obs=mz_obs,
        phi_C_obs=phi_C_obs,
        phi_TC_obs=phi_TC_obs,
        phi_Sk_obs=phi_Sk_obs,
    )
    print(f"\nSaved: {root / 'module1_output.npz'}")

    x = np.arange(len(n_obs))

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    axes[0].plot(x, n_obs, "C0-o", markersize=2, label="n_obs")
    axes[0].set_ylabel("n_obs")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(x, mz_obs, "C1-o", markersize=2, label="mz_obs")
    axes[1].set_ylabel("mz_obs")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(x, phi_C_obs, "C2-o", markersize=2, label="phi_C")
    axes[2].plot(x, phi_TC_obs, "C3-o", markersize=2, label="phi_TC")
    axes[2].plot(x, phi_Sk_obs, "C4-o", markersize=2, label="phi_Sk")
    axes[2].set_xlabel("Observation step")
    axes[2].set_ylabel("Phase fraction")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    fig.suptitle("Module1 simulation output from H_trajectory_920_const.npy")
    fig.tight_layout()
    fig.savefig(root / "module1_output.png", dpi=150)
    print(f"Saved plot: {root / 'module1_output.png'}")


if __name__ == "__main__":
    main()
