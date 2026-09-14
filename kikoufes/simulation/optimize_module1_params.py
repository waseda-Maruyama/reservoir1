import numpy as np
from scipy.optimize import minimize
from pathlib import Path

from module1 import simulate_module1_step31


root = Path(__file__).resolve().parent
H_trajectory = np.load(root / "H_trajectory_920_const.npy").astype(float).ravel()

base_params = {
    "H_c2": 170.0,
    "Gamma_C_TC": 0.02,
    "Gamma_TC_C": 0.008,
    "m": 3.0,
    "nu_0": 1.0,
    "H_min": 25.0,
    "H_max": 120.0,
    "dt": 1.0,
}

# Fig 2b から読み取った目標点
# 近い実測値に合わせたペア
# 0.0 付近の値が大きくぶれないように、0.0 を 0.01 で丸める
# R^2 ではなく誤差平方和で最適化
# ここでは n_obs のピーク強度に対して最適化する

target_points = {
    100: 0.0,
    150: 0.6,
    180: 0.1,
}


def loss(x):
    gamma_0, delta_E0, kappa, k_B_T = x
    p = {
        **base_params,
        "gamma_0": gamma_0,
        "delta_E0": delta_E0,
        "kappa": kappa,
        "k_B_T": k_B_T,
    }
    n_obs, _, _ = simulate_module1_step31(H_trajectory, p)
    err = 0.0
    for idx, target in target_points.items():
        if idx >= len(n_obs):
            continue
        pred = float(n_obs[idx])
        err += (pred - target) ** 2
    return err


if __name__ == "__main__":
    x0 = [0.003, 0.5, 4.0, 0.05]
    res = minimize(loss, x0=x0, method="Nelder-Mead")

    print("Optimization result:")
    print(res)
    print("\nBest parameters:")
    print({
        "gamma_0": res.x[0],
        "delta_E0": res.x[1],
        "kappa": res.x[2],
        "k_B_T": res.x[3],
    })

    best_params = {
        **base_params,
        "gamma_0": res.x[0],
        "delta_E0": res.x[1],
        "kappa": res.x[2],
        "k_B_T": res.x[3],
    }
    n_obs, mz_obs, phi_C_obs, phi_TC_obs, phi_Sk_obs = simulate_module1_step31(
        H_trajectory, best_params, return_all=True
    )

    print("\nPredicted values at target indices:")
    for idx, target in target_points.items():
        if idx < len(n_obs):
            print(idx, "target=", target, "pred=", float(n_obs[idx]))

    np.savez(root / "module1_optimized_params.npz", **best_params)
    print(f"\nSaved parameters to {root / 'module1_optimized_params.npz'}")
