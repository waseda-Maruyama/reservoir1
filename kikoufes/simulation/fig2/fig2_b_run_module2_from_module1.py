import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from module2 import run_module2_fig2b_style, plot_waterfall, load_module1_result, default_params


root = Path(__file__).resolve().parent
module1_output = root / "fig2_module1_output.npz"

if not module1_output.exists():
    raise FileNotFoundError(
        f"module1 output not found: {module1_output}. "
        "Run run_module1_from_npy.py first."
    )

phi_C_obs, phi_TC_obs, phi_Sk_obs, mz_obs = load_module1_result(module1_output)

H_c2 = 170.0
params = default_params()

f_axis, N_indices, H_arr, S11_arr = run_module2_fig2b_style(
    phi_C_obs=phi_C_obs,
    phi_TC_obs=phi_TC_obs,
    phi_Sk_obs=phi_Sk_obs,
    mz_obs=mz_obs,
    H_c2=H_c2,
    N_start=100,
    N_end=200,
    N_step=10,
    params=params,
)

print("N indices:", N_indices)
print("H at each N (mT):", np.round(H_arr, 2))
print("S11 shape:", S11_arr.shape)

fig, ax = plot_waterfall(f_axis, N_indices, S11_arr, title="Module2 from Module1 output")
fig.savefig(root / "module2_from_module1.png", dpi=150)
print(f"Saved: {root / 'module2_from_module1.png'}")
