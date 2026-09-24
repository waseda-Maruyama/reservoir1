import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from module2 import (
    default_params,
    export_prcpy_scan_files,
    load_module1_result,
    plot_waterfall,
    run_module2,
)


root = Path(__file__).resolve().parent
module1_output = root / "module1_output.npz"

if not module1_output.exists():
    raise FileNotFoundError(
        f"module1 output not found: {module1_output}. "
        "Run run_module1_from_npy.py first."
    )

phi_He_obs, phi_TC_obs, phi_Sk_obs, mz_obs = load_module1_result(module1_output)

H_c2 = 170.0
n_end = len(mz_obs) - 1
output_root = root / "data_full" / "Cu2OSeO3"

# 旧Model 0〜3(TC有無・非線形指数pの掃引)は、理論的根拠のない要素の削除に伴い廃止。
# 現時点で文献根拠のある唯一の残存アブレーション軸は split_sk_modes
# (スキルミオンをCCW/breathingの2モードに分割するか、単一モードにまとめるか)。
variants = {
    "SkUnified": {"split_sk_modes": False},
    "SkSplit": {"split_sk_modes": True},
}

for label, overrides in variants.items():
    params = default_params()
    params.update(overrides)

    f_axis, N_indices, H_arr, S11_arr = run_module2(
        phi_He_obs=phi_He_obs,
        phi_TC_obs=phi_TC_obs,
        phi_Sk_obs=phi_Sk_obs,
        mz_obs=mz_obs,
        H_c2=H_c2,
        N_start=0,
        N_end=n_end,
        N_step=1,
        params=params,
    )

    output_dir = output_root / label
    written = export_prcpy_scan_files(
        f_axis,
        N_indices,
        H_arr,
        S11_arr,
        output_dir,
    )

    print(
        f"[{label}] split_sk_modes={params['split_sk_modes']}, "
        f"include_helical_n2={params['include_helical_n2']}"
    )
    print(f"  Wrote {len(written)} scan files to: {output_dir}")

    plot_positions = [
        i for i, n in enumerate(N_indices)
        if 200 <= n <= 300 and (n - 100) % 10 == 0
    ]
    fig, ax = plot_waterfall(
        f_axis,
        N_indices[plot_positions],
        S11_arr[plot_positions],
        title=f"{label} from Module1 output",
    )
    figure_path = root / f"module2_from_module1_{label}.png"
    fig.savefig(figure_path, dpi=150)
    print(f"  Saved: {figure_path}")