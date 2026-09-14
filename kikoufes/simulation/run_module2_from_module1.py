import numpy as np
from pathlib import Path

from module2 import (
    run_module2,
    export_prcpy_scan_files,
    default_params,
    load_module1_result,
)


def main():
    root = Path(__file__).resolve().parent

    # --- Module1の出力(npz)を読み込む ---
    # run_module1_from_npy.py が保存している想定のファイル名
    module1_npz_path = root / "module1_output.npz"

    if not module1_npz_path.exists():
        raise FileNotFoundError(
            f"Module1 output not found: {module1_npz_path}\n"
            f"先に run_module1_from_npy.py 等でModule1を実行し、"
            f"module1_output.npz をこのディレクトリに置いてください。"
        )

    phi_C_obs, phi_TC_obs, phi_Sk_obs, mz_obs = load_module1_result(module1_npz_path)

    print("Loaded Module1 output:")
    print(f"  phi_C_obs  shape = {phi_C_obs.shape}")
    print(f"  phi_TC_obs shape = {phi_TC_obs.shape}")
    print(f"  phi_Sk_obs shape = {phi_Sk_obs.shape}")
    print(f"  mz_obs     shape = {mz_obs.shape}")

    # --- Module2 パラメータ ---
    # 必要に応じて default_params() の値を上書きしてここで調整する
    params = default_params()

    # H_c2 は Module1 側の params["H_c2"] と合わせる
    H_c2 = 170.0

    # --- Module2 実行: 全ステップ(N_step=1)でΔS11(f)を合成 ---
    f_axis, N_indices, H_arr, S11_arr = run_module2(
        phi_C_obs,
        phi_TC_obs,
        phi_Sk_obs,
        mz_obs,
        H_c2=H_c2,
        N_start=0,
        N_end=None,   # 配列の末尾まで
        N_step=1,     # 全サイクルを出力（reservoir評価用）
        params=params,
    )

    print(f"\nModule2 synthesized {len(N_indices)} spectra "
          f"(N={N_indices[0]}..{N_indices[-1]})")

    # --- PRCpy互換のscanファイル群として出力 ---
    output_dir = root / "data_full" / "Cu2OSeO3" / "skyrmion"
    written_paths = export_prcpy_scan_files(
        f_axis, N_indices, H_arr, S11_arr,
        output_dir=output_dir,
        scan_start_index=1,
    )

    print(f"\nWrote {len(written_paths)} scan files to: {output_dir}")
    print("Example files:")
    for p in written_paths[:3]:
        print(" ", p.name)
    print("  ...")
    for p in written_paths[-3:]:
        print(" ", p.name)


if __name__ == "__main__":
    main()