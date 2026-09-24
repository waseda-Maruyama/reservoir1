"""
run_simulation.py --- Module1 -> Module2 -> PRCpy 入力ファイル生成 の一括実行。

使い方:
    python run_simulation.py                 # 既定設定で実行
    python run_simulation.py --n-cycles 500 --warmup 900

出力:
    <out>/data_full/Cu2OSeO3/skyrmion/scan_*.txt   PRCpy が読み込む scan ファイル
    <out>/module1_output.npz                       Module1 の状態 (相分率・psi・ドメイン別)
    <out>/diagnostics.png                          相分率とスペクトルの診断図
    <out>/metrics.json                             NL / MC / CP / MSE

生成後は、既存の main_skyrmion.py 等で
    data_dir_path = "<out>/data_full/Cu2OSeO3/skyrmion"
    process_params["sample_rate"] = 8      # main_skyrmion.py と揃える
として評価できる。

MG時系列を外部ファイルから与える場合 (ハイブリッド方式):
    python run_simulation.py --mg-file mackey_glass_t17.npy

--mg-file を指定すると、
  - 評価区間 (n-cycles分, ファイル先頭から --mg-offset 分ずらして取得) は
    そのファイルの実際の値をそのまま使う (main_skyrmion.py/data_skyrmion.py の
    target_values[:500] と同じ切り出し方)。PRCpy本番・Lee論文の指標と
    直接比較する対象なので、実測で使われた系列と厳密に一致させる。
  - ウォームアップ区間 (warmup分) は、記録前の磁場履歴がそもそもデータセットに
    含まれておらず復元不能なため、内部の generate_mackey_glass() で別途合成する。
    ここは相分率を定常状態に持っていく助走に過ぎず、具体的な値が評価区間と
    連続している必要はない (Module1のレート方程式は磁場強度の統計にしか
    反応しないため)。
  - 正規化 (0-1への min-max) は区間ごとに独立して行う。評価区間はファイル単体の
    min/maxで正規化する (実測での正規化と同じ) ので、合成したウォームアップの
    値域が混ざって評価区間のスケールが歪むことはない。
--mg-file を指定しない場合は、warmup+n-cycles全体を単一の generate_mackey_glass()
で生成する (従来通り)。
対応形式: .npy (1次元配列) / .csv, .txt (数値の列。--mg-column で列選択、
既定は先頭の数値列。区切り文字はカンマ・空白・タブを自動判定)。
"""
import argparse
import json
from pathlib import Path

import numpy as np

import module1 as m1
import module2 as m2


def load_mg_series(path, column=0):
    """
    外部ファイルから1次元のMackey-Glass時系列を読み込む。

    - .npy       : np.load。1次元ならそのまま、2次元なら column 列目を使用。
    - .csv/.txt  : ヘッダの有無・区切り文字(, / 空白 / タブ)を自動判定して読み込み、
                   1次元ならそのまま、2次元(多列)なら column 列目を使用。
    それ以外の拡張子は同様にテキストとして解析を試みる。
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"MG時系列ファイルが見つかりません: {path}")

    suffix = path.suffix.lower()
    if suffix == ".npy":
        arr = np.load(path)
    else:
        # 区切り文字を推定 (カンマ優先 -> 空白/タブ)
        with open(path, "r") as fh:
            sample = fh.readline()
        delimiter = "," if "," in sample else None  # None -> 空白/タブ区切り

        # ヘッダ行(数値に変換できない行)がある場合は skiprows=1 で読み直す
        def _try_load(skiprows):
            return np.loadtxt(path, delimiter=delimiter, skiprows=skiprows)

        try:
            arr = _try_load(0)
        except ValueError:
            arr = _try_load(1)

    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 2:
        arr = arr[:, column]
    elif arr.ndim != 1:
        raise ValueError(f"MG時系列は1次元(または2次元で列選択)である必要があります: shape={arr.shape}")

    if not np.all(np.isfinite(arr)):
        raise ValueError("MG時系列にNaN/infが含まれています")

    return arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-cycles", type=int, default=500,
                    help="評価に使うサイクル数 (PRCpyでのRC評価入力長は500)")
    ap.add_argument("--warmup", type=int, default=900,
                    help="ウォームアップのサイクル数。Lee論文は 920 field-cycles 後の"
                         "状態でスペクトルを記録しているため、核生成の過渡期を"
                         "評価対象から外す")
    ap.add_argument("--Hc", type=float, default=73.0,
                    help="中心磁場 [mT]。実測ファイル名から復元した磁場範囲"
                         "28.00-117.90 mT は Hc=73, Hrange=90 と一致する")
    ap.add_argument("--Hrange", type=float, default=90.0, help="サイクル幅 [mT]")
    ap.add_argument("--n-onset", type=float, default=140.0,
                    help="スキルミオン相が立ち上がるサイクル数 (仮定するダイナミクス)")
    ap.add_argument("--out", type=str, default="./sim_out")
    ap.add_argument("--no-metrics", action="store_true", help="指標計算を省略する")
    ap.add_argument("--mg-file", type=str,
                    default=str(Path(__file__).resolve().parent / "mackey_glass_t17.npy"),
                    help="外部のMackey-Glass時系列ファイル (.npy/.csv/.txt)。"
                        "既定は kanseihin/mackey_glass_t17.npy。"
                        "空文字を指定した場合は内部生成を使う")
    ap.add_argument("--mg-column", type=int, default=0,
                    help="--mg-file が多列データの場合に使う列番号 (既定0)")
    ap.add_argument("--mg-offset", type=int, default=0,
                    help="--mg-file の時系列を読み飛ばす先頭サンプル数 "
                         "(評価区間の切り出し開始位置。main_skyrmion.py等の"
                         "target_values[:500]に合わせる場合は既定0のままでよい)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    total = args.warmup + args.n_cycles

    # ---- 入力時系列と磁場軌道 ----
    def _minmax_norm(x):
        return (x - x.min()) / (x.max() - x.min())

    if args.mg_file:
        # ハイブリッド方式:
        #   評価区間 -> --mg-file の実データをそのまま使用 (実測/PRCpy本番と同じ切り出し)
        #   ウォームアップ区間 -> 記録前の履歴は存在しないため合成 (generate_mackey_glass)
        #
        # 正規化は「ファイル全体の min-max で正規化してから切り出す」順序でなければならない
        # (get_npy_data(path, norm=True) はファイル全体を正規化してから [:n_cycles] を
        #  切り出す実装になっており、評価スクリプト側もこの順序で target_values を作っている)。
        # 500点の切り出し後にその場で再度min-max正規化すると、局所的な変動幅が
        # 無理やり[0,1]いっぱいに引き伸ばされ、実際より過大な振幅のu(N)になって
        # 磁場軌道が歪み、評価スクリプトの target_values と対応しなくなってしまう。
        mg_full = load_mg_series(args.mg_file, column=args.mg_column)
        if args.mg_offset + args.n_cycles > len(mg_full):
            raise ValueError(
                f"--mg-file の時系列が短すぎます: --mg-offset {args.mg_offset} から "
                f"{len(mg_full) - args.mg_offset} 点しか取れません "
                f"(必要 n-cycles={args.n_cycles})。"
                f"--n-cycles/--mg-offset を見直すか、より長い時系列を用意してください。"
            )
        file_norm_full = _minmax_norm(mg_full)  # ファイル全体で正規化 (get_npy_data(norm=True)と同じ順序)
        eval_norm = file_norm_full[args.mg_offset: args.mg_offset + args.n_cycles]

        warmup_raw = m1.generate_mackey_glass(args.warmup, tau=17) if args.warmup > 0 \
            else np.zeros(0)
        # ウォームアップは無関係な合成系列なので、それ自身の中で正規化してよい
        warmup_norm = _minmax_norm(warmup_raw) if args.warmup > 0 else warmup_raw
        mg_norm = np.concatenate([warmup_norm, eval_norm])

        print(f"[1/4] MG時系列(ハイブリッド): 評価区間 {args.n_cycles} 点を "
              f"{args.mg_file} (offset={args.mg_offset}, ファイル全体{len(mg_full)}点で正規化後に切り出し) "
              f"から実データとして使用、ウォームアップ {args.warmup} 点は generate_mackey_glass() で合成")
    else:
        mg = m1.generate_mackey_glass(total, tau=17)
        mg_norm = _minmax_norm(mg)

    u = 2.0 * mg_norm - 1.0
    H_traj = m1.build_field_trajectory(u, args.Hc, args.Hrange)
    print(f"[1/4] 磁場軌道: {H_traj.min():.2f} - {H_traj.max():.2f} mT "
          f"({total} サイクル = ウォームアップ {args.warmup} + 評価 {args.n_cycles})")

    # ---- Module 1 (自動校正つき) ----
    target = m1.TargetDynamics(n_onset=args.n_onset)
    p1 = m1.calibrate_module1(H_traj, target, verbose=True)
    r1 = m1.simulate_module1(H_traj, p1)
    print("[2/4] Module1 完了")

    np.savez(out / "module1_output.npz",
             **{k: v for k, v in r1.items()
                if isinstance(v, np.ndarray) and not k.startswith("full")})

    # ---- Module 2 ----
    p2 = m2.Module2Params()
    f_axis, N_idx, H_arr, S11 = m2.run_module2(r1, p2)
    sl = slice(args.warmup, total)
    f_e, N_e, H_e, S_e = f_axis, N_idx[sl] - args.warmup, H_arr[sl], S11[sl]
    print(f"[3/4] Module2 完了: S11 {S_e.shape}, "
          f"range [{S_e.min():.3f}, {S_e.max():.3f}]")

    scan_dir = out / "data_full" / "Cu2OSeO3" / "skyrmion"
    written = m2.export_prcpy_scan_files(f_e, N_e, H_e, S_e, scan_dir)
    print(f"[4/4] scan ファイル {len(written)} 個を書き出し: {scan_dir}")

    # ---- 指標 ----
    if not args.no_metrics:
        try:
            from fast_metrics import evaluate_fast
            met = evaluate_fast(S_e, u[sl], mg_norm[sl], sample_rate=8,
                                test_size=0.3)
            met.pop("mc_curve", None)
            print("\n--- reservoir metrics (sample_rate=8, main_skyrmion.py と同条件) ---")
            for k in ("n_feat", "NL", "MC", "CP", "train_MSE", "test_MSE"):
                print(f"  {k:<10}: {met[k]:.4g}")
            print("  実測(参考): NL=0.550  MC=4.942  CP=1.770  test_MSE=3.70e-03")
            json.dump(met, open(out / "metrics.json", "w"), indent=2)
        except Exception as e:  # noqa: BLE001
            print(f"(指標計算をスキップ: {e})")

    # ---- 診断図 ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(3, 2, figsize=(14, 10))
        n = len(r1["phi_Sk_obs"])
        x = np.arange(n)
        ax[0, 0].plot(x, r1["phi_He_obs"], label=r"$\phi_{He}$")
        ax[0, 0].plot(x, r1["phi_TC_obs"], label=r"$\phi_{TC}$")
        ax[0, 0].plot(x, r1["phi_Sk_obs"], label=r"$\phi_{Sk}$")
        ax[0, 0].axvline(args.n_onset, ls="--", c="k", lw=0.8)
        ax[0, 0].axvline(args.warmup, ls=":", c="r", lw=0.8)
        ax[0, 0].legend(); ax[0, 0].set_title("phase fractions"); ax[0, 0].set_xlabel("N")

        ax[0, 1].plot(x, r1["psi_obs"], c="crimson")
        ax[0, 1].set_title(r"annealing $\psi$ (long-term memory)"); ax[0, 1].set_xlabel("N")

        ax[1, 0].plot(x, r1["H_obs"], lw=0.6, c="purple")
        ax[1, 0].set_title("$H_{low}$ (readout field)"); ax[1, 0].set_xlabel("N")

        tail = slice(int(n * 0.6), n)
        ax[1, 1].scatter(r1["phi_Sk_obs"][tail], r1["phi_TC_obs"][tail], s=3)
        cc = np.corrcoef(r1["phi_Sk_obs"][tail], r1["phi_TC_obs"][tail])[0, 1]
        ax[1, 1].set_xlabel(r"$\phi_{Sk}$"); ax[1, 1].set_ylabel(r"$\phi_{TC}$")
        ax[1, 1].set_title(f"Sk-TC tradeoff (corr={cc:.3f})")

        sel = np.arange(0, 101, 10)
        step = 0.35 * (S_e.max() - S_e.min())
        for k, i in enumerate(sel):
            ax[2, 0].plot(f_e, S_e[i] + k * step, lw=0.9)
        ax[2, 0].set_title("waterfall (N=0-100 of eval window)")
        ax[2, 0].set_xlabel("f (GHz)"); ax[2, 0].set_yticks([])

        for k, i in enumerate([0, 20, 40, 60, 80]):
            ax[2, 1].plot(f_e, S_e[i] + k * 0.9, lw=0.9, label=f"N={i}")
        ax[2, 1].set_xlim(3.3, 5.6); ax[2, 1].legend(fontsize=7)
        ax[2, 1].set_title("4 GHz fine structure (cycle-to-cycle)")
        ax[2, 1].set_xlabel("f (GHz)"); ax[2, 1].set_yticks([])

        fig.tight_layout()
        fig.savefig(out / "diagnostics.png", dpi=110)
        print(f"診断図: {out / 'diagnostics.png'}")
    except Exception as e:  # noqa: BLE001
        print(f"(診断図をスキップ: {e})")


if __name__ == "__main__":
    main()