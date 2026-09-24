"""
ablation_experiment.py --- Module1/Module2 のアブレーション実験を自動実行する。

「引き継ぎ&ポスター準備ドキュメント」4章および「最新の設計変更」に記載の
アブレーション結果表を、実際にコードを実行して再現・検証するためのスクリプト。
これまでの数値は前回セッションでの実行結果の書き起こしであり、本スクリプトで
初めて再実行可能な形にする。

パイプラインは run_simulation.py と同一 (MG時系列生成 -> 磁場軌道 -> Module1
-> Module2 -> fast_metrics.evaluate_fast) だが、8つのモデル設定 (フルモデル +
7つの要素除去) をまとめて実行し、比較表を CSV / 標準出力に整理する。

使い方:
    python ablation_experiment.py                       # 既定 (n_cycles=1000, warmup=900)
    python ablation_experiment.py --n-cycles 500 --warmup 900
    python ablation_experiment.py --mg-file mackey_glass_t17.npy   # 実測と同じMG系列を使う場合

出力:
    ablation_results.csv        各設定の NL/MC/CP/test_MSE/実測比
    (実測比は --measured-mse で指定した値、既定 3.70e-3, Lee et al. 実測値)

注意:
    - PRCpy 本体は使わず、fast_metrics.py (PRCpy と同一定義を再実装したもの) で
      評価する。run_simulation.py 内蔵の指標計算と同じ経路。
    - MG系列は既定では内部の generate_mackey_glass() で生成する (--mg-file 未指定時)。
      generate_mackey_glass() は乱数を使わない決定論的な数値積分なので、
      同じ引数であれば再実行しても同じ系列になる。
    - 各アブレーション条件の意味:
        full                  : フルモデル (何も除去しない)
        no_tc_fine            : TC微細構造(4GHz帯ギザギザ)を除去 (include_tc_fine=False)
        no_helical_n2         : ヘリカルn=2高調波を除去 (include_helical_n2=False)
        no_sk_split           : スキルミオンCCW/breathing分割を除去 (split_sk_modes=False)
        no_history_shift      : 履歴由来の周波数シフトを除去 (enable_history_shift=False)
        no_field_shift        : 磁場由来の周波数シフトを除去 (enable_field_shift=False)
        no_both_shifts        : 磁場・履歴シフトの両方を除去
        no_multidomain        : 多ドメイン化を外し単一平均場に戻す (Module1Params.n_domains=1)
                                 この条件のみ Module1 のレート自体を n_domains=1 で
                                 再校正する (calibrate_module1 の base に渡す)
"""
import argparse
import csv
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

import module1 as m1
import module2 as m2
from fast_metrics import evaluate_fast


def load_mg_series(path, column=0):
    """run_simulation.py の load_mg_series と同一。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"MG時系列ファイルが見つかりません: {path}")
    suffix = path.suffix.lower()
    if suffix == ".npy":
        arr = np.load(path)
    else:
        with open(path, "r") as fh:
            sample = fh.readline()
        delimiter = "," if "," in sample else None

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
        raise ValueError(f"MG時系列は1次元である必要があります: shape={arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("MG時系列にNaN/infが含まれています")
    return arr


def _minmax_norm(x):
    return (x - x.min()) / (x.max() - x.min())


def build_input(args):
    """run_simulation.py と同じロジックで MG系列・磁場軌道・評価スライスを作る。"""
    total = args.warmup + args.n_cycles

    if args.mg_file is not None:
        mg_full = load_mg_series(args.mg_file, column=args.mg_column)
        if args.mg_offset + args.n_cycles > len(mg_full):
            raise ValueError(
                f"--mg-file の時系列が短すぎます: {len(mg_full) - args.mg_offset} 点しか"
                f"取れません (必要 n-cycles={args.n_cycles})"
            )
        file_norm_full = _minmax_norm(mg_full)
        eval_norm = file_norm_full[args.mg_offset: args.mg_offset + args.n_cycles]
        warmup_raw = m1.generate_mackey_glass(args.warmup, tau=17) if args.warmup > 0 else np.zeros(0)
        warmup_norm = _minmax_norm(warmup_raw) if args.warmup > 0 else warmup_raw
        mg_norm = np.concatenate([warmup_norm, eval_norm])
        print(f"[入力] MG系列(ハイブリッド): 評価区間{args.n_cycles}点を{args.mg_file}から使用")
    else:
        mg = m1.generate_mackey_glass(total, tau=17)
        mg_norm = _minmax_norm(mg)
        print(f"[入力] MG系列: 内部生成 generate_mackey_glass() (n_points={total})")

    u = 2.0 * mg_norm - 1.0
    H_traj = m1.build_field_trajectory(u, args.Hc, args.Hrange)
    print(f"[入力] 磁場軌道: {H_traj.min():.2f} - {H_traj.max():.2f} mT "
          f"({total}サイクル = warmup{args.warmup} + 評価{args.n_cycles})")
    return u, mg_norm, H_traj


def run_one_config(name, H_traj, u, mg_norm, args,
                    m1_overrides=None, m2_overrides=None):
    """
    1つのアブレーション設定でパイプライン全体 (Module1 -> Module2 -> 評価) を実行する。

    m1_overrides : dict | None   Module1Params の校正前 base に適用する上書き
    m2_overrides : dict | None   Module2Params に適用する上書き (トグル類)
    """
    target = m1.TargetDynamics(n_onset=args.n_onset)
    base = m1.Module1Params(**(m1_overrides or {}))
    p1 = m1.calibrate_module1(H_traj, target, base=base, verbose=False)
    r1 = m1.simulate_module1(H_traj, p1)

    p2 = m2.Module2Params(**(m2_overrides or {}))
    f_axis, N_idx, H_arr, S11 = m2.run_module2(r1, p2)

    total = args.warmup + args.n_cycles
    sl = slice(args.warmup, total)
    S_e = S11[sl]
    u_e = u[sl]
    tgt_e = mg_norm[sl]

    met = evaluate_fast(S_e, u_e, tgt_e, sample_rate=args.sample_rate)
    met.pop("mc_curve", None)
    met["name"] = name
    met["ratio_vs_measured"] = met["test_MSE"] / args.measured_mse
    return met


ABLATIONS = [
    ("full",              None, None),
    ("no_tc_fine",        None, {"include_tc_fine": False}),
    ("no_helical_n2",     None, {"include_helical_n2": False}),
    ("no_sk_split",       None, {"split_sk_modes": False}),
    ("no_history_shift",  None, {"enable_history_shift": False}),
    ("no_field_shift",    None, {"enable_field_shift": False}),
    ("no_both_shifts",    None, {"enable_history_shift": False, "enable_field_shift": False}),
    ("no_multidomain",    {"n_domains": 1}, None),
]

LABELS_JA = {
    "full":             "フルモデル",
    "no_tc_fine":       "− TC微細構造(4GHz帯ギザギザ)",
    "no_helical_n2":    "− ヘリカルn=2高調波",
    "no_sk_split":      "− スキルミオンCCW/breathing分割",
    "no_history_shift": "− 履歴由来の周波数シフト",
    "no_field_shift":   "− 磁場由来の周波数シフト",
    "no_both_shifts":   "− 両方の周波数シフト",
    "no_multidomain":   "− 多ドメイン化(単一平均場に戻す)",
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-cycles", type=int, default=500,
                     help="評価に使うサイクル数 (引き継ぎ資料のアブレーション表はn=1000)")
    ap.add_argument("--warmup", type=int, default=900)
    ap.add_argument("--Hc", type=float, default=73.0)
    ap.add_argument("--Hrange", type=float, default=90.0)
    ap.add_argument("--n-onset", type=float, default=140.0)
    ap.add_argument("--sample-rate", type=int, default=16,
                     help="実測パイプラインと揃えるサンプルレート (既定16)")
    ap.add_argument("--measured-mse", type=float, default=3.70e-3,
                     help="実測Testing MSE (Lee et al.、実測比の分母)")
    ap.add_argument("--mg-file", type=str, default=None)
    ap.add_argument("--mg-column", type=int, default=0)
    ap.add_argument("--mg-offset", type=int, default=0)
    ap.add_argument("--out", type=str, default="ablation_results.csv")
    args = ap.parse_args()

    u, mg_norm, H_traj = build_input(args)

    rows = []
    print()
    print(f"{'設定':<34}{'NL':>8}{'MC':>8}{'CP':>8}{'test_MSE':>12}{'実測比':>9}")
    print("-" * 79)
    for name, m1_ov, m2_ov in ABLATIONS:
        met = run_one_config(name, H_traj, u, mg_norm, args,
                              m1_overrides=m1_ov, m2_overrides=m2_ov)
        rows.append(met)
        print(f"{LABELS_JA[name]:<34}{met['NL']:>8.3f}{met['MC']:>8.2f}"
              f"{met['CP']:>8.3f}{met['test_MSE']:>12.3e}{met['ratio_vs_measured']:>9.2f}")

    out_path = Path(args.out)
    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            "name", "label_ja", "n_feat", "NL", "MC", "CP",
            "train_MSE", "test_MSE", "ratio_vs_measured"])
        writer.writeheader()
        for met in rows:
            writer.writerow({
                "name": met["name"],
                "label_ja": LABELS_JA[met["name"]],
                "n_feat": met["n_feat"],
                "NL": met["NL"],
                "MC": met["MC"],
                "CP": met["CP"],
                "train_MSE": met["train_MSE"],
                "test_MSE": met["test_MSE"],
                "ratio_vs_measured": met["ratio_vs_measured"],
            })
    print(f"\n書き出し: {out_path.resolve()}")

    full_mse = rows[0]["test_MSE"]
    print("\n--- 各設定のフルモデル比 (MSEが何倍悪化したか) ---")
    for met in rows[1:]:
        print(f"  {LABELS_JA[met['name']]:<34}: {met['test_MSE'] / full_mse:>6.2f}倍")


if __name__ == "__main__":
    main()