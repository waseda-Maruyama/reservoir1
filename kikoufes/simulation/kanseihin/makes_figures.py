"""
make_figures.py --- ポスター用のデータ図 (Fig.5-9) を生成する。

イメージ図 (Fig.1-3) と流れ図 (Fig.4, LaTeX 内の TikZ) は対象外。
モデルの計算は ablation_experiment.py と同じ経路 (MG 系列 -> 磁場軌道 -> Module1
-> Module2 -> fast_metrics) で行うので、ablation_experiment.py と同じ引数を渡せば
同じデータから描かれる。

使い方 (module1.py / module2.py / fast_metrics.py / ablation_experiment.py と同じフォルダで):
    python make_figures.py --mg-file <mackey_glass_t17.npy> --csv ablation_results.csv

出力 (--out, 既定 ./figures):
    fig5_phase.pdf      相分率と秩序度の時間変化
    fig6_spectrum.pdf   合成スペクトルの成分内訳
    fig7_results.pdf    (a) 4 指標の比較 (本モデル / 実測 / 論文 [1] / リザーバーなし)
                        (b) 10 ステップ先予測の時系列
    fig8_ablation.pdf   要素除去: MSE のフルモデル比
    fig9_channels.pdf   (a) TC 微細構造の除去  (b) チャネル間相関の累積分布
    (同名の .png はプレビュー用)

図中の文字は日本語。フォントは原ノ味ゴシック (TeX Live 同梱, ポスター本文と同じ) を
優先し、見つからなければ OS の日本語ゴシック体を使う。
図のサイズはポスター上の枠 (mm) に合わせてあり、文字は印刷時 12-14 pt 相当。
"""
import argparse
import csv
import shutil
import subprocess
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from sklearn.linear_model import Ridge

import module1 as m1
import module2 as m2
from fast_metrics import preprocess
from ablation import build_input

# ---------------------------------------------------------------------------
# 比較用の値
# ---------------------------------------------------------------------------
# 実測: PRCpy 同梱の実測データ ([1] の測定) を本研究と同一の手順で評価した値
MEASURED = {"test_MSE": 3.70e-3, "MC": 4.94, "NL": 0.550, "CP": 1.77}
# 論文 [1] のスキルミオン相 (4 K) の値。MSE は本文 (Hc = 60 mT)、
# MC・NL は Fig. 4i のスキルミオン相 (Hc = 40-115 mT) から読み取った範囲。
# CP は算出に用いる行列の大きさが異なるため比較しない。
PAPER = {"test_MSE": (3.7e-3, 3.7e-3), "MC": (4.6, 6.4), "NL": (0.45, 0.58)}
# 論文 [1] 本文のリザーバーなし (入力に直接リッジ回帰) の MSE。
# 本文の表記は 6.2x10^2 だが、スキルミオン相の約 18 倍 (コニカル相) が
# リザーバーなしと同程度という記述から 6.2x10^-2 と解釈する。
PAPER_NORES_MSE = 6.2e-2

# ---------------------------------------------------------------------------
# 配色 (相ごとに固定。validate_palette.js で 3 色全ペア合格, light)
# ---------------------------------------------------------------------------
C_HE = "#2a78d6"     # ヘリカル
C_TC = "#eb6834"     # tilted conical
C_SK = "#1baf7a"     # スキルミオン
C_MODEL = "#2a78d6"  # 本モデル
C_REF = "#0b0b0b"    # 実測
C_PAPER = "#a3a29c"  # 論文 [1]
C_HI = "#eb6834"     # 要素除去で寄与が大きいもの
C_LO = "#a3a29c"     # 寄与が小さいもの
TXT = "#0b0b0b"
TXT2 = "#52514e"
GRID = "#e6e5e0"
BAND = "#f0efeb"
MM = 1 / 25.4        # mm -> inch


# ---------------------------------------------------------------------------
# フォント
# ---------------------------------------------------------------------------
def find_japanese_font():
    """原ノ味ゴシック (TeX Live) を優先し、なければ OS の日本語フォント名を返す。"""
    if shutil.which("kpsewhich"):
        try:
            p = subprocess.run(["kpsewhich", "HaranoAjiGothic-Regular.otf"],
                               capture_output=True, text=True, timeout=10).stdout.strip()
            if p:
                font_manager.fontManager.addfont(p)
                pb = subprocess.run(["kpsewhich", "HaranoAjiGothic-Bold.otf"],
                                    capture_output=True, text=True, timeout=10).stdout.strip()
                if pb:
                    font_manager.fontManager.addfont(pb)
                return font_manager.FontProperties(fname=p).get_name()
        except Exception:  # noqa: BLE001
            pass
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Noto Sans CJK JP", "Hiragino Sans", "Hiragino Kaku Gothic ProN",
                 "Yu Gothic", "Meiryo", "IPAexGothic", "IPAGothic"):
        if name in installed:
            return name
    print("警告: 日本語フォントが見つかりません。文字化けする場合はフォント名を指定してください。")
    return "DejaVu Sans"


def setup_style():
    jp = find_japanese_font()
    print(f"[フォント] {jp}")
    plt.rcParams.update({
        "font.family": jp, "mathtext.fontset": "dejavusans",
        "axes.unicode_minus": False,
        "font.size": 12, "axes.labelsize": 13, "axes.titlesize": 13,
        "xtick.labelsize": 11.5, "ytick.labelsize": 11.5, "legend.fontsize": 11.5,
        "axes.edgecolor": TXT2, "axes.labelcolor": TXT, "xtick.color": TXT2,
        "ytick.color": TXT2, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7,
        "axes.axisbelow": True, "lines.linewidth": 2.0,
        "legend.frameon": False, "savefig.dpi": 300, "pdf.fonttype": 3,
    })


def save(fig, out, name):
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out / f"{name}.png", bbox_inches="tight", pad_inches=0.02, dpi=200)
    plt.close(fig)
    print(f"  -> {out / name}.pdf")


# ---------------------------------------------------------------------------
# モデル計算
# ---------------------------------------------------------------------------
def run_model(H_traj, args, m1_overrides=None, m2_overrides=None):
    target = m1.TargetDynamics(n_onset=args.n_onset)
    base = m1.Module1Params(**(m1_overrides or {}))
    p1 = m1.calibrate_module1(H_traj, target, base=base, verbose=False)
    r1 = m1.simulate_module1(H_traj, p1)
    p2 = m2.Module2Params(**(m2_overrides or {}))
    return r1, p2


def _z(arr):
    """module2.run_module2 内の z-score 化と同一。"""
    a = np.asarray(arr, dtype=float)
    tail = a[len(a) // 4:] if len(a) > 8 else a
    mu, sd = float(np.mean(tail)), float(np.std(tail))
    return (a - mu) / sd if sd > 1e-12 else np.zeros_like(a)


def synth(r1, p2, N_list, phases=("He", "TC", "Sk")):
    """run_module2 と同じ入力で、指定データ番号のスペクトルを合成する。
    phases: 含める相 (含めない相の分率を 0 にして成分を分離する)"""
    dom_TC, dom_Sk, dom_psi = r1["dom_TC_obs"], r1["dom_Sk_obs"], r1["dom_psi_obs"]
    n_dom = dom_TC.shape[1]
    f = np.linspace(p2.f_min, p2.f_max, p2.n_freq)
    forest = m2.build_mode_forest(n_modes=p2.n_modes, seed=p2.forest_seed, n_domains=n_dom)
    zT, zS, zP = _z(r1["phi_TC_obs"]), _z(r1["phi_Sk_obs"]), _z(r1["psi_obs"])
    zdT = np.column_stack([_z(dom_TC[:, j]) for j in range(n_dom)])
    zdS = np.column_stack([_z(dom_Sk[:, j]) for j in range(n_dom)])
    zdP = np.column_stack([_z(dom_psi[:, j]) for j in range(n_dom)])
    out = np.empty((len(N_list), len(f)))
    for i, N in enumerate(N_list):
        he = float(r1["phi_He_obs"][N]) if "He" in phases else 0.0
        tc = float(r1["phi_TC_obs"][N]) if "TC" in phases else 0.0
        sk = float(r1["phi_Sk_obs"][N]) if "Sk" in phases else 0.0
        dtc = dom_TC[N] if "TC" in phases else np.zeros(n_dom)
        out[i] = m2.synthesize_S11(f, float(r1["H_obs"][N]), he, tc, sk,
                                   float(r1["psi_obs"][N]),
                                   z=(zT[N], zS[N], zP[N]), params=p2, forest=forest,
                                   dom_z=(zdT[N], zdS[N], zdP[N]), dom_phi_TC=dtc)
    return f, out


# ---------------------------------------------------------------------------
# Fig.5 相分率と秩序度
# ---------------------------------------------------------------------------
def fig5(r1, args, out):
    he, tc, sk, psi = (r1[k] for k in ("phi_He_obs", "phi_TC_obs", "phi_Sk_obs", "psi_obs"))
    N = np.arange(len(sk))
    above = np.where(sk >= 0.25)[0]
    n_on = int(above[0]) if len(above) else None

    fig, ax = plt.subplots(figsize=(140 * MM, 92 * MM))
    ax.axvspan(args.warmup, len(sk) - 1, color=BAND, zorder=0, lw=0)
    ax.text((args.warmup + len(sk)) / 2, 0.97, "評価区間", color=TXT2,
            fontsize=11, ha="center", va="top")
    for y, c, lab in ((he, C_HE, r"$\phi_\mathrm{He}$"), (tc, C_TC, r"$\phi_\mathrm{TC}$"),
                      (sk, C_SK, r"$\phi_\mathrm{Sk}$")):
        ax.plot(N, y, color=c, lw=2)
        ax.text(N[-1] * 1.015, y[-1], lab, color=TXT, va="center", fontsize=13)
    ax.plot(N, psi, color=TXT2, lw=1.5, ls=(0, (4, 2)))
    ax.text(N[-1] * 1.015, psi[-1], r"$\psi$", color=TXT2, va="center", fontsize=13)
    if n_on is not None:
        ax.axvline(n_on, color=TXT2, lw=0.9, ls=":")
        ax.text(n_on + 18, 0.97, f"N = {n_on}", color=TXT2, fontsize=11, va="top")
    ax.set_xlim(0, len(sk) - 1); ax.set_ylim(0, 1.0)
    ax.set_xlabel("入力データ番号  $N$")
    ax.set_ylabel("相分率 $\\phi$ ・ 秩序度 $\\psi$")
    save(fig, out, "fig5_phase")


# ---------------------------------------------------------------------------
# Fig.6 スペクトル成分
# ---------------------------------------------------------------------------
def fig6(r1, p2, out, N_show):
    from dataclasses import replace
    f, tot = synth(r1, p2, [N_show])
    _, sk = synth(r1, p2, [N_show], phases=("Sk",))
    _, tc = synth(r1, p2, [N_show], phases=("TC",))
    _, he1 = synth(r1, replace(p2, include_helical_n2=False), [N_show], phases=("He",))
    _, he = synth(r1, p2, [N_show], phases=("He",))
    he2 = he - he1

    fig, ax = plt.subplots(figsize=(140 * MM, 92 * MM))
    off = 1.15 * abs(tot.min())
    rows = [(tot[0], TXT, "合計", 0, "-"),
            (tc[0], C_TC, "TC 微細構造", -1, "-"),
            (sk[0], C_SK, "スキルミオン", -2, "-"),
            (he1[0], C_HE, "ヘリカル n=1", -3, "-"),
            (he2[0], C_HE, "ヘリカル n=2", -3.55, (0, (3, 1.5)))]
    for y, c, lab, k, ls in rows:
        ax.plot(f, y + k * off, color=c, lw=1.8 if lab == "合計" else 1.6, ls=ls)
        ax.text(6.08, k * off, lab, color=TXT, fontsize=11.5, va="center")
    ax.set_xlim(1, 6); ax.set_yticks([])
    ax.set_xlabel("周波数  $f$  (GHz)")
    ax.set_ylabel(r"$\Delta S_{11}$（縦にずらして表示）")
    ax.spines["left"].set_visible(False)
    ax.grid(axis="y", visible=False)
    save(fig, out, "fig6_spectrum")


# ---------------------------------------------------------------------------
# Fig.7 実測・論文との比較と予測
# ---------------------------------------------------------------------------
def ridge_predict(X, target, tau=10, test_size=0.25, alpha=0.1):
    """fast_metrics.evaluate_fast と同じ切り出しで tau ステップ先を予測する。"""
    Xr, yr = X[:-tau], target[tau:]
    n_tr = int(len(yr) * (1 - test_size))
    mdl = Ridge(alpha=alpha).fit(Xr[:n_tr], yr[:n_tr])
    pred = mdl.predict(Xr[n_tr:])
    return yr[n_tr:], pred, float(np.mean((pred - yr[n_tr:]) ** 2))


def fig7(full_row, nores_mse, y_true, y_pred, y_nores, out):
    fig = plt.figure(figsize=(372 * MM, 88 * MM))
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1, 1, 3.3], wspace=0.75)

    # (a) 指標ごとの小さな図。本モデル・実測・論文 [1] (範囲は縦棒)・入力のみを横に並べる
    specs = [("test_MSE", r"MSE ($\times10^{-3}$)", 1e3),
             ("MC", "MC", 1), ("NL", "NL", 1), ("CP", "CP", 1)]
    handles = {}
    for i, (key, title, scale) in enumerate(specs):
        ax = fig.add_subplot(gs[0, i])
        handles["model"] = ax.scatter([0], [full_row[key] * scale], s=75, color=C_MODEL, zorder=4)
        handles["meas"] = ax.scatter([1], [MEASURED[key] * scale], s=75, color=C_REF, zorder=4)
        if key in PAPER:
            lo, hi = PAPER[key]
            if hi > lo:
                ax.plot([2, 2], [lo * scale, hi * scale], color=C_PAPER, lw=7,
                        solid_capstyle="round", zorder=3)
            else:
                ax.scatter([2], [lo * scale], s=75, color=C_PAPER, zorder=4)
        if key == "test_MSE":
            handles["nores"] = ax.scatter([3], [nores_mse * scale], s=70, marker="D",
                                          facecolor="white", edgecolor=TXT2, lw=1.6, zorder=4)
            handles["nores_p"] = ax.scatter([3.35], [PAPER_NORES_MSE * scale], s=75,
                                            color=C_PAPER, zorder=4)
            ax.set_yscale("log"); ax.set_ylim(1, 200)
            ax.set_yticks([1, 10, 100]); ax.set_yticklabels(["1", "10", "100"])
            ax.set_xlim(-0.6, 3.9)
        else:
            ax.set_xlim(-0.6, 2.6)
            vals = [full_row[key], MEASURED[key]] + list(PAPER.get(key, ()))
            lo_, hi_ = min(vals), max(vals)
            pad = (hi_ - lo_) * 0.35 + 1e-9
            ax.set_ylim(lo_ - pad, hi_ + pad)
        ax.set_xticks([])
        ax.spines["bottom"].set_visible(False)
        ax.set_title(title, fontsize=12.5)
        ax.grid(axis="x", visible=False)
    from matplotlib.lines import Line2D
    paper_h = Line2D([0], [0], color=C_PAPER, lw=7, solid_capstyle="round")
    fig.legend([handles["model"], handles["meas"], paper_h, handles["nores"], handles["nores_p"]],
               ["本モデル", "実測", "論文 [1] の範囲", "入力のみ（本研究の手順）", "入力のみ（[1]）"],
               loc="lower left", bbox_to_anchor=(0.045, 0.935), ncol=5, fontsize=10.5,
               handletextpad=0.3, columnspacing=1.0)

    # (b) 予測の時系列
    b = fig.add_subplot(gs[0, 4])
    t = np.arange(len(y_true))
    b.plot(t, y_true, color=TXT, lw=1.8, label="正解")
    b.plot(t, y_pred, color=C_MODEL, lw=1.8, ls=(0, (4, 1.5)), label="本モデル")
    b.plot(t, y_nores, color=C_PAPER, lw=1.6, label="入力のみ")
    b.set_xlabel("テスト区間のデータ番号"); b.set_ylabel("規格化した値")
    b.set_xlim(0, len(t) - 1)
    b.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=3, fontsize=10.5)
    b.text(-0.13, 1.10, "(b)", transform=b.transAxes, fontsize=13, va="bottom")
    fig.text(0.02, 0.965, "(a)", fontsize=13, va="bottom")
    save(fig, out, "fig7_results")


# ---------------------------------------------------------------------------
# Fig.8 要素除去の棒グラフ
# ---------------------------------------------------------------------------
LABELS = {
    "no_multidomain": "単一ドメイン化",
    "no_tc_fine": "TC 微細構造の除去",
    "no_both_shifts": "周波数シフトの除去（両方）",
    "no_field_shift": "磁場による周波数シフトの除去",
    "no_history_shift": "履歴による周波数シフトの除去",
    "no_sk_split": "Sk の 2 モードの統合",
    "no_helical_n2": "ヘリカル n=2 の除去",
}


def read_csv(path):
    rows = {}
    with open(path, encoding="Shift_JIS") as fh:
        for r in csv.DictReader(fh):
            rows[r["name"]] = {k: (float(v) if k not in ("name", "label_ja") else v)
                               for k, v in r.items()}
    return rows


def fig8(rows, out):
    full = rows["full"]
    items = [(k, rows[k]["test_MSE"] / full["test_MSE"]) for k in LABELS if k in rows]
    items.sort(key=lambda kv: kv[1])
    fig, ax = plt.subplots(figsize=(183 * MM, 100 * MM))
    y = np.arange(len(items))
    for yi, (k, r) in zip(y, items):
        ax.barh(yi, r, height=0.62, color=C_HI if r >= 1.5 else C_LO)
        ax.text(r * 1.06, yi, f"×{r:.2f}" if r < 2 else f"×{r:.1f}",
                va="center", fontsize=11, color=TXT)
        ax.text(1.02, yi, f"MC {rows[k]['MC']:.2f}   CP {rows[k]['CP']:.2f}",
                transform=ax.get_yaxis_transform(), va="center", fontsize=10, color=TXT2)
    ax.text(1.02, len(items) - 0.35, f"（フルモデル MC {full['MC']:.2f}  CP {full['CP']:.2f}）",
            transform=ax.get_yaxis_transform(), va="bottom", fontsize=9.5, color=TXT2)
    ax.set_xscale("log"); ax.set_xlim(0.8, 25)
    ax.axvline(1, color=TXT, lw=1.1)
    ax.set_yticks(y); ax.set_yticklabels([LABELS[k] for k, _ in items])
    ax.set_xticks([1, 2, 5, 10, 20]); ax.set_xticklabels(["1", "2", "5", "10", "20"])
    ax.set_xlabel("MSE のフルモデル比")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    fig.subplots_adjust(right=0.70)
    save(fig, out, "fig8_ablation")


# ---------------------------------------------------------------------------
# Fig.9 TC 微細構造の除去とチャネル間相関
# ---------------------------------------------------------------------------
C9 = {"full": TXT, "no_tc_fine": C_TC, "no_multidomain": C_HE, "no_both_shifts": C_SK}
L9 = {"full": "フルモデル", "no_tc_fine": "TC 微細構造なし",
      "no_multidomain": "単一ドメイン", "no_both_shifts": "周波数シフトなし"}


def channel_corr(models, key, sl, band=(3.4, 5.3), step=16):
    """評価区間の読み出し (前処理後) について、band 内チャネル対の |相関係数| を返す。
    相関係数は 2 つのチャネルの評価区間での時系列どうしの相関。"""
    r1, p2 = models[key]
    f, _, _, S = m2.run_module2(r1, p2)
    X = preprocess(S[sl], sample_rate=1)
    m = (f >= band[0]) & (f <= band[1])
    C = np.corrcoef(X[:, m][:, ::step].T)
    return np.abs(C[np.triu_indices_from(C, 1)])


def fig9(models, args, out, N_show, band=(3.4, 5.3)):
    fig, (a, b) = plt.subplots(1, 2, figsize=(183 * MM, 100 * MM),
                               gridspec_kw={"wspace": 0.32})
    r1, p2 = models["full"]
    f, full = synth(r1, p2, [N_show])
    r1n, p2n = models["no_tc_fine"]
    _, notc = synth(r1n, p2n, [N_show])
    a.axvspan(*band, color=BAND, lw=0, zorder=0)
    a.plot(f, full[0], color=C9["full"], lw=1.5)
    a.plot(f, notc[0], color=C9["no_tc_fine"], lw=1.7)
    a.set_xlim(1, 6); a.set_yticks([])
    a.set_xlabel("周波数  $f$  (GHz)"); a.set_ylabel(r"$\Delta S_{11}$")
    ymin = full[0].min()
    for yy, key in ((0.80, "full"), (0.92, "no_tc_fine")):
        a.plot([1.15, 1.45], [ymin * yy] * 2, color=C9[key], lw=1.7)
        a.text(1.52, ymin * yy, L9[key], va="center", fontsize=10.5, color=TXT)
    a.set_title("(a) スペクトル", loc="left", fontsize=12)

    sl = slice(args.warmup, args.warmup + args.n_cycles)
    meds = {}
    for key in ("full", "no_multidomain", "no_both_shifts", "no_tc_fine"):
        c = np.sort(channel_corr(models, key, sl, band))
        meds[key] = float(np.median(c))
        b.plot(c, np.arange(1, len(c) + 1) / len(c), color=C9[key], lw=2)
        b.scatter([meds[key]], [0.5], s=28, color=C9[key], zorder=3)
    b.axhline(0.5, color=GRID, lw=1)
    b.set_xlim(0, 1.02); b.set_ylim(0, 1)
    b.set_xlabel("チャネル対の相関係数（絶対値）")
    b.set_ylabel("累積割合")
    ys = {"full": 0.94, "no_multidomain": 0.86, "no_both_shifts": 0.78, "no_tc_fine": 0.70}
    for key, yv in ys.items():
        b.plot([0.02, 0.08], [yv, yv], color=C9[key], lw=2)
        b.text(0.10, yv, f"{L9[key]}（{meds[key]:.2f}）", va="center", fontsize=10, color=TXT)
    b.text(0.02, 0.62, "（ ）内は中央値", fontsize=9.5, color=TXT2, va="center")
    b.set_title("(b) チャネル間の相関（3.4–5.3 GHz）", loc="left", fontsize=12)
    save(fig, out, "fig9_channels")
    return meds


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-cycles", type=int, default=500)
    ap.add_argument("--warmup", type=int, default=900)
    ap.add_argument("--Hc", type=float, default=73.0)
    ap.add_argument("--Hrange", type=float, default=90.0)
    ap.add_argument("--n-onset", type=float, default=140.0)
    ap.add_argument("--mg-file", type=str, default=None)
    ap.add_argument("--mg-column", type=int, default=0)
    ap.add_argument("--mg-offset", type=int, default=0)
    ap.add_argument("--csv", type=str, default="ablation_results.csv",
                    help="ablation_experiment.py の出力 (Fig.7a, Fig.8 に使用)")
    ap.add_argument("--show-index", type=int, default=50,
                    help="Fig.6/9a で表示する評価区間内のデータ番号")
    ap.add_argument("--test-size", type=float, default=0.25,
                    help="予測の試験割合 (evaluate_fast の既定と同じ)")
    ap.add_argument("--out", type=str, default="figures")
    args = ap.parse_args()

    setup_style()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    u, mg_norm, H_traj = build_input(args)
    N_show = args.warmup + args.show_index
    sl = slice(args.warmup, args.warmup + args.n_cycles)

    print("[モデル計算]")
    models = {
        "full": run_model(H_traj, args),
        "no_tc_fine": run_model(H_traj, args, m2_overrides={"include_tc_fine": False}),
        "no_multidomain": run_model(H_traj, args, m1_overrides={"n_domains": 1}),
        "no_both_shifts": run_model(H_traj, args, m2_overrides={
            "enable_history_shift": False, "enable_field_shift": False}),
    }
    r1, p2 = models["full"]

    print("[作図]")
    fig5(r1, args, out)
    fig6(r1, p2, out, N_show)

    rows = read_csv(args.csv)
    _, _, _, S = m2.run_module2(r1, p2)
    X = preprocess(S[sl], sample_rate=16)
    y_true, y_pred, _ = ridge_predict(X, mg_norm[sl], test_size=args.test_size)
    # リザーバーなし: 入力 u(N) に直接リッジ回帰 ([1] Fig. 1e の灰色線と同じ比較)
    _, y_nores, nores_mse = ridge_predict(u[sl][:, None], mg_norm[sl], test_size=args.test_size)
    fig7(rows["full"], nores_mse, y_true, y_pred, y_nores, out)

    fig8(rows, out)
    meds = fig9(models, args, out, N_show)

    print("\n[ポスター本文の数値と照合してください]")
    print(f"  リザーバーなし (入力に直接リッジ回帰) の MSE: {nores_mse:.3e}")
    print("  チャネル間 |相関| の中央値 (3.4-5.3 GHz): "
          + ", ".join(f"{L9[k]} {v:.2f}" for k, v in meds.items()))


if __name__ == "__main__":
    main()