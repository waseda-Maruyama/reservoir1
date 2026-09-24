"""
reservoir_metrics.py

PRCpy本体には実装されていない CP (Complexity) 指標を計算するためのモジュール。
NL / MC は prcpy.RC.Pipeline_RC.Pipeline の get_non_linearity() / get_linear_memory_capacity()
で計算できるが、CPに対応するメソッドは無いため、原論文(2209.06962 Supplementary S6)の
定義に基づき独自に実装する。

--- 原論文 S6節 CPの定義（要約） ---
- 入力信号は使わず、リザーバーの読み出し値（スペクトル）のみを使用する。
- 読み出し値を2つの正方行列に整形する（原論文では各480x480）。
- それぞれの行列について「有効ランク」を計算する：
    1. SVD（特異値分解）で特異値 s_1, s_2, ... を得る
    2. 正規化して確率分布とみなす: p_i = s_i / sum(s_i)
    3. シャノンエントロピー H = -sum(p_i * ln(p_i)) を計算（自然対数、nats単位）
    4. 有効ランク = exp(H)   （Roy & Vetterli, EUSIPCO 2007の定義）
- 2つの行列の有効ランクの平均値がCPスコア。

原論文の"480x480"という具体的なサイズは、その実験でのサイクル数・チャンネル選定に
依存した値であり、本プロジェクトのデータサイズには直接当てはまらない。
そのためここでは「サンプル（サイクル）方向を前半・後半の2つに分割し、それぞれを
（行数・列数の小さい方に合わせて）正方行列に切り詰める」という一般化した手順を実装する。
分割方法は split引数で "half"（前半/後半）と "odd_even"（奇数番目/偶数番目）から選べる。
"""
import numpy as np


def effective_rank(X):
    """
    Roy & Vetterli (2007) の有効ランク（effective rank）を計算する。

    X: (n, m) の行列（正方でなくても計算は可能だが、CP計算では正方行列を渡す想定）

    手順：
      1. SVDで特異値 s を得る
      2. p_i = s_i / sum(s_i) として正規化
      3. シャノンエントロピー H = -sum(p_i * ln(p_i)) （自然対数）
      4. 有効ランク = exp(H)

    返り値は [1, min(X.shape)] の範囲に収まる（全特異値が均等なら次元数と一致、
    1つの特異値に集中していれば1に近づく）。
    """
    X = np.asarray(X, dtype=float)
    s = np.linalg.svd(X, compute_uv=False)

    s = s[s > 1e-12]  # ゼロ（数値誤差含む）特異値は除外
    if s.size == 0:
        return 0.0

    p = s / s.sum()
    H = -np.sum(p * np.log(p))  # 自然対数（nats）。PRCpyのshannon_entropyと同じ流儀。
    return float(np.exp(H))


def _to_square(M):
    """行数・列数のうち小さい方に合わせて正方行列に切り詰める。"""
    r, c = M.shape
    d = min(r, c)
    return M[:d, :d]


def get_readout_matrix(rc_pipeline):
    """
    PRCpyのPipelineオブジェクトから、target列を除いた
    readout（リザーバー出力）行列を (n_scans, n_features) の形で取り出す。
    """
    df = rc_pipeline.get_rc_df()
    cols = [c for c in df.columns if c != "target"]
    return df[cols].to_numpy(dtype=float)


def select_top_range_channels(X, top_frac=0.3):
    """
    原論文S6節の前処理：入力に反応していない（＝どのサイクルでもほぼ値が変わらない）
    「死んだ」周波数チャンネルを除外するため、各チャンネル（列）のレンジ
    (max - min) が大きい上位 top_frac の列だけを残す。

    X: (n_scans, n_features)
    top_frac: 残す列の割合（デフォルト0.3 = 上位30%、原論文と同じ）

    NL/MC/CPいずれの計算にも使える前処理だが、現状のパイプラインでは未適用。
    必要ならCP計算の前、あるいはNL/MC計算の前にも同様に適用できる。
    """
    X = np.asarray(X, dtype=float)
    ranges = X.max(axis=0) - X.min(axis=0)
    n_keep = max(1, int(np.ceil(X.shape[1] * top_frac)))
    top_idx = np.argsort(ranges)[::-1][:n_keep]
    top_idx.sort()  # 元の周波数順を保つ
    return X[:, top_idx]


def get_complexity(readout_matrix, split="half"):
    """
    CP（複雑性）スコアを計算する。

    readout_matrix: (n_scans, n_features) のリザーバー出力行列（targetは含めない）。
                     get_readout_matrix(rc_pipeline) で取得したものを渡す想定。
    split: "half"     -> 前半サイクル / 後半サイクルの2グループに分割
           "odd_even" -> 奇数番目 / 偶数番目のサイクルの2グループに分割

    戻り値: (CP, (er1, er2))
        CP  : 2つの有効ランクの平均値
        er1, er2 : それぞれのグループの有効ランク（内訳を見たい場合用）
    """
    X = np.asarray(readout_matrix, dtype=float)
    n_scans = X.shape[0]

    if split == "half":
        half = n_scans // 2
        X1, X2 = X[:half], X[half:half * 2]
    elif split == "odd_even":
        X1, X2 = X[0::2], X[1::2]
        m = min(len(X1), len(X2))
        X1, X2 = X1[:m], X2[:m]
    else:
        raise ValueError(f"Unknown split mode: {split!r}. Use 'half' or 'odd_even'.")

    X1_sq = _to_square(X1)
    X2_sq = _to_square(X2)

    er1 = effective_rank(X1_sq)
    er2 = effective_rank(X2_sq)

    cp = 0.5 * (er1 + er2)
    return cp, (er1, er2)


if __name__ == "__main__":
    # --- 簡単な自己チェック ---
    rng = np.random.default_rng(0)

    # 1. ランク1に近い行列（全チャンネルがほぼ同じ情報）-> 有効ランクは1に近いはず
    base = rng.standard_normal(200)
    X_low_rank = np.outer(base, rng.standard_normal(200)) + 1e-3 * rng.standard_normal((200, 200))
    er_low = effective_rank(X_low_rank)
    print(f"[sanity] near-rank-1 matrix: effective_rank = {er_low:.3f} (expect close to 1)")

    # 2. フルランクのランダム行列 -> 有効ランクは次元数に近いはず
    X_full_rank = rng.standard_normal((200, 200))
    er_full = effective_rank(X_full_rank)
    print(f"[sanity] random full-rank matrix: effective_rank = {er_full:.3f} (expect close to 200)")

    # 3. get_complexity の動作確認（ダミーのreadout行列で）
    dummy_readout = rng.standard_normal((500, 101))  # 500サイクル x 101特徴量（sample_rate=16相当）
    cp, (er1, er2) = get_complexity(dummy_readout, split="half")
    print(f"[sanity] dummy readout (500x101): CP = {cp:.3f}  (er1={er1:.3f}, er2={er2:.3f})")