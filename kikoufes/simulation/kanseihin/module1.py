"""
module1.py --- 磁気相ダイナミクス（Module 1）

Cu2OSeO3 の mapped field-cycling (MFC) 下での相分率ダイナミクスを、
4つの状態変数の常微分方程式系として記述する。

    phi_He : ヘリカル相分率
    phi_TC : tilted conical (核生成中間状態) 相分率
    phi_Sk : 低温スキルミオン相 (LTS) 分率
    psi    : スキルミオン格子のアニーリング度 (長期秩序化パラメータ, [0,1])

============================================================================
設計の根拠と、旧版からの変更点
============================================================================

【1】 2値ゲート in_sky の撤廃 -> 連続窓関数 S(H)

旧版は in_sky = 1[H_min <= H <= H_max] という2値ゲートで全ての遷移レートを
制御していた。しかし実際の実験条件 (Hc=73 mT, Hrange=90 mT -> H は 28-118 mT
を動き、準安定窓 25-120 mT に完全に内包される) では in_sky が全区間で 1 に
固定される。この場合レートが定数となり、系は H(N) の実際の値に依存しない
自律系に退化し、単一の固定点へ収束するだけで履歴依存性が原理的に生じない。

これは Lee論文 (arXiv:2209.06962) 本文の
  "The best forecasting performance is found when the field-cycling lies
   entirely inside the skyrmion phase at lower temperatures."
(最良の予測性能は、フィールドサイクリングが完全にスキルミオン相の内側に
 収まるときに得られる) という記述と真っ向から矛盾する。

そこで S(H) = sigmoid((H-H_min)/w) * sigmoid((H_max-H)/w) という滑らかな
連続関数に置き換えた。w -> 0 の極限で旧来の2値ゲートに一致する一般化であり、
窓の内側にいる間も H(N) の実際の値が連続的に力学へ伝わる。
物理的には「準安定領域の中心に近いほど自由エネルギー的に安定＝核生成に
有利、境界に近いほど隣接相へ緩和しやすい」という相図の一般論に対応する。

【2】 TC->Sk 成長則: 根拠のない自己触媒項 -> Avrami (JMAK) 界面成長

旧版の gamma_0 * phi_TC^m (m=3) は TC 単独の累乗に比例する形だった。これには
構造的欠陥があり、TC->He (定数レート) と TC->Sk (phi_TC の3乗、低密度では
極端に弱い) が競合するため、phi_TC がある閾値密度に達するまで TC->He が常に
優勢となり、He->TC 供給をいくら強めても phi_Sk が 0.3-0.4 で頭打ちになる
(フラックス収支計算で確認済み)。

新式は Avrami (JMAK) 核生成成長理論に基づく:
    J(TC->Sk) = (k_nuc + k_growth * phi_TC * phi_Sk) * S(H)
  - k_nuc : 自発核生成 (phi_Sk=0 からの立ち上がりの種)
  - k_growth * phi_TC * phi_Sk : 界面成長。既存の Sk "種" が隣接する TC を
    侵食する速度に比例
Malsch et al. (arXiv:2509.00517, MFM 実空間観察, H||<100>) の
  "the skyrmion lattice grows on expense of the tilted conical state"
(スキルミオン格子は tilted conical 状態を消費して成長する) に対応する。

【3】 Sk->TC 逆流の導入 (本版での新規追加)

Aqeel et al. (PRL 126, 017202, H||<100>) は、LTS の核生成が tilted conical
状態を必須の中間状態として経由する2段階過程であることを報告している。
その逆過程として、磁場が最適値から外れたとき Sk が TC へ巻き戻る経路
    J(Sk->TC) = k_unwind * phi_Sk * (1 - S(H))
を導入した。これにより定常状態の揺らぎが Sk<->TC 間のトレードオフとして生じる。

この設計は本プロジェクトの観測的要請にも合致する: Sk<->He のトレードオフだと
phi_TC が低い値に収束してしまい、Module 2 で phi_TC に比例させている 4 GHz 帯
の微細構造 (ギザギザ) が消えてしまう。実測スペクトルではギザギザが常時
観測されるため、揺らぎは Sk<->TC 間で起きていると考えるのが整合的である。

【4】 アバランシェ崩壊 Sk->He はオプション扱いに降格

旧版は KTHNY 融解理論に基づく熱活性化崩壊を必須要素としていたが、実測の
グラフ化では一度スキルミオンが形成された後にヘリカル相が優勢に戻る様子が
確認されなかった。そこで nu_0 = 0 (既定) とし、アバランシェ崩壊が起きない
シナリオを標準とする。nu_0 > 0 を与えれば旧来の挙動も再現できる。

【5】 アニーリング変数 psi の新規導入

Malsch et al. Fig.3e は、フィールドサイクル数 n_c = 10, 15, 21, 28, 36, 45,
55, 136, 435 にわたってスキルミオン格子の配向分布が徐々に単一ピークへ凝集
していく (= 多結晶から単結晶へアニーリングされる) 過程を示している。これは
数百サイクルという非常に長い時間スケールで単調に進行する、実測に裏付けられた
長期記憶変数である。

    d(psi)/dt = r_ann * phi_Sk * (1 - psi)

phi_Sk が存在する間だけ進行し 1 に飽和する。Module 2 側でこの psi を共鳴周波数
のシフト・モード強度比の変化に結びつけることで、Lee論文 Fig.S4d が示す
「同一磁場値でもサイクル数によってピーク位置が ±10% 変動する」長期履歴依存性
を表現する。

【6】 多ドメインアンサンブルへの拡張 (本版での最重要変更)

単一の平均場 ODE では状態変数が実質 3 個 (phi_He, phi_TC, phi_Sk のうち
独立なのは 2 個) + psi しかない。線形メモリ容量 MC は「読み出しから過去の
入力を線形復元できる量」なので、情報処理不等式により状態次元数で上限が
決まる。さらに単一平均場では全ての緩和時定数が同程度 (本モデルでは 18-70
サイクル) になり、MC が問う k = 1..8 サイクルの短期記憶を保持できない。
実際、単一平均場版では PRCpy の MC が負の値 (入力自身の自己相関以下) に
なることを確認した。Lee論文が報告する MC = 4-7 には原理的に届かない。

実材料が多ドメイン系であることは複数の文献が明記している:
  - Malsch et al. (MFM 実空間観察): スキルミオン格子は回転ドメインからなる
    「多結晶」であり、FFT に六回対称のリング状の複数ピークが現れる。各
    ドメインのアニーリングは個別に進行する
  - Lee論文 SM S3: ヘリカル相は Q||[100] 方向の "multidomain state"
  - Weiler et al.: n=2, n=3 モードがヘリカル相・コニカル相ともに 2 組observed
    され、"attributed to a multidomain state" と帰属されている

そこで系を n_domains 個のサブドメインの集合として扱う。各ドメイン j は
同じ形の ODE に従うが、局所的な反磁場・異方性・ピン止めのばらつきを反映して
  - 準安定窓の中心が delta_H[j] だけずれる
  - 遷移レートが rate_mult[j] 倍される
という個性を持つ。レート倍率を対数一様に散らすことで時定数が数サイクルから
数十サイクルまで分布し、異なる遅延 k に対する記憶を別々のドメインが担う
ようになる (フィルタバンクとして働く)。

Module 2 側では、4 GHz 帯の微細構造 (ギザギザ) を構成する多数のモードを
これらのドメインに割り当てる。これにより「ギザギザのランダムさの背後に
履歴依存がある」という要請が、多ドメインという物理的実体に裏付けられた形で
実現される。

============================================================================
"""
from dataclasses import dataclass, asdict
import numpy as np


# ---------------------------------------------------------------------------
# 窓関数
# ---------------------------------------------------------------------------
def soft_window(H, h_min, h_max, edge_width):
    """
    スキルミオン準安定窓 [h_min, h_max] の滑らかな連続版指示関数 S(H) in [0,1]。
    中心付近で 1、端に近づくにつれ滑らかに 0 へ減衰し、窓の外でも連続的に 0 へ。
    edge_width -> 0 の極限で 2 値ゲート 1[h_min<=H<=h_max] に一致する。
    """
    z1 = np.clip((np.asarray(H, dtype=float) - h_min) / edge_width, -60.0, 60.0)
    z2 = np.clip((h_max - np.asarray(H, dtype=float)) / edge_width, -60.0, 60.0)
    return (1.0 / (1.0 + np.exp(-z1))) * (1.0 / (1.0 + np.exp(-z2)))


# ---------------------------------------------------------------------------
# パラメータ管理
# ---------------------------------------------------------------------------
# パラメータが増えて見通しが悪くなる問題への対処として、
#   (a) 役割ごとにグループ分けした dataclass で保持し、
#   (b) 各パラメータに「根拠のレベル」をメタデータとして併記し、
#   (c) 生のレートではなく「高レベルの目標値」(onset サイクル・定常相分率) から
#       逆算する calibrate_module1() を用意する
# という 3 段構えにしている。(c) により、恣意的な手当たり次第のパラメータ探索
# ではなく「再現したいダイナミクスを宣言 -> レートは自動で逆算」という
# 再現可能なワークフローになる。
PARAM_PROVENANCE = {
    "H_c2":             ("literature", "コニカル相飽和臨界磁場 (Janson et al. 2014)"),
    "H_min":            ("literature", "準安定窓下限。Lee論文 SM S3: 25 < H < 120 mT"),
    "H_max":            ("literature", "準安定窓上限。同上"),
    "H_edge_width":     ("assumed",    "窓端の遷移の滑らかさ。2値->連続化のための仮定"),
    "Gamma_He_TC":      ("derived",    "He->TC 核生成レート。TC 立ち上がり時定数から逆算"),
    "Gamma_TC_He":      ("derived",    "TC->He 緩和レート。定常 phi_He/phi_TC 比から逆算"),
    "k_nuc":            ("assumed",    "自発核生成レート。onset の起点を与える微小値"),
    "k_growth":         ("derived",    "Avrami 界面成長係数。onset サイクル数から数値校正"),
    "k_unwind":         ("derived",    "Sk->TC 巻き戻しレート。定常 phi_Sk から逆算"),
    "r_ann":            ("derived",    "アニーリング速度。psi の飽和サイクル数から逆算"),
    "nu_0":             ("assumed",    "アバランシェ基底レート。既定 0 (崩壊なしシナリオ)"),
    "delta_E0":         ("assumed",    "アバランシェ活性化障壁基底値"),
    "kappa":            ("assumed",    "密度依存の障壁低下係数"),
    "k_B_T":            ("assumed",    "アバランシェの熱的鋭さ"),
    "edge_decay_boost": ("assumed",    "窓端での崩壊加速。アバランシェ有効時のみ意味を持つ"),
    "dt":               ("assumed",    "数値積分の時間刻み"),
    "steps_per_cycle":  ("measured",   "1サイクルの磁場点数 (Hmid/Hhigh/Hlow の3点)"),
}


@dataclass
class Module1Params:
    """Module 1 のパラメータ。役割ごとにグループ化して保持する。"""

    # --- 磁場・相図に関する量 (文献由来、原則として動かさない) ---
    H_c2: float = 170.0          # [mT] コニカル相飽和臨界磁場
    H_min: float = 25.0          # [mT] スキルミオン準安定窓 下限
    H_max: float = 120.0         # [mT] スキルミオン準安定窓 上限
    H_edge_width: float = 12.0   # [mT] 窓端の遷移の滑らかさ

    # --- 相間遷移レート (校正対象) ---
    Gamma_He_TC: float = 0.030   # He -> TC  (S(H) ゲートあり)
    Gamma_TC_He: float = 0.010   # TC -> He  (ゲートなし)
    k_nuc: float = 2.0e-5        # TC -> Sk  自発核生成項
    k_growth: float = 0.060      # TC -> Sk  Avrami 界面成長係数
    k_unwind: float = 0.090      # Sk -> TC  巻き戻し ((1-S(H)) 比例)

    # --- 長期アニーリング (Malsch Fig.3e) ---
    r_ann: float = 0.004

    # --- アバランシェ崩壊 Sk->He (既定では無効: nu_0 = 0) ---
    nu_0: float = 0.0
    delta_E0: float = 0.5
    kappa: float = 1.5
    k_B_T: float = 0.02
    edge_decay_boost: float = 3.0

    # --- 多ドメインアンサンブル ---
    n_domains: int = 12          # サブドメイン数
    domain_dH: float = 14.0      # [mT] 窓中心のドメイン間ばらつき (標準偏差)
    domain_rate_spread: float = 0.35  # レート倍率の対数10スプレッド (10^±値)
    domain_seed: int = 7           # ドメインの個性を決める乱数種

    # --- 数値積分 ---
    dt: float = 1.0
    steps_per_cycle: int = 3

    def to_dict(self):
        return asdict(self)

    def describe(self):
        """パラメータ一覧を根拠レベル付きで文字列化する。"""
        lines = [f"{'parameter':<18}{'value':>12}  {'provenance':<11} note",
                 "-" * 94]
        for k, v in asdict(self).items():
            prov, note = PARAM_PROVENANCE.get(k, ("-", ""))
            lines.append(f"{k:<18}{float(v):>12.6g}  {prov:<11} {note}")
        return "\n".join(lines)


@dataclass
class TargetDynamics:
    """
    再現したいダイナミクスを「高レベルの目標」として宣言する。
    calibrate_module1() がこれを満たすようレートを逆算する。

    既定値は本プロジェクトで仮定したシナリオ:
      サイクルを重ねると徐々に核が形成され N~140 でスキルミオン相が生まれる。
      その後スキルミオン相は準安定となり、ヘリカル相は小さいまま多少変動する。
      アバランシェ崩壊は起きず、揺らぎは Sk<->TC のトレードオフとして現れる。
    """
    n_onset: float = 140.0        # phi_Sk が onset_level に到達するサイクル数
    onset_level: float = 0.25     # onset の判定閾値
    phi_Sk_ss: float = 0.68       # 定常での phi_Sk
    phi_TC_ss: float = 0.24       # 定常での phi_TC (ギザギザが見える程度に保つ)
    phi_He_ss: float = 0.08       # 定常での phi_He (小さく保つ)
    tc_rise_cycles: float = 20.0  # phi_TC の立ち上がり時定数 [cycle]
    psi_sat_cycles: float = 400.0 # psi が約 63% まで進むサイクル数


# ---------------------------------------------------------------------------
# 本体
# ---------------------------------------------------------------------------
def build_domains(params):
    """
    多ドメインアンサンブルの「個性」を生成する。

    各ドメイン j は
      - 準安定窓の中心が dH[j] だけずれる (局所反磁場・異方性・ピン止めの
        ばらつきを表す)
      - 全遷移レートが rate_mult[j] 倍される (対数一様に散らすことで時定数が
        数サイクルから数十サイクルまで分布し、異なる遅延 k に対する記憶を
        別々のドメインが担うフィルタバンクとして働く)
      - 存在比 (体積分率) weight[j] を持つ
    という個性を持つ。乱数はこの個性を一度だけ決めるために使われ、
    時間発展自体は完全に決定論的である。
    """
    p = params.to_dict() if isinstance(params, Module1Params) else dict(params)
    J = int(p.get("n_domains", 1))
    rng = np.random.default_rng(int(p.get("domain_seed", 7)))

    if J <= 1:
        return {"dH": np.zeros(1), "rate_mult": np.ones(1), "weight": np.ones(1)}

    # 窓中心のずれ: 対称に配置しつつ僅かに乱数で崩す
    base = np.linspace(-1.0, 1.0, J)
    dH = p.get("domain_dH", 10.0) * (base + 0.25 * rng.normal(0, 1, J) / np.sqrt(J))

    # レート倍率: 対数一様。時定数のスプレッドを作る
    spread = p.get("domain_rate_spread", 0.6)
    expo = np.linspace(-spread, spread, J) + 0.12 * rng.normal(0, 1, J)
    rate_mult = 10.0 ** expo

    weight = np.ones(J) / J
    return {"dH": dH, "rate_mult": rate_mult, "weight": weight}


def simulate_module1(H_trajectory, params, return_all=True):
    """
    磁場軌道 H_trajectory (長さ T = steps_per_cycle * n_cycles) を入力として
    多ドメイン相分率ダイナミクスを積分する。

    Returns
    -------
    dict :
      phi_He_obs, phi_TC_obs, phi_Sk_obs, psi_obs : (n_cycles,)  アンサンブル平均
      dom_TC_obs, dom_Sk_obs, dom_psi_obs         : (n_cycles, J) ドメイン別
      mz_obs, H_obs                                : (n_cycles,)
      domains                                      : build_domains() の戻り値
    """
    p = params.to_dict() if isinstance(params, Module1Params) else dict(params)

    T = len(H_trajectory)
    dt = p.get("dt", 1.0)
    spc = int(p.get("steps_per_cycle", 3))

    h_min, h_max = p["H_min"], p["H_max"]
    edge_width = p.get("H_edge_width", 12.0)

    dom = build_domains(p)
    dH, rate_mult, wj = dom["dH"], dom["rate_mult"], dom["weight"]
    J = len(dH)

    G_He_TC = p["Gamma_He_TC"] * rate_mult
    G_TC_He = p["Gamma_TC_He"] * rate_mult
    k_nuc = p["k_nuc"] * rate_mult
    k_growth = p["k_growth"] * rate_mult
    k_unwind = p["k_unwind"] * rate_mult
    r_ann = p.get("r_ann", 0.0) * rate_mult

    nu_0 = p.get("nu_0", 0.0)
    delta_E0 = p.get("delta_E0", 0.5)
    kappa = p.get("kappa", 1.5)
    k_B_T = p.get("k_B_T", 0.02)
    edge_decay_boost = p.get("edge_decay_boost", 0.0)

    H_arr = np.asarray(H_trajectory, dtype=float)
    m_z = np.clip(H_arr / p["H_c2"], 0.0, 1.0)

    # ドメインごとに窓がずれるので S は (T, J)
    S_all = soft_window(H_arr[:, None] - dH[None, :], h_min, h_max, edge_width)

    he = np.ones((T, J)); tc = np.zeros((T, J))
    sk = np.zeros((T, J)); ps = np.zeros((T, J))

    for t in range(T - 1):
        S_t = S_all[t]; U_t = 1.0 - S_t
        a0, b0, c0, d0 = he[t], tc[t], sk[t], ps[t]

        J_He_TC = G_He_TC * a0 * S_t
        J_TC_He = G_TC_He * b0
        J_TC_Sk = (k_nuc + k_growth * b0 * c0) * S_t
        J_Sk_TC = k_unwind * c0 * U_t
        if nu_0 > 0.0:
            E_act = np.maximum(delta_E0 - kappa * c0 * c0, 0.0)
            J_Sk_He = nu_0 * np.exp(-E_act / k_B_T) * c0 * (1.0 + edge_decay_boost * U_t)
        else:
            J_Sk_He = np.zeros(J)

        # 在庫を超えないようにクリップ (負値化防止)
        J_He_TC = np.minimum(J_He_TC, a0 / dt)
        out_tc = J_TC_He + J_TC_Sk
        s = np.where(out_tc * dt > b0, b0 / np.maximum(out_tc * dt, 1e-30), 1.0)
        J_TC_He = J_TC_He * s; J_TC_Sk = J_TC_Sk * s
        out_sk = J_Sk_TC + J_Sk_He
        s = np.where(out_sk * dt > c0, c0 / np.maximum(out_sk * dt, 1e-30), 1.0)
        J_Sk_TC = J_Sk_TC * s; J_Sk_He = J_Sk_He * s

        a = np.clip(a0 + (J_TC_He + J_Sk_He - J_He_TC) * dt, 0.0, 1.0)
        b = np.clip(b0 + (J_He_TC + J_Sk_TC - J_TC_He - J_TC_Sk) * dt, 0.0, 1.0)
        c = np.clip(c0 + (J_TC_Sk - J_Sk_TC - J_Sk_He) * dt, 0.0, 1.0)
        tot = np.maximum(a + b + c, 1e-30)
        he[t + 1], tc[t + 1], sk[t + 1] = a / tot, b / tot, c / tot

        ps[t + 1] = np.clip(d0 + r_ann * sk[t + 1] * (1.0 - d0) * dt, 0.0, 1.0)

    obs = np.arange(spc - 1, T, spc)
    out = {
        "phi_He_obs": he[obs] @ wj,
        "phi_TC_obs": tc[obs] @ wj,
        "phi_Sk_obs": sk[obs] @ wj,
        "psi_obs": ps[obs] @ wj,
        "dom_TC_obs": tc[obs],
        "dom_Sk_obs": sk[obs],
        "dom_psi_obs": ps[obs],
        "dom_He_obs": he[obs],
        "mz_obs": m_z[obs],
        "H_obs": H_arr[obs],
        "domains": dom,
    }
    if return_all:
        out.update({"full_phi_He": he @ wj, "full_phi_TC": tc @ wj,
                    "full_phi_Sk": sk @ wj, "full_psi": ps @ wj, "full_H": H_arr})
    return out


# ---------------------------------------------------------------------------
# 磁場軌道の生成 (MFC スキーム)
# ---------------------------------------------------------------------------
def build_field_trajectory(u_norm, H_c, H_range):
    """
    Lee論文 SM S1 の mapped field-cycling (MFC) スキームに従って磁場軌道を作る。

      "u(t) is normalised between [-1, 1] and offset by a central cycling field
       value Hc, where two additional copies (Hhigh and Hlow) are generated
       above and below Hmid using the cycling width Hrange."
      "the distance between Hlow and Hhigh is the cycling width Hrange/2"

    A = H_range/4 として Hmid = H_c + A*u, Hhigh = Hmid + A, Hlow = Hmid - A。
    Hhigh - Hlow = H_range/2 が常に成立し、u が [-1,1] を振り切る場合の全体の
    振れ幅は H_c ± H_range/2 となる。

    実測データ (PRCpy のファイル名) から復元した磁場範囲 28.00-117.90 mT は
    H_c=73, H_range=90 のとき [28.0, 118.0] となり、ほぼ完全に一致する。

    リザーバ出力は各サイクルの最低磁場点 (Hlow) で読み出される
      "We construct the reservoir outputs using the FMR spectra measured at the
       lowest field point (yellow dots) within the cycles."
    ため、1サイクルを [Hmid, Hhigh, Hlow] の順に並べ Hlow を終点に置く。
    """
    u_norm = np.asarray(u_norm, dtype=float)
    A = H_range / 4.0
    H_mid = H_c + A * u_norm
    n = len(u_norm)
    H_traj = np.empty(n * 3)
    H_traj[0::3] = H_mid
    H_traj[1::3] = H_mid + A
    H_traj[2::3] = H_mid - A
    return H_traj


def generate_mackey_glass(n_points, tau=17, beta=0.2, gamma=0.1, n_exp=10,
                          dt=1.0, discard=2000, x0=1.2):
    """標準的な Mackey-Glass 時系列 (delay=17) を Euler 法で生成する。"""
    hist_len = int(round(tau / dt))
    total = discard + n_points + hist_len + 1
    x = np.zeros(total)
    x[:hist_len] = x0
    for t in range(hist_len, total - 1):
        x_tau = x[t - hist_len]
        x[t + 1] = x[t] + dt * (beta * x_tau / (1.0 + x_tau ** n_exp) - gamma * x[t])
    return x[hist_len + discard:][:n_points]


# ---------------------------------------------------------------------------
# 自動校正
# ---------------------------------------------------------------------------
def measure_dynamics(res, onset_level=0.25, tail_frac=0.6):
    """シミュレーション結果から onset サイクルと定常相分率を測る。"""
    sk = res["phi_Sk_obs"]
    n = len(sk)
    above = np.where(sk >= onset_level)[0]
    n_onset = float(above[0]) if len(above) else float(n * 3)
    tail = slice(int(n * tail_frac), n)
    return {
        "n_onset": n_onset,
        "phi_Sk_ss": float(np.mean(res["phi_Sk_obs"][tail])),
        "phi_TC_ss": float(np.mean(res["phi_TC_obs"][tail])),
        "phi_He_ss": float(np.mean(res["phi_He_obs"][tail])),
        "phi_Sk_std": float(np.std(res["phi_Sk_obs"][tail])),
        "phi_TC_std": float(np.std(res["phi_TC_obs"][tail])),
        "phi_He_std": float(np.std(res["phi_He_obs"][tail])),
        "psi_end": float(res["psi_obs"][-1]),
    }


def calibrate_module1(H_trajectory, target=None, base=None, verbose=True):
    """
    「再現したいダイナミクス」(TargetDynamics) から Module1 のレートを逆算する。

    手順:
      1. 実際の磁場軌道から S(H) の平均 S_bar と平均不安定度 U_bar を測る
      2. 定常状態の釣り合いから解析的に決まるものを逆算
           He 収支: Gamma_He_TC * phi_He * S_bar = Gamma_TC_He * phi_TC
           Sk 収支: (k_nuc + k_growth*phi_TC*phi_Sk) * S_bar
                      = k_unwind * phi_Sk * U_bar
      3. onset サイクル数は解析的に解けないため k_growth のみ二分法で合わせる
      4. r_ann は psi_sat_cycles から逆算

    自由に選ぶのは Gamma_He_TC (TC 立ち上がり時定数から決まる) と k_nuc のみで、
    残りは全て目標値から導かれるため、恣意的な手探り探索が発生しない。
    """
    target = target or TargetDynamics()
    p = Module1Params() if base is None else Module1Params(**base.to_dict())

    # S_bar はドメインアンサンブル平均で評価する (ドメインごとに窓が dH だけ
    # ずれているため、単一ドメインの値を使うと定常相分率が目標からずれる)
    dom = build_domains(p)
    H_arr = np.asarray(H_trajectory, dtype=float)
    S_vals = soft_window(H_arr[:, None] - dom["dH"][None, :],
                         p.H_min, p.H_max, p.H_edge_width)
    S_bar = float(np.mean(S_vals @ dom["weight"]))
    U_bar = 1.0 - S_bar
    spc = p.steps_per_cycle

    # (a) Gamma_He_TC : TC の立ち上がり時定数から
    p.Gamma_He_TC = 1.0 / (target.tc_rise_cycles * S_bar * spc)
    # (b) Gamma_TC_He : 定常 phi_He/phi_TC 比から
    p.Gamma_TC_He = p.Gamma_He_TC * S_bar * target.phi_He_ss / target.phi_TC_ss
    # (c) r_ann : psi の飽和サイクル数から
    p.r_ann = 1.0 / (target.psi_sat_cycles * target.phi_Sk_ss * spc)

    def _k_unwind_for(kg):
        num = (p.k_nuc + kg * target.phi_TC_ss * target.phi_Sk_ss) * S_bar
        return num / max(target.phi_Sk_ss * U_bar, 1e-12)

    def _onset_for(kg):
        p.k_growth = kg
        p.k_unwind = _k_unwind_for(kg)
        res = simulate_module1(H_trajectory, p, return_all=False)
        return measure_dynamics(res, target.onset_level)["n_onset"]

    # (d) k_growth : onset サイクル数に二分法で合わせる (k_growth 大 -> onset 早い)
    lo, hi = 1e-3, 5.0
    for _ in range(45):
        mid = np.sqrt(lo * hi)
        if _onset_for(mid) > target.n_onset:
            lo = mid
        else:
            hi = mid
        if hi / lo < 1.0005:
            break
    p.k_growth = float(np.sqrt(lo * hi))
    p.k_unwind = _k_unwind_for(p.k_growth)

    if verbose:
        res = simulate_module1(H_trajectory, p, return_all=False)
        m = measure_dynamics(res, target.onset_level)
        print("[calibrate_module1]")
        print(f"  S_bar={S_bar:.4f}  U_bar={U_bar:.4f}")
        print(f"  Gamma_He_TC={p.Gamma_He_TC:.5g}  Gamma_TC_He={p.Gamma_TC_He:.5g}")
        print(f"  k_growth={p.k_growth:.5g}  k_unwind={p.k_unwind:.5g}  r_ann={p.r_ann:.5g}")
        print("  --- achieved (target) ---")
        print(f"  n_onset   : {m['n_onset']:.0f} ({target.n_onset:.0f})")
        print(f"  phi_Sk_ss : {m['phi_Sk_ss']:.3f} ({target.phi_Sk_ss:.2f})  std={m['phi_Sk_std']:.4f}")
        print(f"  phi_TC_ss : {m['phi_TC_ss']:.3f} ({target.phi_TC_ss:.2f})  std={m['phi_TC_std']:.4f}")
        print(f"  phi_He_ss : {m['phi_He_ss']:.3f} ({target.phi_He_ss:.2f})  std={m['phi_He_std']:.4f}")
        print(f"  psi(end)  : {m['psi_end']:.3f}")
    return p


# ---------------------------------------------------------------------------
# 後方互換ラッパ
# ---------------------------------------------------------------------------
def simulate_module1_step31(H_trajectory, params, return_all=False):
    """旧 API 互換ラッパ。新規コードでは simulate_module1 を使うこと。"""
    res = simulate_module1(H_trajectory, params, return_all=False)
    if return_all:
        return (res["phi_Sk_obs"], res["mz_obs"], res["phi_He_obs"],
                res["phi_TC_obs"], res["phi_Sk_obs"])
    return res["phi_Sk_obs"], res["mz_obs"], res["phi_TC_obs"]


if __name__ == "__main__":
    n_cycles = 1000  # Lee論文の NL/MC 評価は 750 学習 + 250 試験 = 1000 サイクル
    mg = generate_mackey_glass(n_cycles, tau=17)
    u = 2.0 * (mg - mg.min()) / (mg.max() - mg.min()) - 1.0
    H_traj = build_field_trajectory(u, H_c=73.0, H_range=90.0)
    print(f"H range: {H_traj.min():.2f} - {H_traj.max():.2f} mT\n")
    params = calibrate_module1(H_traj, TargetDynamics())
    print()
    print(params.describe())