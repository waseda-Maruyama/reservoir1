import numpy as np

# --- 1. 実験条件（Fig. 2b プロトコル） ---
H_c = 60.0              # 中心磁場 (mT)
H_range = 90.0          # 掃引幅 (mT)  -> H_high(max) - H_low(min) = H_range
n_cycles = 920           # サイクル数

# sine変調の振幅: H_high(max) - H_low(min) = H_range となるように
# Hmid(N) = Hc + A*sin(...), Hlow = Hmid - A, Hhigh = Hmid + A
# => Hhigh_max - Hlow_min = (Hc+A+A) - (Hc-A-A) = 4A = H_range
A = H_range / 4.0        # 22.5 mT

# 入力信号の周期（field-cycle数）。Fig. 2bはN=100-200付近で
# 「無ピーク -> 成長 -> 崩壊」の一山が見えるので、まずはこの近辺に
# 山が来るように仮設定。実測パターンに合わせて要チューニング。
period = 100.0

# --- 2. 920 サイクル分の連続軌道生成（MFC: Hlow -> Hmid -> Hhigh の3点/サイクル） ---
N = np.arange(n_cycles)
u_N = np.sin(2.0 * np.pi * N / period)

H_mid_N = H_c + A * u_N
H_low_N = H_mid_N - A
H_high_N = H_mid_N + A

# 各サイクルを [H_low, H_mid, H_high] の順で並べてフル軌道に展開
# (どの順で走査するかは実装依存。ここではmid->high->low->次サイクルのmidと
#  連続的に動く順を採用。observationはH_low地点で行う設計と対応させる)
H_trajectory = np.empty(3 * n_cycles, dtype=np.float64)
H_trajectory[2::3] = H_low_N
H_trajectory[0::3] = H_mid_N
H_trajectory[1::3] = H_high_N

# --- 3. .npy 形式で保存 ---
output_filename = "H_trajectory_920_sine.npy"
np.save(output_filename, H_trajectory)

print("Sine変調MFC軌道を生成・保存しました:")
print(f"  ファイル名 : {output_filename}")
print(f"  配列形状   : {H_trajectory.shape}")
print(f"  サイクル数 : {n_cycles}")
print(f"  振幅 A     : {A} mT (= H_range/4)")
print(f"  周期       : {period} cycles")
print(f"  H_mid range: [{H_mid_N.min():.2f}, {H_mid_N.max():.2f}] mT")
print(f"  H_low range: [{H_low_N.min():.2f}, {H_low_N.max():.2f}] mT")
print(f"  H_high range: [{H_high_N.min():.2f}, {H_high_N.max():.2f}] mT")