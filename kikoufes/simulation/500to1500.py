import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- 1. CSVデータの読み込み ---
csv_path = "scan_fields_extracted_from_filenames.csv"  # 手元のファイルパスに合わせて変更
df = pd.read_csv(csv_path)

# H_mid の配列（500点）
h_low = df["extracted_field_mT"].values
num_cycles = len(h_low)
print(f"読み込み完了: {num_cycles} サイクル")

# --- 2. H_mid, H_high の復元と1500点軌道の生成 ---
H_RANGE = 90  # mT

h_high = h_low + (H_RANGE / 2.0)
h_mid = h_low + (H_RANGE / 4.0)


# 1サイクルあたり 3点 (H_mid -> H_high -> H_low) として結合する場合:
# 各サイクルの状態更新用シーケンス
# shape: (500, 3)
cycle_3points = np.column_stack([h_mid, h_high, h_low])

# 全1500点のフラットな時系列配列 (T=1500)
H_trajectory_1500 = cycle_3points.flatten()

print(f"復元された磁場軌道 shape: {H_trajectory_1500.shape}")
print(f"磁場範囲: {H_trajectory_1500.min():.2f} mT ～ {H_trajectory_1500.max():.2f} mT")

# --- 3. 保存 ---
# ODE入力用として npy に保存
np.save("H_trajectory_1500.npy", H_trajectory_1500)
# サイクル単位でも保持しておくと後で便利
np.save("H_cycle_3points.npy", cycle_3points)
print("-> H_trajectory_1500.npy, H_cycle_3points.npy に保存しました。")

# --- 4. 復元波形の目視確認 (先頭5サイクル) ---
plt.figure(figsize=(10, 4))
plot_points = 15  # 5サイクル分 = 15点
time_steps = np.arange(plot_points)

plt.plot(time_steps, H_trajectory_1500[:plot_points], "o-", label="MFC Trajectory")
# 測定点（各サイクルの終点 = index 2, 5, 8, ...）を強調
meas_indices = np.arange(2, plot_points, 3)
plt.scatter(meas_indices, H_trajectory_1500[meas_indices], color="red", s=100, zorder=5, label="Measurement (H_low)")

plt.xlabel("Trajectory Step")
plt.ylabel("Magnetic Field (mT)")
plt.title("MFC Protocol Field Trajectory (First 5 Cycles)")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()