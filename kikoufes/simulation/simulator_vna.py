import os
import numpy as np
from pathlib import Path

def run_simulation(
    input_signal, 
    phase_mode="skyrmion", 
    output_dir="sim_data/Cu2OSeO3/skyrmion",
    H_c=60.0, 
    H_range=90.0,
    prefix="scan"
):
    """
    Cu2OSeO3の磁気相ダイナミクスとVNAスペクトルを模擬して
    PRCpy準拠のscan_*.txtファイルを生成する
    """
    os.makedirs(output_dir, exist_ok=True)
    N_cycles = len(input_signal)
    
    # 1.0 GHz から 6.0 GHz の 1601 点 (実機仕様)
    freqs = np.linspace(1.0, 6.0, 1601)
    
    # MFCプロトコル: H_low(N) の計算 (mT)
    # u in [-1, 1]
    H_low = (H_c - H_range / 2.0) + input_signal * (H_range / 4.0)
    
    # 内部状態変数の初期化
    state_mem = 0.0
    
    print(f"Generating {N_cycles} VNA spectra files for [{phase_mode}] mode...")
    
    for n in range(N_cycles):
        u_n = input_signal[n]
        h_n = H_low[n]
        
        if phase_mode == "skyrmion":
            # スキルミオン相: 履歴依存のフェーディングメモリ + 漸進的核形成
            # 論文: 2〜3 GHz帯のCCW/Breathingモード + 弱残存4 GHzモード
            decay = 0.85  # メモリ保持係数
            gain = 0.30
            state_mem = decay * state_mem + gain * np.tanh(u_n * 1.5)
            
            # 共鳴周波数の変調
            f_res1 = 2.2 + 0.003 * (h_n - H_c) + 0.05 * state_mem
            f_res2 = 2.6 + 0.004 * (h_n - H_c) + 0.08 * state_mem
            
            # ピーク振幅 (サイクル数と履歴に伴う核形成密度の寄与)
            amp1 = 2.5 + 1.2 * state_mem
            amp2 = 1.8 + 0.8 * state_mem
            gamma1, gamma2 = 0.12, 0.15
            
            # ローレンツ吸収成分の合成
            peak1 = amp1 * (gamma1**2) / ((freqs - f_res1)**2 + gamma1**2)
            peak2 = amp2 * (gamma2**2) / ((freqs - f_res2)**2 + gamma2**2)
            spectra = -(peak1 + peak2)
            
        elif phase_mode == "conical":
            # コニカル相: 履歴なし (即時応答), 強い非線形分散シフト (3.5〜4.5 GHz)
            f_res = 3.9 + 0.015 * (h_n - H_c) - 0.0001 * ((h_n - H_c)**2)
            amp = 4.5 + 1.5 * np.cos(u_n * np.pi)
            gamma = 0.08
            
            peak = amp * (gamma**2) / ((freqs - f_res)**2 + gamma**2)
            spectra = -peak
            
        else:
            raise ValueError(f"Unknown phase_mode: {phase_mode}")
            
        # 軽微な観測ノイズ付加 (SNR調整)
        noise = np.random.normal(0, 0.02, size=len(freqs))
        spectra = spectra + noise
        
        # 実機フォーマットに合わせてカンマ区切りで保存
        # ヘッダー: Frequency,Spectra
        filename = os.path.join(output_dir, f"{prefix}_{n+1}.txt")
        file_content = np.column_stack((freqs, spectra))
        np.savetxt(filename, file_content, delimiter=",", header="Frequency,Spectra", comments="")

    print(f"Simulation completed. Files saved to: {output_dir}")

if __name__ == "__main__":
    # テスト実行: Mackey-Glassデータの読み込み
    mg_path = "../PRCpy/data_full/chaos/mackey_glass_t17.npy"
    if os.path.exists(mg_path):
        mg_data = np.load(mg_path)
        # [-1, 1] に正規化
        u = 2.0 * (mg_data - np.min(mg_data)) / (np.max(mg_data) - np.min(mg_data)) - 1.0
        
        # スキルミオン相のシミュレーション出力 (先頭500点)
        run_simulation(
            input_signal=u[:500], 
            phase_mode="skyrmion", 
            output_dir="data_full/sim_test/skyrmion", 
            H_c=60.0
        )
    else:
        print(f"Input file not found at {mg_path}. Please check your path.")