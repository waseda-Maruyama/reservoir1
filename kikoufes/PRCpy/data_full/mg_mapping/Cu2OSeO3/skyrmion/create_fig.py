from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATA_DIR = Path(__file__).resolve().parent
N_START = 200
N_END = 300
N_STEP = 10
BACKGROUND_NAME = "BG_450mT_1_0to6_0GHz_4K.txt"
OUTPUT_NAME = "skyrmion_bg_subtracted_waterfall_N100_N200_step10.png"


def scan_number(path):
    match = re.match(r"scan_(\d+)_", path.name)
    if match is None:
        raise ValueError(f"Could not read N from filename: {path.name}")
    return int(match.group(1))


def load_scan(path):
    data = pd.read_csv(path)
    required_columns = {"Frequency", "Spectra"}
    missing = required_columns - set(data.columns)
    if missing:
        raise ValueError(f"Missing columns in {path.name}: {sorted(missing)}")
    return data["Frequency"].to_numpy(), data["Spectra"].to_numpy()


def load_background():
    path = DATA_DIR / BACKGROUND_NAME
    frequency, spectra = load_scan(path)
    return frequency, spectra


def find_scan_files():
    files_by_n = {}
    for path in DATA_DIR.glob("scan_*.txt"):
        n = scan_number(path)
        if n in files_by_n:
            raise ValueError(f"Multiple files found for N={n}: {files_by_n[n].name}, {path.name}")
        files_by_n[n] = path

    selected_n = range(N_START, N_END + 1, N_STEP)
    missing = [n for n in selected_n if n not in files_by_n]
    if missing:
        raise FileNotFoundError(f"Scan files not found for N={missing}")
    return [(n, files_by_n[n]) for n in selected_n]


def plot_waterfall(scan_data):
    first_n, first_path = scan_data[0]
    first_frequency, _ = load_scan(first_path)
    background_frequency, background_spectra = load_background()
    if not np.array_equal(background_frequency, first_frequency):
        raise ValueError(
            f"Background frequency axis differs from N={first_n}: {BACKGROUND_NAME}"
        )

    fig, ax = plt.subplots(figsize=(7, 8))
    offset_step = 1.5
    for index, (n, path) in enumerate(scan_data):
        frequency, spectra = load_scan(path)
        if not np.array_equal(frequency, first_frequency):
            raise ValueError(f"Frequency axis differs from N={first_n}: {path.name}")
        spectra = spectra - background_spectra
        y_offset = index * offset_step
        ax.plot(frequency, spectra + y_offset, lw=1.2)
        ax.text(frequency[-1] + 0.05, y_offset, f"N={n}", va="center", fontsize=8)

    ax.set_xlabel("f (GHz)")
    ax.set_ylabel(r"$\Delta S_{11} - \Delta S_{11}^{BG}$ (offset, arb.)")
    ax.set_title("Skyrmion measured spectra (background subtracted)")
    ax.set_yticks([])
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    scan_data = find_scan_files()
    figure = plot_waterfall(scan_data)
    output_path = DATA_DIR / OUTPUT_NAME
    figure.savefig(output_path, dpi=150)
    print(f"Saved: {output_path}")
    print(f"Loaded N={N_START}..{N_END} every {N_STEP} cycles ({len(scan_data)} scans)")
    plt.show()
