import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt


def load_single_column_csv(csv_path: Path) -> list[float]:
    values: list[float] = []
    with csv_path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            if not row:
                continue
            values.append(float(row[-1]))
    return values


def normalize(values: list[float]) -> list[float]:
    min_value = min(values)
    max_value = max(values)
    if max_value == min_value:
        return [0.0 for _ in values]
    return [(value - min_value) / (max_value - min_value) for value in values]


def mse(values_a: list[float], values_b: list[float]) -> float:
    if len(values_a) != len(values_b):
        raise ValueError("Series must have the same length")
    return sum((a - b) ** 2 for a, b in zip(values_a, values_b)) / len(values_a)


def pearson_corr(values_a: list[float], values_b: list[float]) -> float:
    if len(values_a) != len(values_b):
        raise ValueError("Series must have the same length")

    mean_a = sum(values_a) / len(values_a)
    mean_b = sum(values_b) / len(values_b)
    numerator = sum((a - mean_a) * (b - mean_b) for a, b in zip(values_a, values_b))
    denom_a = math.sqrt(sum((a - mean_a) ** 2 for a in values_a))
    denom_b = math.sqrt(sum((b - mean_b) ** 2 for b in values_b))
    if denom_a == 0 or denom_b == 0:
        return 0.0
    return numerator / (denom_a * denom_b)


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    mg_csv = base_dir / "MG.csv"
    scan_csv = base_dir / "scan_fields_extracted_from_filenames.csv"

    mg_values = load_single_column_csv(mg_csv)
    scan_values: list[float] = []
    with scan_csv.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            scan_values.append(float(row["extracted_field_mT"]))

    window_size = len(scan_values)
    if len(mg_values) < window_size:
        raise ValueError(f"MG data must have at least {window_size} points, but got {len(mg_values)}")

    scan_norm = normalize(scan_values)

    candidates: list[dict[str, float | int]] = []
    for start_index in range(0, len(mg_values) - window_size + 1):
        window = mg_values[start_index:start_index + window_size]
        window_norm = normalize(window)
        score_mse = mse(window_norm, scan_norm)
        score_corr = pearson_corr(window_norm, scan_norm)
        candidates.append(
            {
                "start_index": start_index,
                "end_index": start_index + window_size - 1,
                "mse": score_mse,
                "corr": score_corr,
            }
        )

    best_by_mse = min(candidates, key=lambda item: item["mse"])
    best_by_corr = max(candidates, key=lambda item: item["corr"])
    top5_by_mse = sorted(candidates, key=lambda item: item["mse"])[:5]

    best_window = mg_values[best_by_mse["start_index"] : best_by_mse["end_index"] + 1]
    best_window_norm = normalize(best_window)

    x_axis = list(range(1, window_size + 1))
    output_plot = base_dir / "best_mg_scan_match_normalized.png"
    output_text = base_dir / "best_mg_scan_match_report.txt"

    plt.figure(figsize=(12, 5))
    plt.plot(x_axis, best_window_norm, label=f"best MG window {best_by_mse['start_index'] + 1}-{best_by_mse['end_index'] + 1}", linewidth=1.5)
    plt.plot(x_axis, scan_norm, label="scan (all 500 points)", linewidth=2.0, alpha=0.85)
    plt.title("Best matching normalized MG window versus scan fields")
    plt.xlabel("sample index within 500-point window")
    plt.ylabel("normalized value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_plot, dpi=200)
    plt.close()

    with output_text.open("w", encoding="utf-8") as report_file:
        report_file.write(f"MG total points: {len(mg_values)}\n")
        report_file.write(f"scan points: {len(scan_values)}\n")
        report_file.write(f"window size: {window_size}\n")
        report_file.write("\nBest match by MSE\n")
        report_file.write(
            f"  start_index: {best_by_mse['start_index'] + 1}\n"
            f"  end_index: {best_by_mse['end_index'] + 1}\n"
            f"  mse: {best_by_mse['mse']:.12e}\n"
            f"  corr: {best_by_mse['corr']:.12e}\n"
        )
        report_file.write("\nBest match by correlation\n")
        report_file.write(
            f"  start_index: {best_by_corr['start_index'] + 1}\n"
            f"  end_index: {best_by_corr['end_index'] + 1}\n"
            f"  mse: {best_by_corr['mse']:.12e}\n"
            f"  corr: {best_by_corr['corr']:.12e}\n"
        )
        report_file.write("\nTop 5 candidates by MSE\n")
        for rank, candidate in enumerate(top5_by_mse, start=1):
            report_file.write(
                f"  {rank}. start={candidate['start_index'] + 1}, end={candidate['end_index'] + 1}, "
                f"mse={candidate['mse']:.12e}, corr={candidate['corr']:.12e}\n"
            )

    print(f"MG CSV: {mg_csv}")
    print(f"scan CSV: {scan_csv}")
    print(f"best match plot saved: {output_plot}")
    print(f"best match report saved: {output_text}")
    print(f"best start index (1-based): {best_by_mse['start_index'] + 1}")
    print(f"best end index (1-based): {best_by_mse['end_index'] + 1}")
    print(f"best MSE: {best_by_mse['mse']:.12e}")
    print(f"best corr: {best_by_mse['corr']:.12e}")


if __name__ == "__main__":
    main()