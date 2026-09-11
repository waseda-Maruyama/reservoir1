import csv
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


def make_x_axis(length: int) -> list[float]:
    if length <= 1:
        return [0.0] * length
    return [index / (length - 1) for index in range(length)]


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

    mg_norm = normalize(mg_values)
    scan_norm = normalize(scan_values)
    mg_x = make_x_axis(len(mg_norm))
    scan_x = make_x_axis(len(scan_norm))

    output_path = base_dir / "MG_scan_comparison_normalized.png"

    plt.figure(figsize=(12, 5))
    plt.plot(mg_x, mg_norm, label=f"MG (n={len(mg_values)})", linewidth=1.0, alpha=0.85)
    plt.plot(scan_x, scan_norm, label=f"scan field (n={len(scan_values)})", linewidth=1.5, alpha=0.85)
    plt.title("Normalized comparison of Mackey-Glass and extracted scan field values")
    plt.xlabel("normalized sample position")
    plt.ylabel("normalized value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"MG CSV: {mg_csv}")
    print(f"scan CSV: {scan_csv}")
    print(f"comparison plot saved: {output_path}")
    print(f"MG points: {len(mg_values)}")
    print(f"scan points: {len(scan_values)}")


if __name__ == "__main__":
    main()