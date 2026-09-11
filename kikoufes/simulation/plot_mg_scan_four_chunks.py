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


def make_x_axis(length: int) -> list[int]:
    return list(range(1, length + 1))


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    mg_csv = base_dir / "MG.csv"
    scan_csv = base_dir / "scan_fields_extracted_from_filenames.csv"
    output_path = base_dir / "MG_scan_four_chunks_normalized.png"

    mg_values = load_single_column_csv(mg_csv)
    scan_values: list[float] = []
    with scan_csv.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            scan_values.append(float(row["extracted_field_mT"]))

    mg_front = mg_values[:2000]
    if len(mg_front) < 2000:
        raise ValueError(f"MG data must have at least 2000 points, but got {len(mg_front)}")
    if len(scan_values) < 500:
        raise ValueError(f"scan data must have at least 500 points, but got {len(scan_values)}")

    mg_chunks = [mg_front[i:i + 500] for i in range(0, 2000, 500)]
    scan_chunk = scan_values[:500]

    x_values = make_x_axis(500)

    plt.figure(figsize=(13, 6))
    for index, chunk in enumerate(mg_chunks, start=1):
        plt.plot(
            x_values,
            normalize(chunk),
            linewidth=1.1,
            alpha=0.85,
            label=f"MG{index} (points {500 * (index - 1) + 1}-{500 * index})",
        )

    plt.plot(
        x_values,
        normalize(scan_chunk),
        linewidth=2.0,
        alpha=0.95,
        label="scan (first 500 points)",
        color="black",
    )

    plt.title("Normalized comparison of MG chunks (first 2000 points) and scan fields")
    plt.xlabel("sample index within each 500-point segment")
    plt.ylabel("normalized value")
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"MG CSV: {mg_csv}")
    print(f"scan CSV: {scan_csv}")
    print(f"comparison plot saved: {output_path}")
    print(f"MG total points: {len(mg_values)}")
    print(f"MG used points: {len(mg_front)}")
    print(f"scan used points: {len(scan_chunk)}")


if __name__ == "__main__":
    main()