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


def save_series_plot(values: list[float], title: str, ylabel: str, output_path: Path) -> None:
    x_values = list(range(1, len(values) + 1))

    plt.figure(figsize=(12, 4))
    plt.plot(x_values, values, linewidth=1.0)
    plt.title(title)
    plt.xlabel("sample index")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


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

    mg_plot_path = base_dir / "MG_series_plot.png"
    scan_plot_path = base_dir / "scan_fields_series_plot.png"

    save_series_plot(
        mg_values,
        title=f"Mackey-Glass series ({len(mg_values)} points)",
        ylabel="MG value",
        output_path=mg_plot_path,
    )
    save_series_plot(
        scan_values,
        title=f"Extracted scan field values ({len(scan_values)} points)",
        ylabel="field [mT]",
        output_path=scan_plot_path,
    )

    print(f"MG CSV: {mg_csv}")
    print(f"scan CSV: {scan_csv}")
    print(f"MG plot saved: {mg_plot_path}")
    print(f"scan plot saved: {scan_plot_path}")
    print(f"MG points: {len(mg_values)}")
    print(f"scan points: {len(scan_values)}")


if __name__ == "__main__":
    main()