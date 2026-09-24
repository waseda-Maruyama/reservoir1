"""scan_fields_extracted_from_filenames.csv から磁場列だけを抽出する。"""
import argparse
import csv
from pathlib import Path


def extract_extra_field(input_path: Path, output_path: Path) -> None:
    with input_path.open(newline="", encoding="utf-8") as input_file:
        reader = csv.DictReader(input_file)
        if reader.fieldnames is None or "extracted_field_mT" not in reader.fieldnames:
            raise ValueError(
                "入力 CSV に 'extracted_field_mT' 列がありません: "
                f"{input_path}"
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="", encoding="utf-8") as output_file:
            writer = csv.writer(output_file)
            writer.writerow(["extra_field"])
            for row in reader:
                value = row["extracted_field_mT"]
                if value == "":
                    raise ValueError("磁場列に空の値があります")
                writer.writerow([value])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CSV から extracted_field_mT 列だけを抽出します。"
    )
    parser.add_argument(
        "input_csv",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parent
        / "scan_fields_extracted_from_filenames.csv",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "extra_field.csv",
    )
    args = parser.parse_args()

    extract_extra_field(args.input_csv, args.output)
    print(f"入力: {args.input_csv}")
    print(f"出力: {args.output}")


if __name__ == "__main__":
    main()