import sys
import csv
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from prcpy.DataHandling.Path_handlers import get_directory_file_names, get_raw_input_vals


def main() -> None:
    data_dir = Path(__file__).resolve().parents[1] / "PRCpy" / "data_full" / "mg_mapping" / "Cu2OSeO3" / "skyrmion"
    prefix = "scan"
    output_path = Path(__file__).resolve().parent / "scan_fields_extracted_from_filenames.csv"

    file_names = get_directory_file_names(str(data_dir), prefix)
    field_values = get_raw_input_vals(str(data_dir), prefix)

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["file_name", "extracted_field_mT"])
        for file_name, field_value in zip(file_names, field_values):
            writer.writerow([file_name, field_value])

    print(f"data_dir = {data_dir}")
    print(f"scan file count = {len(file_names)}")
    print(f"extracted field value count = {len(field_values)}")
    print(f"csv output = {output_path}")
    print(f"first 10 fields = {field_values[:10]}")
    print(f"last 10 fields = {field_values[-10:]}")
    print(f"min field = {min(field_values)} mT")
    print(f"max field = {max(field_values)} mT")


if __name__ == "__main__":
    main()