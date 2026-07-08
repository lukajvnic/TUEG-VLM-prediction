from pathlib import Path
import csv
import sys

max_size = sys.maxsize

while True:
    try:
        csv.field_size_limit(max_size)
        break
    except OverflowError:
        max_size = int(max_size / 10)

folder = Path("../results")

rows = [
    ["model", "correct", "total", "accuracy", "balanced"]
]

for file in folder.iterdir():
    if file.is_file() and not file.name.startswith("T"):
        correct_count = 0
        total_count = 0
        correct_epilepsy = 0
        correct_no_epilepsy = 0
        with open(f"../results/{file.name}", 'r', encoding="utf-8") as file:
            reader = csv.reader(file)
            for row in reader:
                total_count += 1
                if row[3] == 'True':
                    correct_count += 1
                    if "NO_EPILEPSY" in row[1]:
                        correct_no_epilepsy += 1
                    else:
                        correct_epilepsy += 1
        acc = correct_count / total_count
        balanced_acc = ()
        if total_count == 2809:
            rows.append([file.name, correct_count, total_count, acc])
        print(f"{file.name} - ({correct_count}/{total_count}) - {acc:.3f}")


with open(f"summary.csv", 'w', encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerows(rows)
