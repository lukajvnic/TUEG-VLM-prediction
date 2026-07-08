import csv

with open(f"labels.csv", 'r', encoding="utf-8") as file:
    reader = csv.reader(file)
    total = 0
    epileptic = 0
    for row in reader:
        total += 1
        if row[1] == 'true':
            epileptic += 1

    print(f"Total: {total}")
    print(f"Epileptic: {epileptic}")