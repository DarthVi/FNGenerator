import csv

names = []
with open("namedb.csv") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=",")
    next(csv_reader)  # skip first line with column names
    for row in csv_reader:
        for name in row:
            if name != "":
                names.append(name)

with open("names.txt", "w") as f:
    for word in names:
        f.write(word + "\n")
