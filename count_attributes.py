import pandas as pd

df = pd.read_csv('Datasets/standardized_dataset.csv', encoding="utf-16", sep="\t")

# Count occurrences of each attribute in column_name
column_name = "reviewRating.alternateName"
counts = df[column_name].value_counts()

# Print the counts
for attribute, count in counts.items():
    print(f"{attribute} : {count}")
