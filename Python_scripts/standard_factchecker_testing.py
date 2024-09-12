from standard_fact_checker.factchecker import *
import sys
import os
import re
import pandas as pd

def main():
    my_config = load_config("my_config.yaml")
    dataset_path = my_config['dataset_path']
    label_column_name = my_config['label_column_name']
    claim_column_name = my_config['claim_column_name']
    date_column_name = my_config['date_column_name']
    author_column_name = my_config['author_column_name']
    # Load the dataset using pandas
    dataset = pd.read_csv(dataset_path)
    # iterate over the rows of the dataset
    for index, row in dataset.iterrows():
        claim = row[claim_column_name]
        # Extract the label from the row
        label = row[label_column_name]
        # Call the fact checker function
        messages = fact_checker(text, label)
        # Print the messages
        for m in messages:
            print(m)

if __name__ == "__main__":
    main()