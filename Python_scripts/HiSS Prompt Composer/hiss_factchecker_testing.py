from factchecker import *
import sys
import os
import re
import pandas as pd

def find_label(model_output, label_list):
    model_output = model_output.lower()
    #take only the last 20 characters
    model_output = model_output[-20:]
    for label in label_list:
        if label in model_output:
            return label
    return "ERR"

def main():
    my_config = load_config("my_config.yaml")
    dataset_path = my_config['dataset_path']
    label_column_name = my_config['label_column_name']
    claim_column_name = my_config['claim_column_name']
    date_column_name = my_config['date_column_name']
    author_column_name = my_config['author_column_name']
    possible_labels = my_config['possible_labels']
    # Load the dataset using pandas
    dataset = pd.read_csv(dataset_path)
    dataset["predicted_label"] = "ERR"
    # iterate over the rows of the dataset
    for index, row in dataset.iterrows():
        claim = row[claim_column_name]
        date = row[date_column_name]
        author = row[author_column_name]
        # Extract the label from the row
        label = row[label_column_name]
        # Call the fact checker function
        if pd.isna(date):
            date = ""
        if pd.isna(author):
            author = ""
        user_input = {"claim": claim, "date": date, "author": author}
        response = generate_output(user_input)
        # Print the messages
        l = find_label(response, possible_labels)
        print("Label extracted: ", l,". True label: ", label+"\n")
        dataset.at[index, "predicted_label"] = l
    # Save the dataset
    dataset.to_csv("Datasets/hiss_predictions.csv", index=False)

if __name__ == "__main__":
    main()