# We use this script to test the standard fact checker on a dataset. 
# We load the dataset, iterate over the rows, and call the fact checker for each row. 
# We then extract the label from the output and compare it with the true label. 
# We save the predictions in a new column of the dataset.
from factchecker import *
import sys
import os
import re
import pandas as pd
import time

class LoggingPrinter:
    def __init__(self, filename):
        self.out_file = open(filename, "w", encoding="utf-8")
        self.old_stdout = sys.stdout
        #this object will take over `stdout`'s job
        sys.stdout = self
    #executed when the user does a `print`
    def write(self, text): 
        self.old_stdout.write(text)
        self.out_file.write(text)
    #executed when `with` block begins
    def __enter__(self): 
        return self
    #executed when `with` block ends
    def __exit__(self, type, value, traceback): 
        #we don't want to log anymore. Restore the original stdout object.
        sys.stdout = self.old_stdout


def find_label(model_output, label_list, language):
    """
    This function extracts the predicted label from the model output
    Args:
        model_output: the output text of the model
        label_list: the list of possible labels
        language: the language of the model output
    Returns:
        the predicted label if it is found in the model output, "ERR" otherwise
    """
    model_output = model_output.lower()
    #take only the part after "final answer:"
    if language == "en":
        model_output = model_output.split("final answer:")[1]
    elif language == "it":
        model_output = model_output.split("verdetto finale:")[1]
    for label in label_list:
        if label in model_output:
            if label == "mostly-true":
                return "mostly true"
            elif label == "mostly-false":
                return "mostly false"
            return label
    return "ERR"

def main():
    # Load the configuration file
    my_config = load_config("my_config.yaml")
    already_predicted_claims = my_config['already_predicted_claims']
    language = my_config['language']
    dataset_path = my_config['dataset_path']
    label_column_name = my_config['label_column_name']
    claim_column_name = my_config['claim_column_name']
    date_column_name = my_config['date_column_name']
    author_column_name = my_config['author_column_name']
    possible_labels = my_config['possible_labels']
    prediction_column_name = my_config['prediction_column_name']
    log_file_path = my_config['log_file_path']
    # Load the dataset using pandas
    dataset = pd.read_csv(dataset_path)
    if prediction_column_name not in dataset.columns:
        dataset[prediction_column_name] = "ERR"
    # iterate over the rows of the dataset
    with LoggingPrinter(log_file_path):
        for index, row in dataset.iterrows():
            if index < already_predicted_claims:
                print("Skipping tuple",index,":",row[claim_column_name])
                continue

            print("Tuple",index)
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
            while True:
                try:
                    response = generate_output(user_input)
                    break
                except Exception as e:
                    print("Error: ", e)
                    print("Pausing for a while and Retrying...")
                    # wait for 10 minutes and retry
                    time.sleep(60*10)
            # Print the messages
            l = find_label(response, possible_labels, language)
            print("Label extracted: ", l,". True label: ", label+"\n")
            dataset.at[index, prediction_column_name] = l
    # Save the dataset
    dataset.to_csv("Datasets/politifact_150.csv", index=False)

if __name__ == "__main__":
    main()