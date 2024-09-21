from factchecker import *
import sys
import os
import re
import pandas as pd

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

def extract_final_answer(output):
    if 'Based on' in output or 'I would classify the claim as' in output:
        pattern = r"\b(false|mostly-false|mixture|mostly-true|true)[,.]?\b"
        matches = re.findall(pattern, output, re.IGNORECASE)
        final_label = matches[-1].lower()
        return final_label
    else:
        print("Error, no answer has been found in the output")
        return ""

def find_label(model_output, label_list):
    model_output = model_output.lower()
    #take only the last 20 characters
    model_output = model_output[-20:]
    for label in label_list:
        if label in model_output:
            if label == "mostly-true":
                return "mostly true"
            elif label == "mostly-false":
                return "mostly false"
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
    with LoggingPrinter("Logs/hiss_factchecker_testing_log.txt"):
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
            while True:
                try:
                    response = generate_output(user_input)
                    break
                except Exception as e:
                    print("Error: ", e)
                    print("Pausing for a while and retrying...")
                    time.sleep(60)
            # Print the messages
            l = find_label(response, possible_labels)
            print("Label extracted: ", l,". True label: ", label+"\n")
            dataset.at[index, "predicted_label"] = l
    # Save the dataset
    dataset.to_csv("Datasets/hiss_predictions.csv", index=False)

if __name__ == "__main__":
    main()