#from Python_scripts.HiSS_Prompt_Composer.factchecker import *
from factchecker import *
import pandas as pd
import re
import sys

my_config = load_config("my_config.yaml")
dataset_input_path = my_config['dataset_input_path']
dataset_output_path = my_config['dataset_output_path']
log_file_path = my_config['log_file_path']
label_column_name = my_config['label_column_name']
claim_column_name = my_config['claim_column_name']
date_column_name = my_config['date_column_name']
author_column_name = my_config['author_column_name']
prediction_column_name = my_config['prediction_column_name']
domain_column_name = my_config['domain_column_name']
model_id = my_config['model_id']
possible_labels = my_config['possible_labels']
already_predicted_claims = my_config['already_predicted_claims']

# The class to keep the logs of our operations
class LoggingPrinter:
    def __init__(self, filename):
        self.out_file = open(filename, "w", encoding="utf-8")
        self.old_stdout = sys.stdout
        # This object will take over `stdout`'s job
        sys.stdout = self

    # Executed when the user does a `print`, print both on the screen as well as on the screen
    def write(self, text): 
        self.old_stdout.write(text)
        self.out_file.write(text)

    # Executed when a `with` block begins (necessary for LoggingPrinter to be used in a with block)
    def __enter__(self): 
        return self
    
    # Executed when a `with` block ends
    def __exit__(self, type, value, traceback): 
        # We don't want to log anymore. Restore the original stdout object.
        sys.stdout = self.old_stdout

# This function extracts the final veracity label given by the model
def extract_final_answer(output, possible_labels):
    if 'Based on' in output or 'I would classify the claim as' in output:
        pattern = r"\b(" + "|".join(map(re.escape, possible_labels)) + r")[,.]?\b"
        matches = re.findall(pattern, output, re.IGNORECASE)
        if matches:
            final_label = matches[-1].lower()
            return final_label
        else:
            print("The model seems to answer the question, but not to provide any veracity label")
            return "ERR"
    else:
        print("Error, no answer has been found in the output")
        return "ERR"

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
    print("Starting testing on the dataset", dataset_input_path)
    print("Output predictions will be saved in", dataset_output_path)
    print("Log file will be saved in", log_file_path)
    print("The model in use is", model_id)

    # Load the dataset using pandas
    dataset = pd.read_csv(dataset_input_path)
    if prediction_column_name not in dataset.columns:
        print("Column for predictions not found. Creating a new one.")
        dataset[prediction_column_name] = "ERR"
    else:
        print("Column for predictions found. Using it.")

    # Iterate over the rows of the dataset
    with LoggingPrinter(log_file_path):
        for index, row in dataset.iterrows():
            if index < already_predicted_claims:
                print(f"Skipping tuple {index}: {row[claim_column_name]}")
                continue

            print("Tuple", index)
            claim = row[claim_column_name]
            date = row[date_column_name]
            author = row[author_column_name]
            domain = row[domain_column_name]
            true_label = row[label_column_name]
            
            # Call the fact checker function
            if pd.isna(date):
                date = ""
            if pd.isna(author):
                author = ""
            if pd.isna(domain):
                domain = ""
            user_input = {"claim": claim, "date": date, "author": author, "domain": domain}

            try:
                response = generate_output(user_input)
            except Exception as e:
                print("Skipping this tuple")
                response = "ERR"

            # Print the predicted label
            predicted_label = extract_final_answer(response, possible_labels)
            print("Label extracted: ", predicted_label, ". True label: ", true_label + "\n")
            dataset.at[index, prediction_column_name] = predicted_label

    # Save the dataset
    dataset.to_csv(dataset_output_path)
    print("The predictions have been saved in", dataset_output_path)

if __name__ == "__main__":
    main()