#from Python_scripts.HiSS_Prompt_Composer.factchecker import *
from factchecker import *
import sys
import os
import re
import pandas as pd

my_config = load_config("my_config.yaml")
dataset_input_path = my_config['dataset_input_path']
dataset_output_path = my_config['dataset_output_path']
log_file_path = my_config['log_file_path']
label_column_name = my_config['label_column_name']
claim_column_name = my_config['claim_column_name']
date_column_name = my_config['date_column_name']
author_column_name = my_config['author_column_name']
possible_labels = my_config['possible_labels']
labels_values = my_config['labels_values']
prediction_column_name = my_config['prediction_column_name']
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
            if label == "mostly-true":
                return "mostly true"
            elif label == "mostly-false":
                return "mostly false"
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

# This function tells us if the two received labels can be considered as having a similar veracity values
def is_similar(predicted_label, true_label):
    if(true_label == "mostly false"):
        true_label = "mostly-false"
    if(true_label == "mostly true"):
        true_label = "mostly-true"

    label_to_number = {possible_labels[i].lower() : labels_values[i] for i in range(len(possible_labels))}

    mapped_predicted_label = label_to_number[predicted_label]
    mapped_true_label = label_to_number[true_label]

    if(abs(mapped_true_label - mapped_predicted_label) <= 1):
        return True
    else:
        return False

def main():
    # Load the dataset using pandas
    dataset = pd.read_csv(dataset_input_path)
    start_index = 0
    #end_index = 15
    end_index = dataset.size - 1
    correct_predictions = 0
    number_of_claims = 0
    #dataset = dataset.iloc[start_index : end_index]
    if prediction_column_name not in dataset.columns:
        dataset[prediction_column_name] = "ERR"

    # Iterate over the rows of the dataset
    with LoggingPrinter(log_file_path):
        for index, row in dataset.iterrows():
            if index < already_predicted_claims:
                print(f"Skipping tuple {index}: {row[claim_column_name]}")
                continue

            print("Tuple",index)
            claim = row[claim_column_name]
            date = row[date_column_name]
            author = row[author_column_name]
            #date = ""
            #author = ""
            true_label = row[label_column_name]
            
            # Call the fact checker function
            if pd.isna(date):
                date = ""
            if pd.isna(author):
                author = ""
            user_input = {"claim": claim, "date": date, "author": author}

            for i in range(100):
                try:
                    response = generate_output(user_input)
                    break
                except Exception as e:
                    print(f"User_input is: \n{user_input}\n End of user input")
                    print(f"Error is: \n {e}\n End of error")
                    print("Pausing for a while and retrying...")
                    time.sleep(60*10)

            # Print the predicted label
            #predicted_label = extract_final_answer(response, possible_labels)
            predicted_label = find_label(response, possible_labels)
            print("Label extracted: ", predicted_label,". True label: ",true_label+"\n")
            dataset.at[index, prediction_column_name] = predicted_label

            # Calculate the current accurancy
            ''' 
            if predicted_label == "ERR":
                number_of_claims -= 1
            elif is_similar(predicted_label, true_label.lower()):
                correct_predictions += 1
            number_of_claims += 1
            print(f"Current accurancy is: {correct_predictions / number_of_claims:.3f}")'''
    # Save the dataset
    dataset.to_csv(dataset_output_path)

if __name__ == "__main__":
    main()