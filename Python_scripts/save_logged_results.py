# read txt Logs/standard_factchecker_testing_log.txt and look for every line in the format: "Label extracted:  [label name] . True label:  [label name]"
# open a csv with a ginven path and write the extracted label in the column "predicted_label"

import re
import csv
import pandas as pd

def save_log_result(log_file_path: str, csv_file_path: str):
    with open(log_file_path, "r", encoding="utf-8") as log_file:
        log_lines = log_file.readlines()
    i = 0
    # open the csv file with pandas
    dataset = pd.read_csv(csv_file_path)
    #print name of the columns
    print(dataset.columns)
    dataset["predicted_label_std_70B"] = "ERR"
    # iterate over the lines of the log file that contain the the lines in the format: "Label extracted:  [label name] . True label:  [label name]"
    for line in log_lines:
        # extract the claim, that can include any character, not just letters so include ALL characters, [a-z] is not enough
        if re.search(r"Claim: .+", line):
            re_out = re.search(r"Claim: (.+)", line)
            claim = re_out.group(1)
        if re.search(r"Label extracted:  [a-za-zA-Z ]+ . True label:  [a-za-zA-Z ]+", line):
            print("\nLogged claim: ", claim)
            print("Corresponding claim in the dataset: ", dataset.at[i, "claimReviewed"])
            # extract the predicted label and the true label
            re_out = re.search(r"Label extracted:  ([a-zA-Z ]+) . True label:  [a-zA-Z ]+", line)
            predicted_label = re_out.group(1)
            print("Predicted label:",predicted_label,". True label:", dataset.at[i, "converted_label"])
            # write the predicted label in the column "predicted_label" of the dataset
            dataset.at[i,"predicted_label_std_70B"] = predicted_label
            i += 1
    #save the dataset
    print("Total number of claims: ", i)
    dataset.to_csv(csv_file_path, index=False)

# DO THIS ONLY WHEN THE DATASET ITERATION PROCESS STOPS BEFORE THE END BUT YOU HAVE THE LOG FILE WITH THE PREDICTED LABELS
# IDEALLY, THE DATASET ITERATION PROCESS SHOULD HANDLE THE SAVING OF THE PREDICTED LABELS AND THIS IS NOT NEEDED
save_log_result("Logs/it_std_70B_log.txt", "Datasets/italian_150.csv")
ds = pd.read_csv("Datasets/italian_150.csv")
print(ds.columns)
print(ds.head())
print(ds["predicted_label_std_70B"].value_counts())