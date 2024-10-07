import csv

# Function to load a txt file and write its content into a csv file
def txt_to_csv(txt_file, csv_file):
    # Open the txt file and read its lines
    with open(txt_file, 'r') as file:
        lines = file.readlines()

    for i, line in enumerate(lines):
        lines[i] = line[94:].lower().replace(".","")

    # Open the csv file in write mode
    with open(csv_file, 'w', newline='') as csvfile:
        # write with colum name specified by column_name
        column_name = 'claim'
        
        csvwriter = csv.writer(csvfile)
        
        # Write each line from txt to a row in the csv
        for line in lines:
            csvwriter.writerow([line.strip()])  # .strip() removes newline characters

# Example usage
txt_to_csv('Results/claims_es_answers_70b.txt', 'Results/spanish.csv')
