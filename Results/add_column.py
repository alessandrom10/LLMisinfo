import pandas as pd

# Load the text.csv file and extract the column you need
csv_file = 'Results/test.csv'
csv_data = pd.read_csv(csv_file)

# Assuming the column you want to append is the first column, adjust as necessary
csv_column = csv_data.iloc[:, 0]  # Adjust index if the column is not the first

# Load the Excel file
excel_file = 'Datasets/results_english_150_before_2024_sample.xlsx'
excel_data = pd.read_excel(excel_file)

# Append the new column with the desired name
excel_data['predicted_react_70b'] = csv_column

# Save the updated Excel file
output_file = 'results_english_150_before_2024_sample_updated.csv'
excel_data.to_csv(output_file, index=False)

print(f"Column 'predicted_react_70b' appended successfully to {output_file}")
