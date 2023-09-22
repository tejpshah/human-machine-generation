import pandas as pd

INPUT_PATH = 'datasets/hcV3-stories.csv'
OUTPUT_PATH = 'datasets/filtered_data.csv'

def process_and_save_csv(input_path, output_path):

    # Load the CSV file
    data = pd.read_csv(input_path)

    # Filter rows where memType is "imagined"
    filtered_data = data[data['memType'] == 'imagined']

    # Keep only the specified columns
    filtered_data = filtered_data[['AssignmentId', 'story', 'summary']]

    # Save the filtered dataset to a new CSV file
    filtered_data.to_csv(output_path, index=False)

    # Print out the number of rows in the filtered dataset
    print(f"The filtered dataset contains {filtered_data.shape[0]} rows.")
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    process_and_save_csv(INPUT_PATH, OUTPUT_PATH)