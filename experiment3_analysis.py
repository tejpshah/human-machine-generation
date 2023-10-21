import pandas as pd

def load_data(file_path):
    return pd.read_csv(file_path)

def prepare_data(data, original_column, new_column_name, label):
    stories = data[[original_column]].rename(columns={original_column: new_column_name})
    stories['label'] = label
    return stories

def combine_and_shuffle(data1, data2, sample_fraction, seed):
    combined_data = pd.concat([data1, data2], axis=0)
    return combined_data.sample(frac=sample_fraction, random_state=seed).reset_index(drop=True)

def save_data(data, file_path, nrows=None):
    subset_data = data.head(nrows) if nrows else data
    subset_data.to_csv(file_path, index=False)

def generate_data():
    # Define constants
    DATA_FILE = 'datasets/hcV3-imagined-stories-with-generated.csv'
    OUTPUT_FILE = 'datasets/experiment3/prompt_engineering.csv'
    HUMAN_STORY_COLUMN = 'story'
    AI_STORY_COLUMN = 'generated_story'
    STORY_COLUMN_NAME = 'story_content'
    SAMPLE_FRACTION = 1
    RANDOM_SEED = 42

    # Load the data
    data = load_data(DATA_FILE)

    # Prepare the data
    human_stories = prepare_data(data, HUMAN_STORY_COLUMN, STORY_COLUMN_NAME, 0)
    ai_stories = prepare_data(data, AI_STORY_COLUMN, STORY_COLUMN_NAME, 1)

    # Combine and shuffle the data
    combined_stories = combine_and_shuffle(human_stories, ai_stories, SAMPLE_FRACTION, RANDOM_SEED)

    # Save the first 250 rows of the data
    save_data(combined_stories, OUTPUT_FILE, nrows=250)

if __name__ == "__main__":
    # generate_data()
    pass 
