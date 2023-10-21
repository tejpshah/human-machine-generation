import os 
import tqdm
import openai
import json
import pandas as pd
from jinja2 import Template
from config import OPENAI_API_KEY
from multiprocessing import Pool
from functools import partial
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import time

# Set up OpenAI API key
openai.api_key = OPENAI_API_KEY 

CONVERSATION_START = """
This is the story: 
{{ story_content }}
"""

ZERO_SHOT_SYSTEM_PROMPT = """
Your expertise lies in discerning human-authored vs. AI-generated stories. 

Half the stories shown are human-authored and half are AI-generated.

Think carefully. Analyze the provided story and determine its origin: AI (1) or human (0). 

Answer "1" if its AI generated and "0" if its human authored. Do not give any other answer.
"""

COT_SYSTEM_PROMPT = """
Your expertise lies in discerning human-authored vs. AI-generated stories.

Analyze the provided story and determine its origin: AI ("AI") or human ("Human").

Firstly, briefly state your criteria for distinguishing AI from human narratives. Next, contrast the story's elements, specifying which lean toward AI or human characteristics. Your comparative analysis is essential. Conclude with: "Therefore, the likely answer is [AI/Human]".

Your detailed yet succinct evaluation is valued. Let's think step by step. 
"""

COT_OUTPUT_PARSER = """
Read the evaluation and determine whether the analysis suggests that the story is AI or human.

Respond only with the answer 0 and 1 and nothing else. 

0 is human, 1 is AI.
"""

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
    save_data(combined_stories, OUTPUT_FILE, nrows=100)

def zero_shot_prompt_result(story):
    system_prompt = Template(ZERO_SHOT_SYSTEM_PROMPT).render()
    user_prompt = Template(CONVERSATION_START).render(story_content=story)
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", 
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )
    return int(response['choices'][0]['message']['content'])

def cot_prompt_result(story):
    system_prompt = Template(COT_SYSTEM_PROMPT).render()
    user_prompt = Template(CONVERSATION_START).render(story_content=story)
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", 
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )

    analysis = response['choices'][0]['message']['content']

    output_prompt = Template(COT_OUTPUT_PARSER).render()
    answer = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", 
        messages=[
            {"role": "system", "content": output_prompt},
            {"role": "user", "content": analysis},
        ]
    )
    return int(answer['choices'][0]['message']['content'])

def self_consistency_prompt_result(story, num_samples=3):
    """Run COT_Prompt 3 and take majority vote of the 3 results"""
    results = [cot_prompt_result(story) for _ in range(num_samples)]
    return max(set(results), key=results.count)

def load_checkpoint(checkpoint_path):
    """Load interim results from a checkpoint file."""
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'r') as f:
            return json.load(f)
    return None  # If no checkpoint exists, return None

def save_checkpoint(data, file_path):
    with open(file_path, 'w') as f:
        # Convert the data to a list if it's a Series
        if isinstance(data, pd.Series):
            data = data.tolist()
        json.dump(data, f)

def run_experiment(data, prompt_func, output_file, checkpoint_path, result_column_name="result"):
    # Load checkpoint if it exists
    checkpoint_data = load_checkpoint(checkpoint_path)
    start_index = 0

    if checkpoint_data is not None:
        # Update the DataFrame only for the indices that have checkpoint data
        for i, val in enumerate(checkpoint_data):
            data.at[i, result_column_name] = val
        start_index = len(checkpoint_data)

    results = list(data.get(result_column_name, [None] * len(data)))  # Initialize with previous results or None

    # Use 'with' to ensure processes are cleaned up promptly
    with Pool(processes=8) as pool:
        # Wrap with list() to ensure results are retrieved before closing the pool
        for i, result in enumerate(tqdm.tqdm(pool.imap(prompt_func, data['story_content'][start_index:]), total=len(data) - start_index), start=start_index):
            results[i] = result

            # Save a checkpoint after each result
            if (i + 1) % 10 == 0:  # Save every 10 results; you can adjust this number
                save_checkpoint(results[:i + 1], checkpoint_path)

        # Save final checkpoint
        save_checkpoint(results, checkpoint_path)

    # Assign results back to data and save it
    data[result_column_name] = results
    save_data(data, output_file)

def generate_combined_plot(zeroshot_path, cot_path, self_consistency_path):
    """
    Generates a combined 2x3 plot visualizing correct and incorrect guesses for both "AI Predicted" 
    and "Human Predicted" labels across three prompting techniques using data from provided CSV files.
    
    Args:
    - zeroshot_path (str): Path to the CSV file with Zero-shot predictions.
    - cot_path (str): Path to the CSV file with CoT predictions.
    - self_consistency_path (str): Path to the CSV file with Self-consistency predictions.
    """
    labels = ['Correct', 'Incorrect']
    techniques = ['Zero-shot', 'CoT', 'Self-consistency']

    # Load the data from provided CSV paths
    zeroshot_df = pd.read_csv(zeroshot_path)
    cot_df = pd.read_csv(cot_path)
    self_consistency_df = pd.read_csv(self_consistency_path)
    
    # Compute confusion matrices for each technique
    zeroshot_cm = confusion_matrix(zeroshot_df['label'], zeroshot_df['zero_shot_result'])
    cot_cm = confusion_matrix(cot_df['label'], cot_df['cot_result'])
    self_consistency_cm = confusion_matrix(self_consistency_df['label'], self_consistency_df['self_consistency_result'])

    # Extracting counts from confusion matrices for plotting
    ai_guess_zeroshot = [zeroshot_cm[1][1], zeroshot_cm[0][1]]  # TP, FP
    human_guess_zeroshot = [zeroshot_cm[0][0], zeroshot_cm[1][0]]  # TN, FN
    ai_guess_cot = [cot_cm[1][1], cot_cm[0][1]]  # TP, FP
    human_guess_cot = [cot_cm[0][0], cot_cm[1][0]]  # TN, FN
    ai_guess_self_consistency = [self_consistency_cm[1][1], self_consistency_cm[0][1]]  # TP, FP
    human_guess_self_consistency = [self_consistency_cm[0][0], self_consistency_cm[1][0]]  # TN, FN

    # Plotting in a single 2x3 grid
    plt.figure(figsize=(15, 8))

    # AI Guesses
    for i, (ai_guess, technique) in enumerate(zip([ai_guess_zeroshot, ai_guess_cot, ai_guess_self_consistency], techniques), 1):
        plt.subplot(2, 3, i)
        plt.bar(labels, ai_guess, color=['g', 'r'])
        plt.title(f'AI Guesses ({technique})')
        plt.ylabel('Count')
        plt.grid(False)

    # Human Guesses
    for i, (human_guess, technique) in enumerate(zip([human_guess_zeroshot, human_guess_cot, human_guess_self_consistency], techniques), 4):
        plt.subplot(2, 3, i)
        plt.bar(labels, human_guess, color=['g', 'r'])
        plt.title(f'Human Guesses ({technique})')
        plt.ylabel('Count')
        plt.grid(False)

    plt.tight_layout()
    plt.savefig('datasets/experiment3/combined_plot.png')
    plt.show()

if __name__ == "__main__":

    checkpoint_path_zero_shot = 'datasets/experiment3/checkpoint_zeroshot.json'
    checkpoint_path_cot = 'datasets/experiment3/checkpoint_cot.json'
    checkpoint_path_consistency = 'datasets/experiment3/checkpoint_consistency.json'

    # generate_data()
    # data = load_data('datasets/experiment3/prompt_engineering.csv')

    # Zero-shot prompt experiment
    # run_experiment(data, zero_shot_prompt_result, 'datasets/experiment3/prompt_engineering_predictions_zeroshot.csv', checkpoint_path_zero_shot, result_column_name="zero_shot_result")

    # COT prompt experiment
    # run_experiment(data, cot_prompt_result, 'datasets/experiment3/prompt_engineering_predictions_cot.csv', checkpoint_path_cot, result_column_name="cot_result")

    # Self-consistency prompt experiment
    # run_experiment(data, self_consistency_prompt_result, 'datasets/experiment3/prompt_engineering_predictions_self_consistency.csv', checkpoint_path_consistency, result_column_name="self_consistency_result")

    # Generate combined plot
    generate_combined_plot('datasets/experiment3/prompt_engineering_predictions_zeroshot.csv', 'datasets/experiment3/prompt_engineering_predictions_cot.csv', 'datasets/experiment3/prompt_engineering_predictions_self_consistency.csv')
