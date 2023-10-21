# Standard Library Imports
import argparse
import json
import os
from functools import partial
from multiprocessing import Pool

# Third-Party Libraries
import matplotlib.pyplot as plt
import numpy as np
import openai
import pandas as pd
import tqdm
from jinja2 import Template
from sklearn.metrics import confusion_matrix

# Local Imports
from config import OPENAI_API_KEY

### CONSTANTS ###

openai.api_key = OPENAI_API_KEY 

DATA_FILE = 'datasets/hcV3-imagined-stories-with-generated.csv'
OUTPUT_FILE = 'datasets/experiment3/prompt_engineering.csv'
HUMAN_STORY_COLUMN = 'story'
AI_STORY_COLUMN = 'generated_story'
STORY_COLUMN_NAME = 'story_content'
SAMPLE_FRACTION = 1
RANDOM_SEED = 42

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
0 is human, 1 is AI. Respond with either "0" or "1".
"""

### Setup Data for Experiment 3 ###

def load_data(file_path):
    '''Load data from a CSV file into a Pandas DataFrame.'''
    return pd.read_csv(file_path)

def prepare_data(data, original_column, new_column_name, label):
    '''Prepare data for the experiment by renaming a column and adding a label column.'''
    stories = data[[original_column]].rename(columns={original_column: new_column_name})
    stories['label'] = label
    return stories

def combine_and_shuffle(data1, data2, sample_fraction, seed):
    '''Combine and shuffle two DataFrames.'''
    combined_data = pd.concat([data1, data2], axis=0)
    return combined_data.sample(frac=sample_fraction, random_state=seed).reset_index(drop=True)

def save_data(data, file_path, nrows=None):
    '''Save data to a CSV file.'''
    subset_data = data.head(nrows) if nrows else data
    subset_data.to_csv(file_path, index=False)

def generate_data():
    '''Generate data for the experiment and save it to a CSV file.'''

    # Load the data
    data = load_data(args.data_file)

    # Prepare the data
    human_stories = prepare_data(data, HUMAN_STORY_COLUMN, STORY_COLUMN_NAME, 0)
    ai_stories = prepare_data(data, AI_STORY_COLUMN, STORY_COLUMN_NAME, 1)

    # Combine and shuffle the data
    combined_stories = combine_and_shuffle(human_stories, ai_stories, SAMPLE_FRACTION, RANDOM_SEED)

    # Save the first 250 rows of the data
    save_data(combined_stories, args.output_file, nrows=100)

### Prompt Engineering ###

def openai_request(system_prompt, user_prompt):
    '''Send a request to the OpenAI API and return the response.'''

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", 
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )
    return response['choices'][0]['message']['content']

def zero_shot_prompt_result(story):
    '''Run Zero-shot Prompt and return the result.'''
    system_prompt = Template(ZERO_SHOT_SYSTEM_PROMPT).render()
    user_prompt = Template(CONVERSATION_START).render(story_content=story)
    return int(openai_request(system_prompt, user_prompt))

def cot_prompt_result(story):
    '''Run COT Prompt and return the result.'''
    system_prompt = Template(COT_SYSTEM_PROMPT).render()
    user_prompt = Template(CONVERSATION_START).render(story_content=story)
    analysis = openai_request(system_prompt, user_prompt)
    output_prompt = Template(COT_OUTPUT_PARSER).render()
    return int(openai_request(output_prompt, analysis))

def self_consistency_prompt_result(story, num_samples=3):
    '''Runs COT Prompt multiple times and returns the most common result.'''
    results = [cot_prompt_result(story) for _ in range(num_samples)]
    return max(set(results), key=results.count)

### Run Experiments based on Prompt Engineering ###

def load_checkpoint(checkpoint_path):
    '''Load a checkpoint from a JSON file.'''
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'r') as f:
            return json.load(f)
    return None  # If no checkpoint exists, return None

def save_checkpoint(data, file_path):
    '''Save a checkpoint to a JSON file.'''
    with open(file_path, 'w') as f:
        # Convert the data to a list if it's a Series
        if isinstance(data, pd.Series):
            data = data.tolist()
        json.dump(data, f)

def run_experiment(data, prompt_func, output_file, checkpoint_path, result_column_name="result"):
    '''Run an experiment and save the results to a CSV file.'''

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

### Analysis - Compute Confidence Intervals ###

def compute_confidence_interval(p, n, z=1.96):
    '''Compute the 95% confidence interval for a proportion.'''
    margin_of_error = z * np.sqrt(p * (1 - p) / n)
    lower_bound = p - margin_of_error
    upper_bound = p + margin_of_error
    return lower_bound, upper_bound

### Command Line Interface ###

def parse_args():
    '''Parse command line arguments.'''
    parser = argparse.ArgumentParser(description="Run AI Experiments.")
    parser.add_argument('--generate', action='store_true', help='Generate data for experiments.')
    parser.add_argument('--zeroshot', action='store_true', help='Run Zero-shot prompt experiment.')
    parser.add_argument('--cot', action='store_true', help='Run COT prompt experiment.')
    parser.add_argument('--consistency', action='store_true', help='Run Self-consistency prompt experiment.')
    parser.add_argument('--data-file', default='datasets/hcV3-imagined-stories-with-generated.csv', help='Path to data file.')
    parser.add_argument('--output-file', default='datasets/experiment3/prompt_engineering.csv', help='Path for output file.')
    parser.add_argument('--checkpoint-zero-shot', default='datasets/experiment3/checkpoints/standard.json', help='Checkpoint path for zero-shot.')
    parser.add_argument('--checkpoint-cot', default='datasets/experiment3/checkpoints/cot.json', help='Checkpoint path for COT.')
    parser.add_argument('--checkpoint-consistency', default='datasets/experiment3/checkpoints/consistency.json', help='Checkpoint path for consistency.')
    parser.add_argument('--csv-zero-shot', default='datasets/experiment3/predictions/standard.csv', help='CSV path for zero-shot results.')
    parser.add_argument('--csv-cot', default='datasets/experiment3/predictions/cot.csv', help='CSV path for COT results.')
    parser.add_argument('--csv-consistency', default='datasets/experiment3/predictions/consistency.csv', help='CSV path for consistency results.')

    return parser.parse_args()

if __name__ == "__main__":

    openai.api_key = OPENAI_API_KEY 

    args = parse_args()
    
    data_path = args.data_file

    if args.generate:
        generate_data()
        
    data = load_data(data_path)
    
    if args.zeroshot:
        run_experiment(
            data, 
            zero_shot_prompt_result, 
            args.csv_zero_shot, 
            args.checkpoint_zero_shot, 
            result_column_name="zero_shot_result"
        )
    
    if args.cot:
        run_experiment(
            data, 
            cot_prompt_result, 
            args.csv_cot, 
            args.checkpoint_cot, 
            result_column_name="cot_result"
        )
    
    if args.consistency:
        run_experiment(
            data, 
            self_consistency_prompt_result, 
            args.csv_consistency, 
            args.checkpoint_consistency, 
            result_column_name="self_consistency_result"
        )
    