import os 
import tqdm
import openai
import pandas as pd
from jinja2 import Template
from config import OPENAI_API_KEY
from multiprocessing import Pool
from functools import partial
import time

# Set up OpenAI API key
openai.api_key = OPENAI_API_KEY 

CONVERSATION_START = """
This is the story: 
{{ story_content }}
"""

ZERO_SHOT_SYSTEM_PROMPT = """
Your expertise lies in discerning human-authored vs. AI-generated stories. 

Respond only with the answer 0 and 1 and nothing else. Think carefully.

Analyze the provided story and determine its origin: AI (1) or human (0). 

0 is human, 1 is AI.
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
    save_data(combined_stories, OUTPUT_FILE, nrows=250)

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

def run_experiment(data, prompt_func, output_file, result_column_name="result"):
    with Pool(processes=os.cpu_count()) as pool:
        results = []
        for i, result in enumerate(tqdm.tqdm(pool.imap(prompt_func, data['story_content']), total=len(data))):
            results.append(result)
            if (i+1) % 50 == 0:
                print("Sleeping for 5 seconds...")
                time.sleep(5)
    data[result_column_name] = results

    # Save the result
    save_data(data, output_file)

if __name__ == "__main__":
    data = load_data('datasets/experiment3/prompt_engineering.csv')
    run_experiment(data, zero_shot_prompt_result, 'datasets/experiment3/prompt_engineering_predictions_zeroshot.csv', result_column_name="zero_shot_result")
    run_experiment(data, cot_prompt_result, 'datasets/experiment3/prompt_engineering_predictions_cot.csv', result_column_name="cot_result")