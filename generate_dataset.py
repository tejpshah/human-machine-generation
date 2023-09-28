import openai 
import os 
import pandas as pd
from jinja2 import Template
from config import OPENAI_API_KEY
from multiprocessing import Pool, cpu_count, current_process
from tqdm import tqdm 

# Set up OpenAI API key
openai.api_key = OPENAI_API_KEY 

# Define file paths
input_path = 'datasets/hcV3-stories.csv'
output_path = 'datasets/hcV3-imagined-stories.csv'

# Define prompt templates
system_prompt_template = """
Given a short prompt summary, write an imagined journal entry about an event. Writing instructions:
- The story must correspond to the summary. 
- Pretend the event happened to you, but do not write about something that actually happened to you.
- Write using first person perspective.  
- Use the timeline of when the event happened (e.g., "3 weeks ago", "6 months ago").

Now, write the journal entry below.:
- **Reminder** Please make sure to write a story that corresponds to the summary written above. 
- Don't write a story about something that actually happened to you.
- Story must be 15-25 sentences and 600-3000 characters including spaces.  
- Do not start off with any salutations (e.g., "Dear Diary") or dates (e.g., 3/1/2021").
- Do not include the specific date or time of the event in the story.
"""

user_prompt_template = """
Summary (from {{ time_since_event }} days ago): {{ summary }}

Generated Story:
"""

def filter_and_save_dataset(input_path, output_path):
  
  # Load data
  data = pd.read_csv(input_path)
  
  # Filter for 'imagined' stories
  imagined_stories = data[data['memType'] == 'imagined']
  
  # Keep only relevant columns
  kept_columns = ['AssignmentId', 'story', 'summary', 'timeSinceEvent']
  imagined_stories = imagined_stories[kept_columns]
  
  # Save filtered data to new CSV
  imagined_stories.to_csv(output_path, index=False)

  print(f"Saved {imagined_stories.shape[0]} rows to {output_path}")

def generate_story(summary, time_since_event):

  # Render prompt templates
  system_prompt = Template(system_prompt_template).render()
  user_prompt = Template(user_prompt_template).render(
    summary=summary,
    time_since_event=time_since_event
  )

  # Generate story
  response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo", 
    messages=[
      {"role": "system", "content": system_prompt},
      {"role": "user", "content": user_prompt},
    ]
  )

  return response['choices'][0]['message']['content']

def generate_stories_for_row(row):

    # Retrieve details from the row
    assignment_id = row['AssignmentId']
    summary = row['summary']
    time_since_event = row['timeSinceEvent']

    # Print the processing status and the core it's being processed on
    print(f"Processing {assignment_id} on {current_process().name}")

    # Generate the story using the summary and time since event
    generated_story = generate_story(summary, time_since_event)

    return assignment_id, generated_story

def generate_all_stories(data):
    num_cpus = cpu_count()
    print(f"Using {num_cpus} CPU cores for processing.")
    
    # Check if temp file exists and load it if it does
    temp_file_path = 'datasets/temp_generated_stories.csv'
    if os.path.exists(temp_file_path):
        temp_df = pd.read_csv(temp_file_path)
        completed_assignment_ids = set(temp_df['AssignmentId'])
        # Filter the data to only include rows not already processed
        data = data[~data['AssignmentId'].isin(completed_assignment_ids)]
        results = temp_df.values.tolist()
    else:
        results = []

    # Update progress bar total based on remaining data
    pbar = tqdm(total=len(data), desc="Progress")
    
    # Save periodically after processing each row
    with Pool(num_cpus) as pool:
        for assignment_id, generated_story in pool.imap_unordered(generate_stories_for_row, [row for _, row in data.iterrows()]):
            results.append((assignment_id, generated_story))
            
            # Update the progress bar
            pbar.update(1)
            
            # Save results to CSV every 250 rows processed
            if len(results) % 20 == 0:
                temp_df = pd.DataFrame(results, columns=['AssignmentId', 'generated_story'])
                temp_df.to_csv(temp_file_path, index=False)
    
    # Close the progress bar
    pbar.close()
    
    # Save the final results to a new dataframe and return it
    return pd.DataFrame(results, columns=['AssignmentId', 'generated_story'])

if __name__ == "__main__":
  
    # Load imagined stories
    imagined_stories = pd.read_csv("datasets/hcV3-imagined-stories.csv")

    # Generate stories for all rows in the dataframe
    stories_df = generate_all_stories(imagined_stories)

    # Merge the original dataframe with the generated stories on AssignmentId
    merged_df = imagined_stories.merge(stories_df, on="AssignmentId")

    # Save the updated dataframe with the generated stories to a new CSV file
    merged_df.to_csv("datasets/hcV3-imagined-stories-with-generated.csv", index=False)

    print(f"Saved {merged_df.shape[0]} rows to datasets/hcV3-imagined-stories-with-generated.csv")