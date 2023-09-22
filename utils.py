import pandas as pd
from jinja2 import Template

INPUT_PATH = 'datasets/hcV3-stories.csv'
OUTPUT_PATH = 'datasets/hcV3-imagined-stories.csv'

prompt_template_zero_shot = """
Given a short prompt summary, write an imagined journal entry about an event. 

Writing instructions:
- The story must correspond to the summary.
- Pretend the event happened to you, but do not write about something that actually happened to you.
- Write using first person perspective.
- Use the timeline of when the event happened (e.g., "3 weeks ago", "6 months ago").

Summary (from {{ time_since_event }} days ago):
{{ summary }}

Now, write the journal entry below.:
- **Reminder** Please make sure to write a story that corresponds to the summary written above.
- Don't write a story about something that actually happened to you.
- Story must be 15-25 sentences and 600-3000 characters including spaces.
- Do not start off with any salutations (e.g., "Dear Diary") or dates (e.g., 3/1/2021").
- Do not include the specific date or time of the event in the story.

Generated Story:
"""

def process_and_save_csv(input_path, output_path):

    # Load the CSV file
    data = pd.read_csv(input_path)

    # Filter rows where memType is "imagined"
    filtered_data = data[data['memType'] == 'imagined']

    # Keep only the specified columns
    filtered_data = filtered_data[['AssignmentId', 'story', 'summary', 'timeSinceEvent']]

    # Save the filtered dataset to a new CSV file
    filtered_data.to_csv(output_path, index=False)

    # Print out the number of rows in the filtered dataset
    print(f"The filtered dataset contains {filtered_data.shape[0]} rows.")
    print(f"Saved to {output_path}")

def generate_story_prompt(summary, time_since_event):
    template = Template(prompt_template_zero_shot)
    return template.render(summary=summary, time_since_event=time_since_event)

if __name__ == "__main__":
    process_and_save_csv(INPUT_PATH, OUTPUT_PATH)
    summary, time_since_event = "My boyfriend and I went to a concert together and had a great time. We met some of my friends there and really enjoyed ourselves watching the sunset.", 90.0
    print(generate_story_prompt(summary, time_since_event))
