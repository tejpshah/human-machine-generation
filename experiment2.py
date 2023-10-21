# Standard libraries
import argparse
import pandas as pd

# External libraries
import openai 
import tiktoken
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

# Local imports
from config import OPENAI_API_KEY
from openai.embeddings_utils import get_embedding

# Constants
PLOT_SAVE_PATH = 'datasets/experiment2'
VALUE_PLOT_FILENAME = 'closer_to_story.png'
BOXPLOT_FILENAME = 'similarity_scores.png'
LABELS_UPDATED = ['Human Generated Story', 'GPT Generated Story']
COLORS = ["lightgreen", "darkgreen"]

# Set up OpenAI API key
openai.api_key = OPENAI_API_KEY 

class OpenAIEmbeddingsProcessor:
    def __init__(self, embedding_model="text-embedding-ada-002", embedding_encoding="cl100k_base", max_tokens=8000):
        self.embedding_model = embedding_model
        self.embedding_encoding = embedding_encoding
        self.max_tokens = max_tokens
        self.encoding = tiktoken.get_encoding(self.embedding_encoding)

    def check_token_count(self, text):
        return len(self.encoding.encode(text))

    def compute_embedding(self, text):
        return get_embedding(text, engine=self.embedding_model)

    def compute_cosine_similarity(self, embedding1, embedding2):
        return cosine_similarity([embedding1], [embedding2])[0][0]

    def compare_similarities(self, row):
        if row['story_summary_similarity'] > row['story_generated_story_similarity']:
            return 'Human Generated Story'
        elif row['story_summary_similarity'] < row['story_generated_story_similarity']:
            return 'AI Generated Story'
        else:
            return 'Equally Likely'

    def process_dataframe(self, df):
        # Filter out entries that are too long to embed
        df = df[df.apply(lambda x: self.check_token_count(x['story']) <= self.max_tokens and
                                   self.check_token_count(x['summary']) <= self.max_tokens and
                                   self.check_token_count(x['generated_story']) <= self.max_tokens, axis=1)]

        # Compute embeddings
        df['story_embedding'] = df['story'].apply(self.compute_embedding)
        df['summary_embedding'] = df['summary'].apply(self.compute_embedding)
        df['generated_story_embedding'] = df['generated_story'].apply(self.compute_embedding)

        # Compute cosine similarities
        df['story_summary_similarity'] = df.apply(lambda row: self.compute_cosine_similarity(row['story_embedding'], row['summary_embedding']), axis=1)
        df['story_generated_story_similarity'] = df.apply(lambda row: self.compute_cosine_similarity(row['story_embedding'], row['generated_story_embedding']), axis=1)

        # Compare similarities
        df['closer_to_story'] = df.apply(self.compare_similarities, axis=1)
        
        return df
def generate_value_plot(df):
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='closer_to_story', palette=COLORS)
    plt.title('Count of Stories With Higher Cosine Similarity to Original Story')
    plt.ylabel('Count')
    plt.xlabel('')
    plt.xticks(ticks=range(len(LABELS_UPDATED)), labels=reversed(LABELS_UPDATED))
    plt.savefig(f"{PLOT_SAVE_PATH}/{VALUE_PLOT_FILENAME}")
    plt.show()

def generate_boxplot(df):
    data_to_plot = [df['story_generated_story_similarity'], df['story_summary_similarity']]
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=data_to_plot, orient="v", palette=COLORS, width=0.6)
    plt.xticks(ticks=range(len(LABELS_UPDATED)), labels=reversed(LABELS_UPDATED))
    plt.ylabel('Similarity Score')
    plt.title('Distribution of Similarity Scores')
    plt.savefig(f"{PLOT_SAVE_PATH}/{BOXPLOT_FILENAME}")
    plt.show()

def plot(df, plot_type):
    if plot_type == 'value_plot':
        generate_value_plot(df)
    elif plot_type == 'boxplot':
        generate_boxplot(df)
    else:
        print("Invalid plot_type. Use 'value_plot' or 'boxplot'.")

def main(args):
    df_processed = None
    if not args.preprocessing:
        # Load data
        df = pd.read_csv(args.input_path)

        # Process data
        processor = OpenAIEmbeddingsProcessor()
        df_processed = processor.process_dataframe(df)

        # only keep relevant columns
        relevant_cols = [
            'story_embedding', 'summary_embedding', 'generated_story_embedding',
            'story_summary_similarity', 'story_generated_story_similarity', 'closer_to_story'
        ]
        df_processed = df_processed[relevant_cols]
        print(df_processed['closer_to_story'].value_counts())

        # Save the updated dataframe
        df_processed.to_csv(args.output_path, index=False)
    else:
        df_processed = pd.read_csv(args.output_path)
        
    # Generate plots
    print(df_processed['closer_to_story'].value_counts())
    plot(df_processed, 'value_plot')
    plot(df_processed, 'boxplot')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process embeddings and plot data")
    parser.add_argument("--input_path", type=str, default="datasets/hcV3-imagined-stories-with-generated.csv")
    parser.add_argument("--output_path", type=str, default="datasets/experiment2/embeddings.csv")
    parser.add_argument("--preprocessing", type=bool, default=False)
    args = parser.parse_args()
    
    main(args)