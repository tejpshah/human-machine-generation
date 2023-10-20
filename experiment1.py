import argparse
import spacy
import ast 
import pandas as pd
from collections import Counter
from textstat import flesch_reading_ease
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

# Initialize global resources
NLP = spacy.load("en_core_web_sm")
ANALYZER = SentimentIntensityAnalyzer()

def extract_features(text):
    """Extract various linguistic features from the text."""
    
    # Tokenize the text using Spacy
    doc = NLP(text)
    
    # Compute various features
    char_freq = len(text)
    word_freq = len(doc)
    sentence_freq = len(list(doc.sents))
    unique_word_freq = len(set(token.text for token in doc))
    sentiment = ANALYZER.polarity_scores(text)['compound']
    flesch_score = flesch_reading_ease(text)
    sentence_lengths = [len(sentence) for sentence in doc.sents]
    avg_sentence_std_dev = pd.Series(sentence_lengths).std()

    # Get POS frequencies
    pos_counts = doc.count_by(spacy.attrs.POS)
    get_pos_count = lambda pos: pos_counts.get(pos, 0)
    noun_freq = get_pos_count(spacy.symbols.NOUN)
    verb_freq = get_pos_count(spacy.symbols.VERB)
    pronoun_freq = get_pos_count(spacy.symbols.PRON)
    adjective_freq = get_pos_count(spacy.symbols.ADJ)
    adverb_freq = get_pos_count(spacy.symbols.ADV)

    # Get punctuation frequencies
    exclamation_freq = text.count('!')
    ellipsis_freq = text.count('...')
    
    return {
        'char_freq': char_freq,
        'word_freq': word_freq,
        'sentence_freq': sentence_freq,
        'unique_word_freq': unique_word_freq,
        'flesch_score': flesch_score,
        'avg_sentence_std_dev': avg_sentence_std_dev,
        'noun_freq': noun_freq,
        'verb_freq': verb_freq,
        'pronoun_freq': pronoun_freq,
        'adjective_freq': adjective_freq,
        'adverb_freq': adverb_freq,
        'exclamation_freq': exclamation_freq,
        'ellipsis_freq': ellipsis_freq,
        'sentiment': sentiment
    }

def plot_features(data):
    """Plot percentage difference of features between human and AI generated texts."""
    
    human_features_df = pd.DataFrame(list(data['human_features']))
    ai_features_df = pd.DataFrame(list(data['ai_features']))
    test_significance(human_features_df, ai_features_df)

    human_means = human_features_df.mean()
    ai_means = ai_features_df.mean()
    percentage_diff = ((ai_means / human_means) - 1) * 100
    
    # Sort the percentage_diff in descending order
    percentage_diff_sorted = percentage_diff.sort_values(ascending=False)

    _, ax = plt.subplots(figsize=(14, 8))
    plt.rcParams["font.family"] = "Times New Roman"
    ax.bar(percentage_diff_sorted.index, percentage_diff_sorted, color='green')
    ax.set_ylabel('Percentage Difference (%)')
    ax.set_title('Percentage Difference of AI-generated Features Relative to Human-generated Features')
    ax.set_xticklabels(percentage_diff_sorted.index, rotation=45, ha="right")
    ax.axhline(y=0, color='red', linestyle='-')
    plt.tight_layout()
    plt.savefig(args.output_plot)
    plt.show()

def test_significance(human_features_df, ai_features_df):
    """Check the statistical significance of differences between human and AI generated features."""
    p_values = []
    for feature in human_features_df.columns:
        _, p_val = ttest_ind(human_features_df[feature], ai_features_df[feature], equal_var=False)
        p_values.append(p_val)

    significance_level = 0.01
    significant = [p < significance_level for p in p_values]
    
    p_values_df = pd.DataFrame({
        'Feature': human_features_df.columns,
        'p-value': p_values,
        'Significant': significant
    })
    
    # save p_values_df to csv
    p_values_df.to_csv(args.significance, index=False)

    return p_values_df

def string_to_dict(dict_string):
    """Converts a string representation of a dictionary into an actual dictionary."""
    try:
        return ast.literal_eval(dict_string)
    except ValueError:
        return {}

def main(args):
    data = None 
    if not args.preprocessed:
        data = pd.read_csv(args.input_file)
        data['human_features'] = data['story'].apply(extract_features)
        data['ai_features'] = data['generated_story'].apply(extract_features)
        data.to_csv(args.output_file, index=False)
    else:
        data = pd.read_csv(args.output_file)
        data['human_features'] = data['human_features'].apply(string_to_dict)
        data['ai_features'] = data['ai_features'].apply(string_to_dict)
    plot_features(data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and plot linguistic features from human and AI-generated texts.")
    parser.add_argument("--input_file", type=str, default="datasets/hcV3-imagined-stories-with-generated.csv", help="Path to input CSV file.")
    parser.add_argument("--output_plot", type=str, default="datasets/experiment1/experiment1.png", help="Path to save the output plot.")
    parser.add_argument("--output_file", type=str, default="datasets/experiment1/experiment1.csv", help="Path to save the output CSV with features.")
    parser.add_argument("--significance", type=str, default="datasets/experiment1/significance.csv", help="Runs t-tests on human vs AI generated features.")
    parser.add_argument("--preprocessed", action="store_true", help="Use this flag if the input file already contains features and there's no need to extract them again.")

    args = parser.parse_args()
    main(args)
