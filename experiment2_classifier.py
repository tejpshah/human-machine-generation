import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

def load_data(filepath):
    """Load data from the given CSV file."""
    return pd.read_csv(filepath)

def preprocess_data(df):
    """
    Preprocess data by extracting embeddings and labels.
    """
    human_embeddings = df['summary_embedding'].apply(eval).tolist()
    human_labels = [0] * len(human_embeddings)

    ai_embeddings = df['generated_story_embedding'].apply(eval).tolist()
    ai_labels = [1] * len(ai_embeddings)

    all_embeddings = human_embeddings + ai_embeddings
    all_labels = human_labels + ai_labels

    df_new = pd.DataFrame({
        'story_embedding': all_embeddings,
        'type': all_labels
    })
    return df_new

def split_data(df):
    """
    Split data into training and test sets.
    """
    X = pd.DataFrame(df['story_embedding'].tolist())
    y = df['type']
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_and_evaluate(x_train, y_train, x_test, y_test):
    """
    Train and evaluate the models.
    """
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    }

    results = {}
    for model_name, model in models.items():
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        results[model_name] = (accuracy, report)

    return results

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train and evaluate a model on story embeddings.")
    parser.add_argument("--filepath", type=str, default='datasets/experiment2/experiment2-embeddings-few-shot.csv',help="Path to the dataset containing embeddings.")
    return parser.parse_args()

def main():
    '''Main function.'''
    args = parse_arguments()

    df = load_data(args.filepath)
    df_new = preprocess_data(df)
    x_train, x_test, y_train, y_test = split_data(df_new)
    results = train_and_evaluate(x_train, y_train, x_test, y_test)
    print(results)

if __name__ == '__main__':
    main()
