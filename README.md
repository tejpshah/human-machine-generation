# human-machine-generation
Tej's CS598 Final Project: Human Generations vs Machine Generations

# Classifying Human vs AI Generated Stories

This repository contains code and data for experiments classifying fictional stories as human or AI generated. It leverages the Hippocorpus dataset. 

This paper investigates the differentiation between human and AI-generated narratives using the augmented Hippocorpus dataset, which includes 2,756 stories generated by GPT-3.5. We analyze linguistic features such as noun frequency and Flesch reading scores and use logistic regression to identify the provenance of the stories. Additionally, we assess the alignment of stories with their original summaries through embeddings and cosine similarity measures. Our findings reveal that linguistic features can distinguish between human and AI-generated stories with high accuracy, and that GPT-generated stories align more closely with the intent of original summaries than human-written stories. The study also evaluates the efficacy of different prompting techniques and the impact of few-shot learning on AI-generated narratives. Surprisingly, few-shot learning enhanced the GPT's alignment with summary intent, challenging the assumption that human examples would lead to greater variability in AI output. The paper discusses the limitations of current methodologies and proposes directions for future research, including the exploration of different LLMs and the refinement of prompt engineering techniques.

## Repository Structure

```
├── README.md
├── config.py # Configuration settings
├── datasets # Input and output datasets
│   ├── experiment1 # Data for linguistic feature analysis
│   ├── experiment2 # Data for semantic similarity analysis 
│   ├── experiment3 # Data for prompt engineering experiments
│   ├── hcV3-imagined-stories.csv # Human generated stories
│   ├── hcV3-imagined-stories-with-generated.csv # GPT generated stories (zero-shot)
│   └── hcV3-imagined-stories-with-generated-few-shot.csv # GPT generated stories (few-shot)
├── generate_dataset.py # Script to generate GPT stories
├── experiment1_analysis.py # Linguistic feature analysis
├── experiment1_classifier.py # Logistic regression classifier
├── experiment2_analysis.py # Semantic similarity analysis
├── experiment2_classifier.py # Embedding based classifier 
├── experiment3_analysis.py # Prompt engineering analysis
└── requirements.txt # Python package dependencies
```

## Experiments

The repository contains code to reproduce the following experiments:

1. **Linguistic Feature Analysis**: Compares lexical features between human and GPT generated stories. Trains a logistic regression classifier to categorize stories based on linguistic signals.

2. **Semantic Similarity Analysis**: Computes semantic embeddings for stories using OpenAI's `text-embedding-ada-002` and compares cosine similarity. Trains an embedding based classifier. 

3. **Prompt Engineering**: Evaluates different prompting strategies (e.g. chain of thought, self-consistency) for enabling GPT-3.5 to accurately discern human vs AI generated stories.

The analysis is performed for both zero-shot and few-shot scenarios. See the [report](tej-cs598-final-report.pdf) for details on the experimental setup, results, and conclusions.

## Usage

To replicate the experiments:

1. Install dependencies: `pip install -r requirements.txt`
2. Generate GPT stories (`generate_dataset.py`) 
3. Run analysis scripts for experiments 1-3

Output datasets, classification results, and plots will be saved to the `datasets` and `results` folders.