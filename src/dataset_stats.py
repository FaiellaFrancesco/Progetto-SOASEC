import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
import argparse
import os


def load_data(file_path):
    """Loads the parquet dataset."""
    return pd.read_parquet(file_path)


def plot_mates_in_distribution(df, output_dir):
    """Plots the distribution of the 'y' column (MatesIn)."""
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='n_remaining', palette='viridis')
    plt.title('Distribution of MatesIn (y)')
    plt.xlabel('Mate In (Moves)')
    plt.ylabel('Frequency')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, 'mates_in_distribution.png'))
    plt.close()


def plot_rating_distribution(df, output_dir):
    """Plots the rating distribution in 50 splits."""
    plt.figure(figsize=(10, 6))
    sns.histplot(df['rating'], bins=50, color='coral', kde=True)
    plt.title('Chess Puzzle Rating Distribution')
    plt.xlabel('Rating')
    plt.ylabel('Count')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, 'rating_distribution.png'))
    plt.close()


def calculate_and_plot_graph_similarity(df, output_dir):
    """
    Approximates board similarity for the same MateIn.
    Aggregates node features (x) to create a graph-level embedding,
    then calculates average cosine similarity within each MateIn group.
    """
    # Create graph-level embeddings by taking the mean of node features (x)
    # Assuming 'x' is a list/array of node features for each board
    df['graph_embedding'] = df['x'].apply(lambda nodes: np.mean(
        np.vstack(nodes), axis=0) if nodes is not None else None)

    # Drop rows where embeddings couldn't be calculated
    valid_df = df.dropna(subset=['graph_embedding'])

    mate_categories = valid_df['y'].unique()
    avg_similarities = {}

    for mate_val in mate_categories:
        group = valid_df[valid_df['y'] == mate_val]
        if len(group) > 1:
            embeddings = np.stack(group['graph_embedding'].values)
            sim_matrix = cosine_similarity(embeddings)

            # Extract upper triangle without diagonal to get pairwise similarities
            upper_tri = sim_matrix[np.triu_indices(sim_matrix.shape[0], k=1)]
            avg_similarities[mate_val] = np.mean(upper_tri)
        else:
            avg_similarities[mate_val] = 0.0

    # Plotting the similarity results
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(avg_similarities.keys()), y=list(
        avg_similarities.values()), palette='magma')
    plt.title('Average Board Similarity by MatesIn Category')
    plt.xlabel('Mate In (y)')
    plt.ylabel('Average Cosine Similarity')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, 'graph_similarity.png'))
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract and graph stats from chess puzzle dataset.")
    parser.add_argument("-i", "--input", type=str, required=True,
                        help="Path to the input parquet file (e.g., merged.parquet)")
    parser.add_argument("-o", "--output", type=str, required=True,
                        help="Path to the output directory for graphs")
    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    try:
        df = load_data(args.input)
        plot_mates_in_distribution(df, args.output)
        plot_rating_distribution(df, args.output)
        calculate_and_plot_graph_similarity(df, args.output)
        print(f"Graphs successfully saved to {args.output}")
    except Exception as e:
        print(f"Error processing dataset: {e}")
