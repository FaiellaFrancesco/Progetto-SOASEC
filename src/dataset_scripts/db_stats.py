import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
from tqdm import tqdm


def process_stats(filepath, output_dir, sample_frac):
    print(f"Loading dataset from {filepath}...")
    df = pd.read_parquet(filepath)

    global_edge_counts = np.zeros(4)
    global_recency = []
    global_sampled_data = {'n_remaining': [], 'puzzle_n': [], 'rating': [],
                           'think_time': [], 'legal_moves': [], 'density': []}

    # Edge Type Composition (Aggregate all)
    if 'edge_attr' in df.columns:
        print("Calculating edge composition...")
        for attr in tqdm(df['edge_attr'].dropna(), desc="Edge Attr"):
            global_edge_counts += np.sum(np.vstack(attr), axis=0)

    # Edge Recency (Sampled)
    if 'edge_time' in df.columns:
        print("Calculating edge recency...")
        for et in tqdm(df['edge_time'].dropna().sample(frac=sample_frac), desc="Edge Time"):
            global_recency.extend(et)

    # Branching, Timing, and Density (Sampled)
    print("Sampling data for correlations...")
    for _, row in tqdm(df.sample(frac=sample_frac).iterrows(), desc="Correlations", total=int(len(df)*sample_frac)):
        global_sampled_data['n_remaining'].append(row.get('n_remaining'))
        global_sampled_data['puzzle_n'].append(row.get('puzzle_n'))
        global_sampled_data['rating'].append(row.get('rating'))
        global_sampled_data['think_time'].append(row.get('think_time'))

        legal = row.get('legal_moves')
        global_sampled_data['legal_moves'].append(
            len(legal) if legal is not None else 0)

        edge_idx = row.get('edge_index')
        if edge_idx is not None and len(edge_idx) > 0:
            edge_arr = np.array(edge_idx)
            if edge_arr.ndim >= 2:
                edges = edge_arr.shape[1]
            else:
                edges = 0
            global_sampled_data['density'].append(edges / 64.0)
        else:
            global_sampled_data['density'].append(0.0)

    # --- Plotting ---
    print("\nGenerating charts...")
    plot_edge_composition(global_edge_counts, output_dir)
    plot_recency(global_recency, output_dir)

    df_plot = pd.DataFrame(global_sampled_data)
    # Only drop rows if essential correlation data is missing
    df_plot = df_plot.dropna(
        subset=['n_remaining', 'puzzle_n', 'rating', 'legal_moves', 'density'])

    if not df_plot.empty:
        plot_correlations(df_plot, output_dir,
                          has_think_time=df_plot['think_time'].notna().any())
    else:
        print("Not enough data to plot correlations.")

    print(f"Analysis complete! Graphs saved to {output_dir}")


def plot_edge_composition(counts, output_dir):
    plt.figure(figsize=(8, 5))
    sns.barplot(x=["Attacks", "Defends", "Moves", "Pushes"],
                y=counts, palette='Set2')
    plt.title('Global Edge Type Composition')
    plt.ylabel('Total Edges')
    plt.savefig(os.path.join(output_dir, 'edge_composition.png'))
    plt.close()


def plot_recency(recency, output_dir):
    plt.figure(figsize=(10, 6))
    sns.histplot(recency, bins=21, color='purple', kde=False)
    plt.title('Edge Recency Distribution (20 = Unknown/Never Moved)')
    plt.xlabel('Plies since last move')
    plt.savefig(os.path.join(output_dir, 'edge_recency.png'))
    plt.close()


def plot_correlations(df, output_dir, has_think_time=True):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Simulated Think Time
    if has_think_time:
        sns.lineplot(data=df, x='n_remaining',
                     y='think_time', ax=axes[0], color='red')
    else:
        axes[0].text(0.5, 0.5, 'think_time data missing',
                     ha='center', va='center')
    axes[0].set_title('Simulated Think Time vs. Mate Depth')

    # Plot 2: Legal Moves vs Rating & Depth
    df = df.sort_values('rating')
    df['rating_bins'] = pd.cut(df['rating'], bins=10).astype(str)
    sns.lineplot(data=df, x='rating_bins', y='legal_moves',
                 hue='n_remaining', ax=axes[1], palette='viridis')
    axes[1].set_title('Action Space vs. Rating & Mate Depth')
    axes[1].tick_params(axis='x', rotation=45)

    # Plot 3: Graph Density by Puzzle Length
    density_grouped = df.groupby(['puzzle_n', 'n_remaining'])[
        'density'].mean().reset_index()
    sns.lineplot(data=density_grouped, x='n_remaining', y='density',
                 hue='puzzle_n', ax=axes[2], palette='tab10')
    axes[2].set_title('Graph Density vs. Mate Depth')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'depth_correlations.png'))
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract and graph stats from chunked chess puzzle dataset.")
    parser.add_argument("-i", "--input_file", type=str,
                        required=True, help="Path to the aggregated parquet file")
    parser.add_argument("-o", "--output_dir", type=str,
                        required=True, help="Directory for output graphs")
    parser.add_argument("--sample_frac", type=float, default=0.1,
                        help="Fraction of rows to sample per chunk (default: 0.1)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    process_stats(args.input_file, args.output_dir, args.sample_frac)
