import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# ---------------------------- SETTINGS ----------------------------

plt.rcParams.update({'font.size': 12})
MAX_CLUSTERS = 10
OPTIMAL_K = 3
ENCODING = 'utf-8-sig'
DATA_PATH = './Task_1/results.csv'
OUTPUT_DIR = './Task_3'


# ---------------------------- FUNCTIONS ----------------------------

def load_and_prepare_data():
    """Load dataset and return standardized numeric values"""
    df = pd.read_csv(DATA_PATH, encoding=ENCODING)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if not num_cols:
        raise ValueError("No numeric data found in the dataset.")

    imputer = SimpleImputer(strategy='mean')
    df[num_cols] = imputer.fit_transform(df[num_cols])

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[num_cols])
    return df, num_cols, scaled_data


def display_elbow_chart(inertia_vals, max_k, best_k):
    plt.figure(figsize=(10, 7))
    k_values = range(1, max_k + 1)
    plt.plot(k_values, inertia_vals, 'o-b', ms=5, lw=2)
    plt.axvline(best_k, color='orangered', linestyle='--', label=f'Optimal k = {best_k}')
    plt.annotate(
        'Elbow Point',
        xy=(best_k, inertia_vals[best_k - 1]),
        xytext=(best_k + 1.5, inertia_vals[best_k - 1] + (inertia_vals[0] - inertia_vals[-1]) * 0.08),
        arrowprops=dict(arrowstyle='->', color='gray')
    )
    plt.title('Elbow Method - Inertia vs. Number of Clusters')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/Elbow_plot.png')
    # plt.show()
    plt.close()


def apply_kmeans(data, k, seed=42):
    kmeans = KMeans(n_clusters=k, random_state=seed)
    labels = kmeans.fit_predict(data)
    return kmeans, labels+1


def cluster_summary(df, labels, columns):
    """Add cluster labels and print cluster-wise means"""
    df['Cluster'] = labels
    summary_stats = df.groupby('Cluster')[columns].mean().round(2)
    print("\nCluster Mean Summary:")
    print(summary_stats)
    return df


def visualize_clusters(data, labels):
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(data)
    reduced_df = pd.DataFrame(reduced, columns=['PCA1', 'PCA2'])
    reduced_df['Cluster'] = labels

    # Tính centroid của mỗi cluster
    centroids = reduced_df.groupby('Cluster')[['PCA1', 'PCA2']].mean().values

    plt.figure(figsize=(10, 7))
    for label in sorted(reduced_df['Cluster'].unique()):
        subset = reduced_df[reduced_df['Cluster'] == label]
        plt.scatter(subset['PCA1'], subset['PCA2'], s=80, alpha=0.7, label=f'Cluster {label+1}')
    
    # Vẽ centroid
    plt.scatter(centroids[:,0], centroids[:,1], s=100, c='purple', marker='*', label='Centroids')

    plt.title('2D PCA Cluster Visualization with Centroids')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/PCA_2D_Clusters_Plot.png')
    # plt.show()
    plt.close()


def print_cluster_commentary():
    """Translated commentary on clustering results (Modify your comments here!!!)"""
    print("\nCommentary:\n")
    print("K=3 was chosen for KMeans based on the Elbow Method, where inertia reduction slows significantly after k = 3.")

    print("Cluster 1: includes primarily defensive players such as center-backs, full-backs, and defensive midfielders, " \
    "characterized by strong tackling and interception stats, but low attacking output.")
    
    print("Cluster 2: represents supporting players or squad players with balanced but modest contributions." \
    "These may be midfielders or rotational players with average impact in both attack and defense.")
    
    print("Cluster 3: identifies attacking players such as strikers and wingers, who contribute " \
    "significantly to goals and assists but engage less in defensive duties.")


# ---------------------------- MAIN PROCESS ----------------------------

def main():
    try:
        df, num_cols, scaled_data = load_and_prepare_data()
        print("✅ Data successfully loaded and preprocessed.")
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        return

    inertias = [apply_kmeans(scaled_data, k)[0].inertia_ for k in range(1, MAX_CLUSTERS + 1)]
    display_elbow_chart(inertias, MAX_CLUSTERS, OPTIMAL_K)

    _, cluster_labels = apply_kmeans(scaled_data, OPTIMAL_K)
    df = cluster_summary(df, cluster_labels, num_cols)
    visualize_clusters(scaled_data, cluster_labels)
    print_cluster_commentary()
    # df.to_csv('./Task_3/stats_by_clusters.csv') # If you want to export a CSV file included Cluster column

if __name__ == '__main__':
    main()
