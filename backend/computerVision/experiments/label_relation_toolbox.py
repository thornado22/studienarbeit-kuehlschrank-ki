from loader import LabelLoader, MetaLoader
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules
from pyvis.network import Network
import networkx as nx
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import umap
import hdbscan
from sklearn.metrics import silhouette_score
    
def calc_co_occurence_matrix(label_df):
    X = label_df

    # compute co-occurrence
    co_matrix = X.T @ X

    # normalize
    co_matrix_norm = co_matrix / np.max(co_matrix.values)
    sns.heatmap(co_matrix_norm, cmap="viridis")
    plt.show()
    return co_matrix_norm

def calc_association(label_df):
    X_bool = label_df.astype(bool)

    frequent_items = apriori(X_bool, min_support=0.05, use_colnames=True)
    rules = association_rules(frequent_items, metric="lift", min_threshold=1.2)

    return rules


def cluster_labels(
    label_df: pd.DataFrame,
    n_components_umap: int = 10,
    min_cluster_size: int = 10,
    min_samples: int | None = None,
    top_n_features: int = 10,
    plot: bool = False,
):
    random_state = 42
    X = label_df.astype(np.uint8).values

    # UMAP reduction for clustering
    reducer = umap.UMAP(
        n_components=n_components_umap,
        metric="jaccard",
        random_state=random_state,
    )

    X_reduced = reducer.fit_transform(X)

    # HDBSCAN clustering
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="leaf",
        prediction_data=True,
    )

    cluster_labels = clusterer.fit_predict(X_reduced)

    probabilities = clusterer.probabilities_

    # Silhouette score
    mask = cluster_labels != -1

    if mask.sum() > 1 and len(np.unique(cluster_labels[mask])) > 1:
        sil_score = silhouette_score(
            X_reduced[mask],
            cluster_labels[mask],
            metric="euclidean",
        )
    else:
        sil_score = None


    # 2D visualization
    if plot:
        reducer_2d = umap.UMAP(
            n_components=2,
            metric="jaccard",
            random_state=random_state,
        )

        X_2d = reducer_2d.fit_transform(X)

        plt.figure(figsize=(10, 8))

        n_clusters = len(np.unique(cluster_labels))
        palette = sns.color_palette("tab20", n_clusters)

        sns.scatterplot(
            x=X_2d[:, 0],
            y=X_2d[:, 1],
            hue=cluster_labels,
            palette=palette,
            s=20,
            linewidth=0,
        )

        plt.title("HDBSCAN Clusters (-1 = Noise)")
        plt.xlabel("UMAP-1")
        plt.ylabel("UMAP-2")
        plt.legend(
            title="Cluster",
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
        )

        plt.tight_layout()
        plt.show()

    # Cluster interpretation
    clustered_df = label_df.copy()
    clustered_df["cluster"] = cluster_labels

    cluster_top_features = {}
    noise_count = (clustered_df["cluster"] == -1).sum()
    noise_ratio = noise_count / len(clustered_df)

    print(f"\nSilhouette score: {sil_score}")
    print(f"Noise: {noise_count} / {len(clustered_df)} ({noise_ratio:.2%})")

    print("\nTop labels per cluster:\n")

    for c in sorted(clustered_df["cluster"].unique()):

        if c == -1:
            continue

        subset = clustered_df[clustered_df["cluster"] == c]

        feature_freqs = (
            subset.drop(columns=["cluster"])
            .mean()
            .sort_values(ascending=False)
            .head(top_n_features)
        )

        cluster_top_features[c] = feature_freqs

        print(f"Cluster {c} | Size = {len(subset)}")
        print(feature_freqs)
        print("-" * 50)

    # Return results
    results = {
        "labels": cluster_labels,
        "probabilities": probabilities,
        "silhouette_score": sil_score,
        "clustered_df": clustered_df,
        "cluster_top_features": cluster_top_features,
    }

    return results



    
def build_label_network(label_df, output_path):
    X = label_df

    # --- Co-occurrence ---
    co_matrix = X.T @ X
    row_sums = X.sum(axis=0)

    # --- Jaccard similarity ---
    intersection = co_matrix.values
    sums = row_sums.values

    union = sums[:, None] + sums[None, :] - intersection
    jaccard = np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection, dtype=float),
        where=union != 0
    )

    jaccard_df = pd.DataFrame(
        jaccard,
        index=co_matrix.index,
        columns=co_matrix.columns
    )

    # --- Build graph ---
    G = nx.Graph()

    threshold = 0.2  # increase to reduce clutter

    for i, food1 in enumerate(jaccard_df.columns):
        for j, food2 in enumerate(jaccard_df.columns):
            if j <= i:
                continue

            weight = jaccard_df.iloc[i, j]

            if weight > threshold:
                G.add_edge(food1, food2, weight=weight)


    net = Network(
        height="1000px",
        width="100%",
        notebook=False,
        cdn_resources="in_line"
    )

    # Normalize node values for coloring
    node_values = np.array([row_sums[node] for node in G.nodes])
    norm_nodes = mcolors.Normalize(vmin=node_values.min(), vmax=node_values.max())

    # Normalize edge weights
    weights = [d["weight"] for (_, _, d) in G.edges(data=True)]
    norm_edges = mcolors.Normalize(vmin=min(weights), vmax=max(weights))

    special_nodes = ["tuer", "schublade", "gesamter_kuehlschrank", "faecher"]

    # --- Add nodes ---
    for node in G.nodes:
        freq = int(row_sums[node])

        size = np.sqrt(freq) * 4

        if node in special_nodes:
            color = "gray"
        else:
            rgba = cm.plasma(norm_nodes(freq))
            color = mcolors.to_hex(rgba)

        net.add_node(
            str(node),
            label=str(node),
            size=size,
            color=color,
            title=f"{node}<br>Count: {freq}"
        )
    # --- Add edges ---
    for u, v, d in G.edges(data=True):
        w = float(d["weight"])

        width = float(w * 10)

        rgba = cm.viridis(norm_edges(w))
        color = mcolors.to_hex(rgba)

        net.add_edge(
            str(u),
            str(v),
            value=width,
            color=color,
            title=f"Strength: {w:.2f}"
        )

    # --- Physics tuning ---
    net.set_options("""
    var options = {
    "physics": {
        "forceAtlas2Based": {
        "gravitationalConstant": -50,
        "centralGravity": 0.01,
        "springLength": 100,
        "springConstant": 0.1
        },
        "solver": "forceAtlas2Based"
    }
    }
    """)
    # --- Save to file with UTF-8 encoding ---
    html = net.generate_html(notebook=False)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    legend_html = """
    <div style="
    position: fixed;
    bottom: 20px;
    left: 20px;
    background-color: white;
    padding: 10px;
    border: 1px solid #ccc;
    border-radius: 8px;
    font-family: Arial;
    font-size: 14px;
    box-shadow: 2px 2px 10px rgba(0,0,0,0.2);
    z-index:999;
    ">
    <b>Legend</b><br><br>

    <span style="color:gray;">●</span> Special nodes<br>
    <span style="color:#440154;">●</span> Low frequency<br>
    <span style="color:#fde725;">●</span> High frequency<br><br>

    <span style="border-bottom:3px solid #440154;">&nbsp;&nbsp;&nbsp;</span> Weak relation<br>
    <span style="border-bottom:3px solid #fde725;">&nbsp;&nbsp;&nbsp;</span> Strong relation
    </div>
    """

    with open(output_path, "a", encoding="utf-8") as f:
        f.write(legend_html)




if __name__ == "__main__":
    label_json = "./images/allLables.json"
    labelLoader = LabelLoader(label_json)

    label_df = labelLoader.load(0, True)
    label_df2 = labelLoader.load(0, False)

    build_label_network(label_df, "./stats/label_network.html")
    build_label_network(label_df2, "./stats/label_network2.html")
    
    # results = cluster_labels(
    #     label_df,
    #     n_components_umap=10,
    #     min_cluster_size=10,
    #     min_samples=5,
    # )

    

