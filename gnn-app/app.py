import os
import numpy as np
import torch
import rustworkx as rx
import gc
from torch_geometric.data import Data
from sklearn.cluster import KMeans, SpectralClustering, DBSCAN, AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score
import joblib
import networkx as nx
import matplotlib.pyplot as plt
import io
import base64
from nicegui import ui, app

# Configuration
DATA_DIR = os.environ.get('DATA_DIR', './data')
PORT = int(os.environ.get('NICEGUI_PORT', 8501))
HOST = os.environ.get('NICEGUI_HOST', '0.0.0.0')

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

# Your existing functions
def pyg_to_rustworkx(edge_index, num_nodes):
    G = rx.PyGraph()
    G.add_nodes_from(range(num_nodes))  # Ensure all nodes are included
    edges = list(map(tuple, edge_index.T.tolist()))  # Convert list of lists to list of tuples
    G.add_edges_from_no_data(edges)  # No edge weights
    return G

# Create a function to convert PyG data to NetworkX for visualization
def pyg_to_networkx(edge_index, num_nodes):
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    edges = list(map(tuple, edge_index.T.tolist()))
    G.add_edges_from(edges)
    return G

# Function to plot and return a graph image as base64
def plot_graph_with_communities(G, labels, method_name):
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42)  # Position nodes using Fruchterman-Reingold force-directed algorithm
    
    # Get unique communities for coloring
    unique_communities = np.unique(labels)
    
    # Plot nodes with community colors
    for comm_id in unique_communities:
        if comm_id == -1:  # Unlabeled nodes
            node_list = [node for node in G.nodes() if labels[node] == -1]
            nx.draw_networkx_nodes(G, pos, nodelist=node_list, node_color='gray', node_size=50, alpha=0.6)
        else:
            node_list = [node for node in G.nodes() if labels[node] == comm_id]
            nx.draw_networkx_nodes(G, pos, nodelist=node_list, node_color=f'C{comm_id % 10}', node_size=80, label=f'Community {comm_id}')
    
    nx.draw_networkx_edges(G, pos, alpha=0.2)
    
    plt.title(f"Network Communities - {method_name}")
    plt.legend(scatterpoints=1, frameon=True, labelspacing=1)
    plt.axis('off')
    
    # Save plot to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150)
    plt.close()
    buf.seek(0)
    
    # Convert plot to base64 string
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    return f'data:image/png;base64,{img_str}'

# Function to run clustering on the embeddings
def run_clustering(node_embeddings, method_name, num_clusters):
    if method_name == "K-Means":
        algo = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    elif method_name == "Spectral Clustering":
        algo = SpectralClustering(n_clusters=num_clusters, affinity='nearest_neighbors', 
                                 random_state=42, assign_labels='kmeans')
    elif method_name == "DBSCAN":
        algo = DBSCAN(eps=0.5, min_samples=5)
    elif method_name == "Agglomerative":
        algo = AgglomerativeClustering(n_clusters=num_clusters)
    elif method_name == "Louvain":
        # This is handled separately through rustworkx
        return None
    
    if method_name != "Louvain":
        predicted_labels = algo.fit_predict(node_embeddings)
        return predicted_labels
    return None

# Data loading function - replace with your actual data loading
def load_data():
    # Check if we have saved data
    data_file = os.path.join(DATA_DIR, 'network_data.pkl')
    
    if os.path.exists(data_file):
        print(f"Loading saved data from {data_file}")
        data = joblib.load(data_file)
        return data
    
    # If no saved data, create mock data for demonstration
    print("Creating mock data")
    num_nodes = 100
    edge_index = torch.tensor(np.random.randint(0, num_nodes, size=(2, 300)), dtype=torch.long)
    x = torch.randn(num_nodes, 10)  # Random node features
    true_labels = torch.randint(0, 5, (num_nodes,))  # Random ground truth labels
    labeled_mask = torch.ones(num_nodes, dtype=torch.bool)  # All nodes labeled for demo
    
    class MockModel:
        def __init__(self):
            pass
        
        def eval(self):
            pass
        
        def __call__(self, x, edge_index):
            # Return mock node features and embeddings
            return None, torch.randn(num_nodes, 16)  # 16-dimensional embeddings
    
    model = MockModel()
    
    # Create and return data object
    data = {
        'num_nodes': num_nodes,
        'edge_index': edge_index,
        'x': x,
        'true_labels': true_labels,
        'labeled_mask': labeled_mask,
        'model': model
    }
    
    return data

# Load data at startup
data = load_data()
num_nodes = data['num_nodes']
edge_index = data['edge_index']
x = data['x']
true_labels = data['true_labels']
labeled_mask = data['labeled_mask']
model = data['model']

# Create NetworkX graph for visualization
nx_G = pyg_to_networkx(edge_index, num_nodes)

# Extract embeddings
model.eval()
_, node_embeddings = model(x, edge_index)
node_embeddings = node_embeddings.detach().numpy()

# Create the NiceGUI app
@ui.page('/')
def home():
    with ui.card().classes('w-full'):
        ui.label('Network Community Detection').classes('text-2xl')
        ui.markdown('''
        This application visualizes community detection results on network data using graph neural networks.
        Select a clustering method and adjust parameters to see different community structures.
        ''')

    # Clustering method selection
    with ui.card().classes('w-full'):
        ui.label('Clustering Settings').classes('text-xl')
        
        method = ui.select(
            ['K-Means', 'Spectral Clustering', 'DBSCAN', 'Agglomerative', 'Louvain'],
            value='K-Means',
            label='Clustering Method'
        )
        
        num_clusters = ui.number('Number of Clusters', value=5, min=2, max=20)
        
        def update_cluster_input():
            if method.value == 'DBSCAN' or method.value == 'Louvain':
                num_clusters.disable()
            else:
                num_clusters.enable()
        
        method.on('change', update_cluster_input)
        update_cluster_input()
        
        # Run clustering button
        run_button = ui.button('Run Community Detection', color='primary')
        
        # Results display area
        result_label = ui.label()
        result_metrics = ui.label()
        graph_img = ui.image()
    
    async def run_community_detection():
        result_label.text = f"Running {method.value} clustering..."
        
        # Update UI to show "processing"
        await ui.sleep(0.1)
        
        if method.value == 'Louvain':
            # Run Louvain with RustWorkx
            G_rx = pyg_to_rustworkx(edge_index, num_nodes)
            partition = rx.community_louvain(G_rx)
            predicted_labels = np.array([partition.get(n, -1) for n in range(num_nodes)])
        else:
            # Run other clustering methods
            predicted_labels = run_clustering(
                node_embeddings, 
                method.value, 
                int(num_clusters.value)
            )
        
        # Calculate metrics
        ari_score = adjusted_rand_score(true_labels.numpy(), predicted_labels)
        
        # Show metrics
        result_metrics.text = (
            f"**Results for {method.value}:**\n"
            f"- Adjusted Rand Index (ARI): {ari_score:.4f}\n"
            f"- Number of Communities: {len(np.unique(predicted_labels))}"
        )
        
        # Generate visualization
        img_data = plot_graph_with_communities(nx_G, predicted_labels, method.value)
        graph_img.set_source(img_data)
        
        result_label.text = f"Completed {method.value} clustering"
        
        # Save results
        result_file = os.path.join(DATA_DIR, f'{method.value.lower()}_results.pkl')
        joblib.dump({
            'predicted_labels': predicted_labels,
            'ari_score': ari_score,
            'method': method.value
        }, result_file)
    
    run_button.on('click', run_community_detection)

    with ui.card().classes('w-full'):
        ui.label('About This Tool').classes('text-xl')
        ui.markdown('''
        This tool combines Graph Neural Networks with traditional and graph-based clustering methods
        to detect communities in network data. The visualization shows different node colors for 
        each detected community.
        
        Metrics:
        - **Adjusted Rand Index (ARI)**: Measures the similarity between the true communities and the detected ones
        - **Number of Communities**: Total number of distinct communities found
        
        For more details about the algorithms, check the documentation.
        ''')

    with ui.card().classes('w-full'):
        ui.label('Deployment Information').classes('text-sm text-gray-500')
        ui.markdown(f'''
        * Running on Docker
        * Data stored in: {DATA_DIR}
        * Version: 1.0.0
        ''')

# Configure for production deployment
ui.run(title='Network Community Detection', host=HOST, port=PORT, show=False)