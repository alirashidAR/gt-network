import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import confusion_matrix, classification_report

def create_network(n_nodes, p):
    """
    Creates an Erdős-Rényi random graph.

    Args:
        n_nodes (int): Number of nodes in the graph.
        p (float): Probability of an edge existing between two nodes.

    Returns:
        networkx.Graph: The generated network.
    """
    G = nx.erdos_renyi_graph(n_nodes, p)
    return G

def calculate_features(G, time_window=None):
    """
    Calculates features for each node in the network.

    Args:
        G (networkx.Graph): The network.
        time_window (int, optional): Window size for temporal features (e.g., activity). Defaults to None.

    Returns:
        dict: A dictionary where keys are nodes and values are dictionaries containing features.
    """
    features = {}
    for node in G.nodes():
        features[node] = {
            'degree': G.degree(node) + np.random.uniform(-1, 1),  # Degree with noise
            'centrality': nx.degree_centrality(G).get(node, 0) + np.random.uniform(-0.1, 0.1),  # Degree centrality with noise
            'reputation': np.random.uniform(0, 1)  # Random reputation score
        }

        # Add temporal features if time_window is provided
        if time_window:
            # Example: Calculate activity within the time window
            activity = sum(1 for t in G.nodes[node]['timestamps'] if t >= max(G.nodes[node]['timestamps']) - time_window)
            features[node]['activity'] = activity

    return features

def payoff_matrix(action1, action2):
    """
    Defines the payoff matrix for interactions between two nodes.

    Args:
        action1 (str): Action of node 1 (share or ignore).
        action2 (str): Action of node 2 (share or ignore).

    Returns:
        tuple: Payoffs for node 1 and node 2, respectively.
    """
    if action1 == 'share' and action2 == 'share':
        return 3, 3  # High payoff for mutual sharing
    elif action1 == 'share' and action2 == 'ignore':
        return 0, 5  # High payoff for the one who ignores
    elif action1 == 'ignore' and action2 == 'share':
        return 5, 0  # High payoff for the one who ignores
    else:
        return 1, 1  # Low payoff for mutual ignoring

def play_game(node1, node2, features, history=None):
    """
    Plays the game between two nodes, considering history (optional).

    Args:
        node1 (int): Node 1 ID.
        node2 (int): Node 2 ID.
        features (dict): Feature dictionary for both nodes.
        history (list, optional): List of past interactions (strategies) for both nodes. Defaults to None.

    Returns:
        tuple: Strategies of node 1 and node 2, respectively.
    """
    strategies = ['share', 'ignore']
    payoffs = np.zeros((2, 2))
    for i, action1 in enumerate(strategies):
        for j, action2 in enumerate(strategies):
            payoffs[i, j] = sum(payoff_matrix(action1, action2))

    # Implement game theory strategy (e.g., Nash Equilibrium)
    # Consider history and features for an informed decision

    # For now, use a random strategy selection (can be replaced)
    action1 = 'share' if np.random.rand() < 0.5 else 'ignore'
    action2 = 'share' if np.random.rand() < 0.5 else 'ignore'

    return action1, action2

def generate_data(G, features, n_rounds, time_window=None):
    """
    Generates data for training the model.

    Args:
        G (networkx.Graph): The network.
        features (dict): Feature dictionary for nodes.
        n_rounds (int): Number of rounds to simulate.
        time_window (int, optional): Window size for temporal features. Defaults to None.

    Returns:
        list: A list of tuples containing feature vectors and strategies for each round.
    """
    data = []
    nodes = list(G.nodes())
    history = {}  # Store past interactions for each node

    for _ in range(n_rounds):
        node1, node2 = np.random.choice(nodes, 2, replace=False)
        history1 = history.get(node1, [])
        history2 = history.get(node2, [])

        strategy1, strategy2 = play_game(node1, node2, features, history=history)

        # Update history
        if node1 not in history:
            history[node1] = []
        if node2 not in history:
            history[node2] = []
        history[node1].append(strategy1)
        history[node2].append(strategy2)

        feature_vector = np.array([
            features[node1]['degree'],
            features[node1]['centrality'],
            features[node1]['reputation'],
            features[node2]['degree'],
            features[node2]['centrality'],
            features[node2]['reputation']
        ])

        data.append((feature_vector, strategy1, strategy2))

    return data

def train_model(data):
    X, y1, y2 = zip(*data)
    X = np.array(X)
    y1 = np.array([1 if s == 'share' else 0 for s in y1])
    y2 = np.array([1 if s == 'share' else 0 for s in y2])

    model1 = RandomForestClassifier(n_estimators=100, random_state=42)
    model2 = RandomForestClassifier(n_estimators=100, random_state=42)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    scores1 = cross_val_score(model1, X, y1, cv=kf)
    scores2 = cross_val_score(model2, X, y2, cv=kf)

    print(f"Cross-Validation Scores (Node 1): {scores1}")
    print(f"Mean Cross-Validation Score (Node 1): {scores1.mean()}")
    print(f"Cross-Validation Scores (Node 2): {scores2}")
    print(f"Mean Cross-Validation Score (Node 2): {scores2.mean()}")

    X_train, X_test, y1_train, y1_test, y2_train, y2_test = train_test_split(X, y1, y2, test_size=0.3, random_state=42)

    model1.fit(X_train, y1_train)
    model2.fit(X_train, y2_train)

    y1_pred = model1.predict(X_test)
    y2_pred = model2.predict(X_test)

    print("Classification Report (Node 1):")
    print(classification_report(y1_test, y1_pred))

    print("Classification Report (Node 2):")
    print(classification_report(y2_test, y2_pred))

    print("Confusion Matrix (Node 1):")
    print(confusion_matrix(y1_test, y1_pred))

    print("Confusion Matrix (Node 2):")
    print(confusion_matrix(y2_test, y2_pred))

    return model1, model2

def predict_strategy(model1, model2, features, node1, node2):
    feature_vector = np.array([
        features[node1]['degree'],
        features[node1]['centrality'],
        features[node1]['reputation'],
        features[node2]['degree'],
        features[node2]['centrality'],
        features[node2]['reputation']
    ]).reshape(1, -1)

    strategy1 = model1.predict(feature_vector)[0]
    strategy2 = model2.predict(feature_vector)[0]

    return 'share' if strategy1 == 1 else 'ignore', 'share' if strategy2 == 1 else 'ignore'

def visualize_network(G, model1, model2, features):
    pos = nx.spring_layout(G)
    plt.figure(figsize=(12, 6))

    # Plot network structure
    plt.subplot(1, 2, 1)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray')
    plt.title("Network Structure")

    # Predict and plot strategies
    plt.subplot(1, 2, 2)
    colors = []
    for node1, node2 in G.edges():
        strategy1, strategy2 = predict_strategy(model1, model2, features, node1, node2)
        if strategy1 == 'share' and strategy2 == 'share':
            colors.append('green')
        elif strategy1 == 'ignore' and strategy2 == 'ignore':
            colors.append('red')
        else:
            colors.append('orange')

    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color=colors)
    plt.title("Predicted Strategies")
    plt.show()

# Example Usage
n_nodes = 20
p = 0.3
G = create_network(n_nodes, p)
features = calculate_features(G)
data = generate_data(G, features, n_rounds=100)
model1, model2 = train_model(data)
visualize_network(G, model1, model2, features)
