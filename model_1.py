import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import confusion_matrix, classification_report

def create_network(n_nodes, p):
    '''
    Creating the network
    '''
    G = nx.erdos_renyi_graph(n_nodes, p)
    return G

def calculate_features(G):
    '''
    Feature Generation.
    '''
    features = {}
    for node in G.nodes():
        features[node] = {
            'degree': G.degree(node) + np.random.uniform(-1, 1),  # Added noise
            'centrality': nx.degree_centrality(G).get(node, 0) + np.random.uniform(-0.1, 0.1),
            'reputation': np.random.uniform(0, 1)  # Random reputation score
        }
    return features

def payoff_matrix(action1, action2):
    '''
    Define the payoffs for different interactions
    '''
    if action1 == 'share' and action2 == 'share':
        return 3, 3  # High payoff for mutual sharing
    elif action1 == 'share' and action2 == 'ignore':
        return 0, 5  # High payoff for the one who ignores
    elif action1 == 'ignore' and action2 == 'share':
        return 5, 0  # High payoff for the one who ignores
    else:
        return 1, 1  # Low payoff for mutual ignoring

def play_game(node1, node2, features):
    '''
    Playing the game with Nash Equilibrium in mind
    '''
    strategies = ['share', 'ignore']
    payoffs = np.zeros((2, 2))
    for i, action1 in enumerate(strategies):
        for j, action2 in enumerate(strategies):
            payoffs[i, j] = sum(payoff_matrix(action1, action2))
    
    # Using RandomForest to predict the strategy
    action1 = 'share' if np.random.rand() < 0.5 else 'ignore'
    action2 = 'share' if np.random.rand() < 0.5 else 'ignore'
    
    return action1, action2

def generate_data(G, features, n_rounds):
    '''
    Generate data for nodes.
    '''
    data = []
    nodes = list(G.nodes())
    for _ in range(n_rounds):
        node1, node2 = np.random.choice(nodes, 2, replace=False)
        strategy1, strategy2 = play_game(node1, node2, features)
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
    y1 = np.array(y1)
    y2 = np.array(y2)
    
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
    
    return strategy1, strategy2

def visualize_network(G, model1, model2, features):
    pos = nx.spring_layout(G)
    plt.figure(figsize=(12, 6))

    # Plot network structure
    plt.subplot(1, 2, 1)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray')
    plt.title('Network Structure')

    # Feature importance
    importances1 = model1.feature_importances_
    importances2 = model2.feature_importances_
    features_names = ['deg1', 'cent1', 'rep1', 'deg2', 'cent2', 'rep2']
    
    importance_df1 = pd.DataFrame({'Feature': features_names, 'Importance': importances1})
    importance_df1 = importance_df1.sort_values(by='Importance', ascending=False)
    
    importance_df2 = pd.DataFrame({'Feature': features_names, 'Importance': importances2})
    importance_df2 = importance_df2.sort_values(by='Importance', ascending=False)

    plt.subplot(1, 2, 2)
    plt.bar(importance_df1['Feature'], importance_df1['Importance'], alpha=0.6, label='Node 1')
    plt.bar(importance_df2['Feature'], importance_df2['Importance'], alpha=0.6, label='Node 2')
    plt.title('Feature Importance')
    plt.xlabel('Feature')
    plt.ylabel('Importance')
    plt.xticks(rotation=45)
    plt.legend()

    plt.tight_layout()
    plt.show()

# Main Execution
n_nodes = 50
p = 0.5
n_rounds = 100

G = create_network(n_nodes, p)
features = calculate_features(G)
data = generate_data(G, features, n_rounds)
model1, model2 = train_model(data)

# Predict strategy for 10 new pairs of nodes
def predict_multiple_pairs(G, model1, model2, features, n_pairs=10):
    nodes = list(G.nodes())
    predictions = []
    for _ in range(n_pairs):
        node1, node2 = np.random.choice(nodes, 2, replace=False)
        strategy1, strategy2 = predict_strategy(model1, model2, features, node1, node2)
        predictions.append((node1, node2, strategy1, strategy2))
    return predictions

predictions = predict_multiple_pairs(G, model1, model2, features, n_pairs=10)
for node1, node2, strategy1, strategy2 in predictions:
    print(f"Predicted strategies for nodes {node1} and {node2}: {strategy1}, {strategy2}")

# Visualization
visualize_network(G, model1, model2, features)
