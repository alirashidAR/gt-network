import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import confusion_matrix, classification_report

# Create network
def create_network(n_nodes, p):
    G = nx.erdos_renyi_graph(n_nodes, p)
    return G

# Calculate node features
def calculate_features(G):
    features = {}
    for node in G.nodes():
        features[node] = {
            'degree': G.degree(node),
            'centrality': nx.degree_centrality(G).get(node, 0),
            'reputation': np.random.uniform(0, 1)  # Random reputation score
        }
    return features

# Generate interaction data
def generate_data(G, features, n_rounds):
    data = []
    nodes = list(G.nodes())
    for _ in range(n_rounds):
        node1, node2 = np.random.choice(nodes, 2, replace=False)
        strategy1, strategy2 = np.random.choice(['share', 'ignore'], 2)  # Random strategies for simplicity

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

# Prepare data for training
def prepare_data(data):
    X, y1, y2 = zip(*data)
    X = np.array(X)
    y1 = np.array([1 if s == 'share' else 0 for s in y1])
    y2 = np.array([1 if s == 'share' else 0 for s in y2])
    return X, y1, y2

# Train model
def train_model(X, y):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

# Predict strategies
def predict_strategy(model, features, node1, node2):
    feature_vector = np.array([
        features[node1]['degree'],
        features[node1]['centrality'],
        features[node1]['reputation'],
        features[node2]['degree'],
        features[node2]['centrality'],
        features[node2]['reputation']
    ]).reshape(1, -1)
    
    prediction = model.predict(feature_vector)[0]
    return 'share' if prediction == 1 else 'ignore'

# Visualize network
def visualize_network(G, model, features):
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
        strategy = predict_strategy(model, features, node1, node2)
        if strategy == 'share':
            colors.append('green')
        else:
            colors.append('red')

    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color=colors)
    plt.title("Predicted Strategies")
    plt.show()
    print("Green (Share):", colors.count('green'))
    print("Red (Ignore):", colors.count('red'))

# Example usage
n_nodes = 20
p = 0.3
G = create_network(n_nodes, p)
features = calculate_features(G)
data = generate_data(G, features, n_rounds=1000)
X, y1, y2 = prepare_data(data)

# Train models
model1 = train_model(X, y1)
model2 = train_model(X, y2)

# Evaluate models
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores1 = cross_val_score(model1, X, y1, cv=kf)
scores2 = cross_val_score(model2, X, y2, cv=kf)

print(f"Cross-Validation Scores (Model 1): {scores1}")
print(f"Mean Cross-Validation Score (Model 1): {scores1.mean()}")
print(f"Cross-Validation Scores (Model 2): {scores2}")
print(f"Mean Cross-Validation Score (Model 2): {scores2.mean()}")

X_train, X_test, y1_train, y1_test, y2_train, y2_test = train_test_split(X, y1, y2, test_size=0.3, random_state=42)

model1.fit(X_train, y1_train)
model2.fit(X_train, y2_train)

y1_pred = model1.predict(X_test)
y2_pred = model2.predict(X_test)

print("Classification Report (Model 1):")
print(classification_report(y1_test, y1_pred))

print("Classification Report (Model 2):")
print(classification_report(y2_test, y2_pred))

print("Confusion Matrix (Model 1):")
print(confusion_matrix(y1_test, y1_pred))

print("Confusion Matrix (Model 2):")
print(confusion_matrix(y2_test, y2_pred))

# Visualize network
visualize_network(G, model1, features)
