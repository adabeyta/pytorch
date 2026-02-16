#!/usr/bin/env python3
"""
FedProx Proof of Concept
Based on experimental setups from:
FedProx paper: https://arxiv.org/abs/1812.06127
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
from torch.optim.fedprox import FedProx


def print_header(text):
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def print_section(text):
    print(f"\n--- {text} ---")


def generate_mnist_style_data(num_classes=10, samples_per_class=500, input_dim=50, seed=42):
    torch.manual_seed(seed)

    X_list, y_list = [], []

    for c in range(num_classes):
        # Each class has a distinct center in feature space
        center = torch.zeros(input_dim)
        # First few dimensions encode class identity (moderate separation)
        center[c % input_dim] = 2.5
        center[(c + 5) % input_dim] = 1.5
        # Add some random offset
        center += torch.randn(input_dim) * 0.3

        # Generate samples with class-specific noise
        noise_scale = 1.2 + 0.1 * c  # Moderate noise = harder problem
        X_class = center + torch.randn(samples_per_class, input_dim) * noise_scale
        y_class = torch.full((samples_per_class,), c, dtype=torch.long)

        X_list.append(X_class)
        y_list.append(y_class)

    X = torch.cat(X_list)
    y = torch.cat(y_list)

    perm = torch.randperm(len(X))
    return X[perm], y[perm]


def create_pathological_non_iid(X, y, num_clients=5, classes_per_client=2):
    num_classes = y.max().item() + 1
    client_data = {}

    for client_id in range(num_clients):
        # Assign 'classes_per_client' classes to this client
        start_class = (client_id * classes_per_client) % num_classes
        client_classes = [(start_class + i) % num_classes for i in range(classes_per_client)]

        # Get samples for these classes
        mask = torch.zeros(len(y), dtype=torch.bool)
        for c in client_classes:
            mask |= (y == c)

        client_X = X[mask]
        client_y = y[mask]

        client_data[f"Client {client_id}"] = {
            "X": client_X,
            "y": client_y,
            "classes": client_classes
        }

    return client_data

class MLP(nn.Module):

    def __init__(self, input_dim=50, hidden_dim=128, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

def clone_model(model, input_dim, hidden_dim, num_classes):
    new_model = MLP(input_dim, hidden_dim, num_classes)
    new_model.load_state_dict(model.state_dict())
    return new_model

def aggregate_models(models, input_dim, hidden_dim, num_classes):
    avg_state = {}
    for key in models[0].state_dict():
        tensors = [m.state_dict()[key].float() for m in models]
        avg_state[key] = torch.stack(tensors).mean(dim=0)

    new_model = MLP(input_dim, hidden_dim, num_classes)
    new_model.load_state_dict(avg_state)
    return new_model

def evaluate(model, X, y):
    model.eval()
    with torch.no_grad():
        logits = model(X)
        loss = F.cross_entropy(logits, y).item()
        acc = (logits.argmax(dim=1) == y).float().mean().item() * 100
    return loss, acc


def compute_drift(model, global_model):
    dist_sq = 0.0
    for p1, p2 in zip(model.parameters(), global_model.parameters()):
        dist_sq += (p1.data - p2.data).pow(2).sum().item()
    return dist_sq ** 0.5


def train_round(client_data, global_model, local_epochs, lr, use_fedprox, mu,
                input_dim, hidden_dim, num_classes):
    client_models = []
    total_drift = 0.0

    for name, data in client_data.items():
        X, y = data["X"], data["y"]

        model = clone_model(global_model, input_dim, hidden_dim, num_classes)

        # Setup optimizer
        if use_fedprox:
            optimizer = FedProx(model.parameters(), lr=lr, mu=mu)
            optimizer.set_global_params(global_model.parameters())
        else:
            optimizer = SGD(model.parameters(), lr=lr)

        # Local training
        model.train()
        batch_size = 32
        n_samples = len(X)

        for epoch in range(local_epochs):
            perm = torch.randperm(n_samples)
            X_shuf, y_shuf = X[perm], y[perm]

            for i in range(0, n_samples, batch_size):
                bx = X_shuf[i:i+batch_size]
                by = y_shuf[i:i+batch_size]

                optimizer.zero_grad()
                loss = F.cross_entropy(model(bx), by)
                loss.backward()
                optimizer.step()

        drift = compute_drift(model, global_model)
        total_drift += drift
        client_models.append(model)

    avg_drift = total_drift / len(client_data)
    return client_models, avg_drift


def run_federated(client_data, test_X, test_y, num_rounds, local_epochs, lr,
                  use_fedprox, mu, input_dim, hidden_dim, num_classes, verbose=True):
    global_model = MLP(input_dim, hidden_dim, num_classes)

    history = {"acc": [], "loss": [], "drift": []}

    for r in range(num_rounds):
        # Train on all clients
        client_models, avg_drift = train_round(
            client_data, global_model, local_epochs, lr,
            use_fedprox, mu, input_dim, hidden_dim, num_classes
        )

        # Aggregate
        global_model = aggregate_models(client_models, input_dim, hidden_dim, num_classes)

        # Evaluate
        loss, acc = evaluate(global_model, test_X, test_y)
        history["acc"].append(acc)
        history["loss"].append(loss)
        history["drift"].append(avg_drift)

        if verbose and (r + 1) % 5 == 0:
            print(f"  Round {r+1:2d}: Acc={acc:5.1f}%  Loss={loss:.3f}  Drift={avg_drift:.2f}")

    return history

def run_demo():
    print_header("FedProx Proof of Concept")

    # Explain what FedProx does
    print("""
THE PROBLEM: In federated learning, each client trains on local data and sends
updates to a central server. When clients have very different data distributions
(non-IID), their models "drift" in different directions, hurting convergence.

THE SOLUTION: FedProx adds a proximal term to the loss function:
    L_fedprox = L_local + (mu/2) * ||w - w_global||^2

This acts like a "rubber band" pulling each client's model toward the global
model, preventing excessive drift while still learning from local data.
""")

    # Configuration
    num_classes = 10
    input_dim = 50
    hidden_dim = 128
    num_clients = 5
    classes_per_client = 2  # Pathological: each client sees only 2/10 classes

    num_rounds = 25
    local_epochs = 20
    lr = 0.03

    print_section("Scenario")
    print(f"{num_classes}-class classification with {num_clients} clients")
    print(f"Each client only sees {classes_per_client}/{num_classes} classes (highly non-IID)")
    print(f"This simulates real-world data silos (e.g., hospitals with different patient populations)")

    X, y = generate_mnist_style_data(num_classes=num_classes, samples_per_class=600,
                                      input_dim=input_dim, seed=42)
    split = int(0.85 * len(X))
    train_X, train_y = X[:split], y[:split]
    test_X, test_y = X[split:], y[split:]
    client_data = create_pathological_non_iid(train_X, train_y, num_clients, classes_per_client)

    print_section("Training FedAvg (baseline)")
    fedavg_hist = run_federated(
        client_data, test_X, test_y, num_rounds, local_epochs, lr,
        use_fedprox=False, mu=0,
        input_dim=input_dim, hidden_dim=hidden_dim, num_classes=num_classes,
        verbose=True
    )

    print_section("Training FedProx (mu=1.0)")
    fedprox_hist = run_federated(
        client_data, test_X, test_y, num_rounds, local_epochs, lr,
        use_fedprox=True, mu=1.0,
        input_dim=input_dim, hidden_dim=hidden_dim, num_classes=num_classes,
        verbose=True
    )

    # Results
    best_fedavg = max(fedavg_hist['acc'])
    best_fedprox = max(fedprox_hist['acc'])
    final_fedavg = fedavg_hist['acc'][-1]
    final_fedprox = fedprox_hist['acc'][-1]
    drift_fedavg = sum(fedavg_hist['drift']) / len(fedavg_hist['drift'])
    drift_fedprox = sum(fedprox_hist['drift']) / len(fedprox_hist['drift'])

    # Results
    print_header("Results Summary")
    print(f"""
                          FedAvg      FedProx     Improvement
                          ------      -------     -----------
  Avg Client Drift:        {drift_fedavg:5.2f}        {drift_fedprox:5.2f}       {(drift_fedavg - drift_fedprox) / drift_fedavg * 100:4.0f}% less drift
  Best Accuracy:          {best_fedavg:5.1f}%       {best_fedprox:5.1f}%       +{best_fedprox - best_fedavg:.1f} points
  Final Accuracy:         {final_fedavg:5.1f}%       {final_fedprox:5.1f}%       +{final_fedprox - final_fedavg:.1f} points

TAKEAWAY: FedProx reduces client drift, leading to better and more stable
convergence in heterogeneous federated settings.
""")


if __name__ == "__main__":
    run_demo()
