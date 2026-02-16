# FedProx Optimizer for PyTorch

This branch adds `torch.optim.FedProx`, an optimizer for federated learning that prevents client drift in heterogeneous data settings.

**Paper:** [Federated Optimization in Heterogeneous Networks](https://arxiv.org/abs/1812.06127) (Li et al., 2020)

## Expected Output

```
======================================================================
  FedProx Proof of Concept
======================================================================

THE PROBLEM: In federated learning, each client trains on local data and sends
updates to a central server. When clients have very different data distributions
(non-IID), their models "drift" in different directions, hurting convergence.

THE SOLUTION: FedProx adds a proximal term to the loss function:
    L_fedprox = L_local + (mu/2) * ||w - w_global||^2
...

--- Training FedAvg (baseline) ---
  Round  5: Acc= 50.9%  Loss=1.511  Drift=1.60
  ...
  Round 25: Acc= 52.9%  Loss=1.772  Drift=0.68   <-- loss increasing (diverging)

--- Training FedProx (mu=1.0) ---
  Round  5: Acc= 45.7%  Loss=1.915  Drift=0.68
  ...
  Round 25: Acc= 59.1%  Loss=1.308  Drift=0.38   <-- loss decreasing (converging)

======================================================================
  Results Summary
======================================================================
                          FedAvg      FedProx     Improvement
  Avg Client Drift:         1.14         0.54         53% less drift
  Final Accuracy:          52.9%        59.1%       +6.2 points
```

## Usage

```python
from torch.optim import FedProx

# Create optimizer with proximal strength mu
optimizer = FedProx(model.parameters(), lr=0.01, mu=0.1)

# At the start of each federated round, set the global model parameters
optimizer.set_global_params(global_model.parameters())

# Training loop (same as any optimizer)
for data, target in dataloader:
    optimizer.zero_grad()
    loss = criterion(model(data), target)
    loss.backward()
    optimizer.step()
```

## Files

| File | Description |
|------|-------------|
| `torch/optim/fedprox.py` | FedProx optimizer implementation |
| `test/optim/test_fedprox.py` | Unit tests |
| `fedprox_poc.py` | Demo script comparing FedAvg vs FedProx |

## Running Tests

```bash
docker run --rm pytorch-fedprox python -m pytest test/optim/test_fedprox.py -v
```
