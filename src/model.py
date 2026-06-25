# ═══════════════════════════════════════════════════════════════
#  model.py — Logistic Regression Model (PyTorch wrapper for Flower)
#  Federated Learning - Network Traffic Anomaly Detection
# ═══════════════════════════════════════════════════════════════

import torch
import torch.nn as nn

class LogisticRegressionModel(nn.Module):
    """
    A simple Logistic Regression model written in PyTorch.
    Flower needs models with numeric weights (tensors) so it can
    average them across clients — this is why we use PyTorch
    instead of sklearn's LogisticRegression for the FL part.
    """
    def __init__(self, input_dim):
        super(LogisticRegressionModel, self).__init__()
        # One linear layer: input_dim features -> 1 output (binary: normal/attack)
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        # Sigmoid squashes output between 0 and 1 (probability of attack)
        return torch.sigmoid(self.linear(x))


def get_model_parameters(model):
    """Extract model weights as a list of numpy arrays (Flower format)."""
    return [val.cpu().numpy() for val in model.state_dict().values()]


def set_model_parameters(model, parameters):
    """Load weights (from Flower server) back into the PyTorch model."""
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    model.load_state_dict(state_dict, strict=True)
    return model


if __name__ == "__main__":
    # Quick test to confirm the model works
    INPUT_DIM = 42  # number of features in our cleaned UNSW-NB15 dataset
    model = LogisticRegressionModel(INPUT_DIM)
    print("✅ Model created successfully!")
    print(model)

    # Test with dummy data
    dummy_input = torch.randn(5, INPUT_DIM)  # 5 samples, 42 features
    output = model(dummy_input)
    print("\n✅ Test forward pass output (5 samples):")
    print(output)