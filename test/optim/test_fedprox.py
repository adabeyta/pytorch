# Owner(s): ["module: optimizer"]

import torch
from torch import nn
from torch.optim import FedProx
from torch.testing._internal.common_utils import run_tests, TestCase


class TestFedProx(TestCase):
    """Tests for the FedProx optimizer."""

    # ============== Setup Helpers ==============

    def _create_model(self, bias=True):
        """Create a simple linear model."""
        return nn.Linear(10, 5, bias=bias)

    def _create_optimizer(self, model, lr=0.1, mu=0.01, weight_decay=0):
        """Create a FedProx optimizer for the model."""
        return FedProx(model.parameters(), lr=lr, mu=mu, weight_decay=weight_decay)

    def _create_model_and_optimizer(self, lr=0.1, mu=0.01, weight_decay=0):
        """Create a model and its FedProx optimizer."""
        model = self._create_model()
        optimizer = self._create_optimizer(model, lr=lr, mu=mu, weight_decay=weight_decay)
        return model, optimizer

    def _get_input_target(self, input_shape=(4, 10), output_shape=(4, 5)):
        """Create random input and target tensors."""
        return torch.randn(input_shape), torch.randn(output_shape)

    def _set_global_params_from_model(self, optimizer, model):
        """Set optimizer's global params from model's current parameters."""
        global_params = [p.clone().detach() for p in model.parameters()]
        optimizer.set_global_params(global_params)
        return global_params

    def _forward_backward(self, model, optimizer, x, target):
        """Run forward pass, compute loss, and backpropagate."""
        output = model(x)
        loss = nn.MSELoss()(output, target)
        optimizer.zero_grad()
        loss.backward()
        return loss

    def _take_step(self, model, optimizer, x, target):
        """Forward, backward, and optimizer step. Returns loss."""
        loss = self._forward_backward(model, optimizer, x, target)
        optimizer.step()
        return loss

    # ============== Basic Functionality Tests ==============

    def test_basic_step(self):
        """Test that FedProx performs a basic optimization step."""
        model, optimizer = self._create_model_and_optimizer()
        self._set_global_params_from_model(optimizer, model)
        x, target = self._get_input_target()

        params_before = [p.clone() for p in model.parameters()]
        self._take_step(model, optimizer, x, target)

        for p_before, p_after in zip(params_before, model.parameters()):
            self.assertFalse(torch.equal(p_before, p_after))

    def test_closure(self):
        """Test that FedProx works with a closure."""
        model, optimizer = self._create_model_and_optimizer()
        self._set_global_params_from_model(optimizer, model)
        x, target = self._get_input_target()

        def closure():
            optimizer.zero_grad()
            loss = nn.MSELoss()(model(x), target)
            loss.backward()
            return loss

        loss = optimizer.step(closure)
        self.assertIsNotNone(loss)
        self.assertIsInstance(loss.item(), float)

    def test_multiple_steps(self):
        """Test multiple optimization steps (simulating FL rounds)."""
        model, optimizer = self._create_model_and_optimizer(mu=0.1)
        x, target = self._get_input_target()

        losses = []
        for _ in range(5):
            self._set_global_params_from_model(optimizer, model)
            for _ in range(3):
                loss = self._take_step(model, optimizer, x, target)
            losses.append(loss.item())

        self.assertLess(losses[-1], losses[0])

    # ============== Mathematical Correctness Tests ==============

    def test_proximal_term_effect(self):
        """Test that the proximal term pulls parameters toward global params."""
        torch.manual_seed(42)

        model = self._create_model(bias=False)
        global_model = self._create_model(bias=False)

        with torch.no_grad():
            model.weight.copy_(torch.randn(5, 10))
            global_model.weight.copy_(torch.randn(5, 10))

        lr, mu = 0.1, 1.0
        optimizer = self._create_optimizer(model, lr=lr, mu=mu)
        optimizer.set_global_params(global_model.parameters())

        x, target = self._get_input_target()
        self._forward_backward(model, optimizer, x, target)

        # Calculate expected update
        param = model.weight
        original_grad = param.grad.clone()
        param_before = param.clone()
        expected_effective_grad = original_grad + mu * (param - global_model.weight)

        optimizer.step()

        expected_param = param_before - lr * expected_effective_grad
        self.assertTrue(torch.allclose(param, expected_param, atol=1e-6))

    def test_mu_zero_equals_sgd(self):
        """Test that FedProx with mu=0 behaves like SGD."""
        torch.manual_seed(42)

        model_fedprox = self._create_model()
        model_sgd = self._create_model()
        with torch.no_grad():
            model_sgd.weight.copy_(model_fedprox.weight)
            model_sgd.bias.copy_(model_fedprox.bias)

        lr = 0.1
        optimizer_fedprox = self._create_optimizer(model_fedprox, lr=lr, mu=0.0)
        optimizer_sgd = torch.optim.SGD(model_sgd.parameters(), lr=lr)

        self._set_global_params_from_model(optimizer_fedprox, model_fedprox)

        torch.manual_seed(123)
        x, target = self._get_input_target()

        self._take_step(model_fedprox, optimizer_fedprox, x, target)
        self._take_step(model_sgd, optimizer_sgd, x, target)

        for p1, p2 in zip(model_fedprox.parameters(), model_sgd.parameters()):
            self.assertTrue(torch.allclose(p1, p2, atol=1e-6))

    def test_weight_decay(self):
        """Test that weight decay works correctly."""
        torch.manual_seed(42)

        model = self._create_model(bias=False)
        with torch.no_grad():
            model.weight.copy_(torch.randn(5, 10))

        lr, weight_decay = 0.1, 0.1
        optimizer = self._create_optimizer(model, lr=lr, mu=0.0, weight_decay=weight_decay)
        # Use same weights for global params (no proximal effect)
        self._set_global_params_from_model(optimizer, model)

        x, target = self._get_input_target()
        self._forward_backward(model, optimizer, x, target)

        param = model.weight
        original_grad = param.grad.clone()
        param_before = param.clone()

        optimizer.step()

        expected_param = param_before - lr * (original_grad + weight_decay * param_before)
        self.assertTrue(torch.allclose(param, expected_param, atol=1e-6))

    # ============== Parameter Group Tests ==============

    def test_param_groups(self):
        """Test that FedProx works with multiple parameter groups."""
        model1 = nn.Linear(10, 5)
        model2 = nn.Linear(5, 2)

        optimizer = FedProx([
            {"params": model1.parameters(), "lr": 0.1, "mu": 0.01},
            {"params": model2.parameters(), "lr": 0.05, "mu": 0.02},
        ])

        all_params = list(model1.parameters()) + list(model2.parameters())
        global_params = [p.clone().detach() for p in all_params]
        optimizer.set_global_params(global_params)

        x = torch.randn(4, 10)
        target = torch.randn(4, 2)

        params_before = [p.clone() for p in all_params]

        output = model2(model1(x))
        loss = nn.MSELoss()(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        for p_before, p_after in zip(params_before, all_params):
            self.assertFalse(torch.equal(p_before, p_after))

    # ============== Error Handling Tests ==============

    def test_global_params_not_set_raises(self):
        """Test that calling step() without set_global_params() raises."""
        model, optimizer = self._create_model_and_optimizer()
        x, target = self._get_input_target()

        self._forward_backward(model, optimizer, x, target)

        with self.assertRaises(RuntimeError) as ctx:
            optimizer.step()
        self.assertIn("Global parameters not set", str(ctx.exception))

    def test_global_params_shape_mismatch_raises(self):
        """Test that mismatched global param shapes raise an error."""
        model, optimizer = self._create_model_and_optimizer()
        wrong_global_params = [torch.randn(3, 3), torch.randn(3)]

        with self.assertRaises(ValueError) as ctx:
            optimizer.set_global_params(wrong_global_params)
        self.assertIn("Shape mismatch", str(ctx.exception))

    def test_global_params_count_mismatch_raises(self):
        """Test that wrong number of global params raises an error."""
        model, optimizer = self._create_model_and_optimizer()
        wrong_global_params = [torch.randn(5, 10)]  # Missing bias

        with self.assertRaises(ValueError) as ctx:
            optimizer.set_global_params(wrong_global_params)
        self.assertIn("Not enough global parameters", str(ctx.exception))

    def test_invalid_lr_raises(self):
        """Test that invalid learning rate raises ValueError."""
        model = self._create_model()
        with self.assertRaises(ValueError) as ctx:
            FedProx(model.parameters(), lr=-0.1)
        self.assertIn("Invalid learning rate", str(ctx.exception))

    def test_invalid_mu_raises(self):
        """Test that invalid mu raises ValueError."""
        model = self._create_model()
        with self.assertRaises(ValueError) as ctx:
            FedProx(model.parameters(), mu=-0.01)
        self.assertIn("Invalid proximal term strength", str(ctx.exception))

    def test_invalid_weight_decay_raises(self):
        """Test that invalid weight_decay raises ValueError."""
        model = self._create_model()
        with self.assertRaises(ValueError) as ctx:
            FedProx(model.parameters(), weight_decay=-0.1)
        self.assertIn("Invalid weight_decay", str(ctx.exception))

    # ============== State Serialization Tests ==============

    def test_state_dict_round_trip(self):
        """Test saving and loading optimizer state."""
        model, optimizer = self._create_model_and_optimizer()
        self._set_global_params_from_model(optimizer, model)
        x, target = self._get_input_target()

        self._take_step(model, optimizer, x, target)

        state_dict = optimizer.state_dict()
        new_optimizer = FedProx(model.parameters(), lr=0.1, mu=0.01)
        new_optimizer.load_state_dict(state_dict)

        # Verify param groups match
        self.assertEqual(len(optimizer.param_groups), len(new_optimizer.param_groups))
        for og, ng in zip(optimizer.param_groups, new_optimizer.param_groups):
            self.assertEqual(og["lr"], ng["lr"])
            self.assertEqual(og["mu"], ng["mu"])

        # Verify step() works after loading
        self._take_step(model, new_optimizer, x, target)

    # ============== Special Cases ==============

    def test_sparse_gradients(self):
        """Test that FedProx handles sparse gradients."""
        embedding = nn.Embedding(100, 10, sparse=True)
        optimizer = FedProx(embedding.parameters(), lr=0.1, mu=0.01)

        global_params = [p.clone().detach() for p in embedding.parameters()]
        optimizer.set_global_params(global_params)

        indices = torch.LongTensor([1, 5, 10])
        loss = embedding(indices).sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()  # Should not raise

    def test_maximize(self):
        """Test that maximize=True negates the gradient."""
        torch.manual_seed(42)

        model1 = self._create_model(bias=False)
        model2 = self._create_model(bias=False)
        with torch.no_grad():
            model2.weight.copy_(model1.weight)

        opt1 = FedProx(model1.parameters(), lr=0.1, mu=0.0, maximize=False)
        opt2 = FedProx(model2.parameters(), lr=0.1, mu=0.0, maximize=True)

        self._set_global_params_from_model(opt1, model1)
        self._set_global_params_from_model(opt2, model2)

        x, target = self._get_input_target()

        self._take_step(model1, opt1, x, target)
        self._take_step(model2, opt2, x, target)

        # With maximize, model2 should move opposite direction
        # Check weights diverged in opposite directions from start
        self.assertFalse(torch.allclose(model1.weight, model2.weight))

    def test_foreach(self):
        """Test that foreach=True works correctly."""
        model, optimizer = self._create_model_and_optimizer()
        optimizer.param_groups[0]["foreach"] = True
        self._set_global_params_from_model(optimizer, model)
        x, target = self._get_input_target()

        params_before = [p.clone() for p in model.parameters()]
        self._take_step(model, optimizer, x, target)

        for p_before, p_after in zip(params_before, model.parameters()):
            self.assertFalse(torch.equal(p_before, p_after))


if __name__ == "__main__":
    run_tests()
