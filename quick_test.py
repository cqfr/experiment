from __future__ import annotations

"""Quick smoke tests for core modules."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def test_config() -> None:
    from config import get_dp_fedavg_config, get_dp_fedsam_config, get_proposed_config

    cfg1 = get_dp_fedavg_config()
    cfg2 = get_dp_fedsam_config()
    cfg3 = get_proposed_config()

    assert cfg1.dp.enabled
    assert cfg2.client.training_strategy.value == "dp_fedsam"
    assert cfg3.client.training_strategy.value == "contrastive"


def test_dp_accountant() -> None:
    from dp.noise import RDPAccountant

    acc = RDPAccountant(
        epsilon_total=8.0,
        delta=1e-5,
        num_rounds=100,
        orders=(2, 4, 8, 16, 32, 64, 128),
    )
    sigma = acc.solve_sigma_for_round(q=0.1)
    eps = acc.consume_round(sigma=sigma, q=0.1)
    assert sigma > 0
    assert eps > 0


def test_client_server_round() -> None:
    from client.client import FLClient
    from config import ClientConfig, DPConfig, ServerConfig
    from server.server import FLServer

    class TinyNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = nn.Linear(8, 4)

        def forward(self, x):
            return self.fc(x)

    model = TinyNet()
    dataset = TensorDataset(torch.randn(64, 8), torch.randint(0, 4, (64,)))
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    client = FLClient(
        client_id=0,
        model=model,
        dataloader=loader,
        config=ClientConfig(),
        dp_config=DPConfig(),
        device=torch.device("cpu"),
    )

    global_weights = model.state_dict()
    update = client.train_and_upload(
        global_weights=global_weights,
        clip_norm=1.0,
        sigma_base=0.1,
    )

    assert update.data_size == 64
    assert len(update.delta_w) > 0

    server = FLServer(model=model, config=ServerConfig(), device=torch.device("cpu"))
    agg = server.aggregate(
        client_updates=[update.delta_w],
        data_sizes=[update.data_size],
        stats=[update.stat],
    )
    metrics = server.update_global_model(agg.global_update, agg.stats_aggregated)
    assert "new_clip" in metrics


def run_all() -> None:
    test_config()
    test_dp_accountant()
    test_client_server_round()
    print("quick_test passed")


if __name__ == "__main__":
    run_all()
