from __future__ import annotations

"""Quick smoke tests for core modules."""

from types import SimpleNamespace

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
    from client.client import FLClient, FLClientWorker
    from server.server import FLServer

    class TinyNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = nn.Linear(8, 4)

        def forward(self, x):
            return self.fc(x)

    cfg = SimpleNamespace(
        client=SimpleNamespace(
            training_strategy="standard",
            local_epochs=1,
            lr=0.01,
            momentum=0.0,
            weight_decay=0.0,
            alpha=0.0,
            beta=0.0,
            contrastive_margin=1.0,
            mu=0.0,
            sam_rho=0.05,
            sam_eps=1e-12,
            stat_type="l2_norm",
        ),
        compressor=SimpleNamespace(
            type="identity",
            importance_strategy="grad_normalized",
            topk_strategy="weighted_layer_norm",
            topk_ratio=1.0,
            layer_weight_method="mean",
            use_residual=False,
        ),
        dp=SimpleNamespace(
            enabled=True,
            use_heterogeneous_noise=False,
            min_relative_noise=0.3,
            max_relative_noise=3.0,
            template_ema=0.9,
            use_global_importance_for_topk=True,
            importance_aggregation="tempered_weighted",
            importance_weight_beta=0.5,
            importance_normalization="mean",
            global_local_mix_lambda=0.4,
            enable_importance_freeze=True,
            importance_freeze_warmup_rounds=10,
            importance_freeze_threshold=1e-3,
            importance_freeze_patience=3,
            local_importance_refresh_interval_after_freeze=5,
            clip_count_noise_multiplier=0.1,
        ),
        server=SimpleNamespace(
            initial_clip=1.0,
            server_lr=1.0,
            aggregation_weight_strategy="data_size",
            clip_update_method="adaptive",
            target_quantile=0.7,
            clip_lr=0.2,
            ema_alpha=0.8,
        ),
        trainer=SimpleNamespace(
            keep_client_model_on_device=False,
            min_cuda_free_ratio=0.15,
            cuda_empty_cache_each_round=False,
        ),
    )

    model = TinyNet()
    dataset = TensorDataset(torch.randn(64, 8), torch.randint(0, 4, (64,)))
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    client = FLClient(
        client_id=0,
        dataloader=loader,
        cfg=cfg,
        device=torch.device("cpu"),
    )
    worker = FLClientWorker(model=model, cfg=cfg, device=torch.device("cpu"))

    global_weights = model.state_dict()
    update = worker.train_and_upload(
        client=client,
        global_weights=global_weights,
        clip_norm=1.0,
        local_noise_std=0.1,
    )

    assert update.data_size == 64
    assert len(update.delta_w) > 0

    server = FLServer(model=model, cfg=cfg, device=torch.device("cpu"))
    agg = server.aggregate(client_updates=[update])
    metrics = server.update_global_model(agg.noisy_update, agg.stats_aggregated)
    assert "new_clip" in metrics


def run_all() -> None:
    test_config()
    test_dp_accountant()
    test_client_server_round()
    print("quick_test passed")


if __name__ == "__main__":
    run_all()
