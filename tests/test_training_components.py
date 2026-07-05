"""Tests for metrics, mixup, EMA, optimizer, and scheduler."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from vit.config import ModelConfig, OptimConfig, SchedulerConfig
from vit.models import build_model
from vit.training.ema import ModelEma
from vit.training.metrics import AverageMeter, accuracy
from vit.training.mixup import MixupCutmix, SoftTargetCrossEntropy
from vit.training.optim import build_optimizer
from vit.training.scheduler import build_scheduler


# -- metrics ----------------------------------------------------------------
def test_average_meter() -> None:
    m = AverageMeter()
    m.update(1.0, n=2)
    m.update(4.0, n=1)
    assert m.average == pytest.approx(2.0)
    m.reset()
    assert m.average == 0.0


def test_average_meter_rejects_negative_n() -> None:
    with pytest.raises(ValueError):
        AverageMeter().update(1.0, n=-1)


def test_accuracy_perfect() -> None:
    logits = torch.tensor([[3.0, 0.0], [0.0, 5.0]])
    targets = torch.tensor([0, 1])
    assert accuracy(logits, targets, topk=(1,)) == [100.0]


def test_accuracy_topk() -> None:
    logits = torch.tensor([[0.1, 0.2, 0.9, 0.0]])  # top-2 = classes 2,1
    targets = torch.tensor([1])
    top1, top2 = accuracy(logits, targets, topk=(1, 2))
    assert top1 == 0.0
    assert top2 == 100.0


def test_accuracy_empty_batch() -> None:
    assert accuracy(torch.empty(0, 3), torch.empty(0, dtype=torch.long)) == [0.0]


def test_accuracy_validation_errors() -> None:
    with pytest.raises(ValueError, match="2-D"):
        accuracy(torch.randn(3), torch.tensor([0]))
    with pytest.raises(ValueError, match="same batch"):
        accuracy(torch.randn(2, 3), torch.tensor([0]))
    with pytest.raises(ValueError, match="exceeds"):
        accuracy(torch.randn(2, 3), torch.tensor([0, 1]), topk=(5,))


# -- mixup ------------------------------------------------------------------
def test_mixup_disabled_returns_smoothed_onehot() -> None:
    mix = MixupCutmix(num_classes=4, mixup_alpha=0.0, cutmix_alpha=0.0, label_smoothing=0.1)
    assert not mix.enabled
    x = torch.randn(2, 3, 8, 8)
    y = torch.tensor([0, 1])
    out_x, soft = mix(x, y)
    torch.testing.assert_close(out_x, x)
    torch.testing.assert_close(soft.sum(-1), torch.ones(2), rtol=1e-5, atol=1e-5)
    assert soft[0, 0] > soft[0, 1]  # correct class has the higher mass


def test_mixup_produces_soft_targets() -> None:
    mix = MixupCutmix(num_classes=4, mixup_alpha=0.5, cutmix_alpha=0.0, seed=0)
    x = torch.randn(6, 3, 8, 8)
    y = torch.randint(0, 4, (6,))
    out_x, soft = mix(x, y)
    assert out_x.shape == x.shape
    torch.testing.assert_close(soft.sum(-1), torch.ones(6), rtol=1e-5, atol=1e-5)


def test_cutmix_path_runs() -> None:
    mix = MixupCutmix(num_classes=4, mixup_alpha=0.0, cutmix_alpha=1.0, seed=1)
    x = torch.randn(4, 3, 16, 16)
    y = torch.randint(0, 4, (4,))
    out_x, soft = mix(x, y)
    assert out_x.shape == x.shape
    assert soft.shape == (4, 4)


def test_soft_target_cross_entropy_matches_ce_for_hard_labels() -> None:
    logits = torch.randn(5, 3)
    target = torch.randint(0, 3, (5,))
    hard = torch.nn.functional.one_hot(target, 3).float()
    soft_loss = SoftTargetCrossEntropy()(logits, hard)
    ref = nn.CrossEntropyLoss()(logits, target)
    assert soft_loss.item() == pytest.approx(ref.item(), abs=1e-5)


# -- EMA --------------------------------------------------------------------
def _model() -> nn.Module:
    return build_model(
        ModelConfig(image_size=16, patch_size=4, num_classes=5, embed_dim=16, depth=1, num_heads=2)
    )


def test_ema_tracks_towards_model() -> None:
    model = _model()
    ema = ModelEma(model, decay=0.5)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)  # shift the online model
    ema.update(model)
    # EMA should move halfway (decay=0.5) toward the updated weights.
    online = dict(model.named_parameters())
    for name, ep in ema.module.named_parameters():
        assert not torch.allclose(ep, online[name])


def test_ema_rejects_bad_decay() -> None:
    with pytest.raises(ValueError):
        ModelEma(_model(), decay=1.5)


def test_ema_state_dict_roundtrip() -> None:
    model = _model()
    ema = ModelEma(model, decay=0.9)
    state = ema.state_dict()
    ema2 = ModelEma(model, decay=0.9)
    ema2.load_state_dict(state)
    for a, b in zip(ema.module.parameters(), ema2.module.parameters(), strict=True):
        torch.testing.assert_close(a, b)


# -- optim / scheduler ------------------------------------------------------
def test_optimizer_no_decay_grouping() -> None:
    model = _model()
    opt = build_optimizer(model, OptimConfig(name="adamw", weight_decay=0.05))
    decay_group, no_decay_group = opt.param_groups
    assert decay_group["weight_decay"] == 0.05
    assert no_decay_group["weight_decay"] == 0.0
    # Biases / norms / pos_embed / cls_token must live in the no-decay group.
    assert len(no_decay_group["params"]) > 0


def test_optimizer_sgd() -> None:
    opt = build_optimizer(_model(), OptimConfig(name="sgd", momentum=0.9))
    assert opt.__class__.__name__ == "SGD"


def test_scheduler_warmup_then_cosine() -> None:
    model = _model()
    opt = build_optimizer(model, OptimConfig(lr=0.1))
    sched = build_scheduler(
        opt,
        SchedulerConfig(name="cosine", warmup_epochs=2, min_lr=0.0, warmup_start_lr=0.0),
        steps_per_epoch=10,
        epochs=10,
        base_lr=0.1,
    )
    lrs = []
    for _ in range(100):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    assert lrs[0] < lrs[20]  # warming up
    assert lrs[20] == pytest.approx(0.1, abs=1e-6)  # peak after warmup
    assert lrs[-1] < lrs[20]  # decayed by the end


def test_scheduler_none_is_constant() -> None:
    opt = build_optimizer(_model(), OptimConfig(lr=0.05))
    sched = build_scheduler(
        opt, SchedulerConfig(name="none"), steps_per_epoch=5, epochs=2, base_lr=0.05
    )
    for _ in range(10):
        opt.step()
        sched.step()
    assert opt.param_groups[0]["lr"] == pytest.approx(0.05)
