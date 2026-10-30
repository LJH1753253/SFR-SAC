import torch

from cloud24.plan import configure
from v2.runtime import run_learner


def test_episode_checkpoint_resume_preserves_training(tmp_path):
    cfg = configure("split_exact", n=4, layout=1000, train_seed=0, small=True)
    cfg["execution"] = dict(device="cpu", require_cuda=False, tf32=False)
    continuous, resumed = tmp_path / "continuous", tmp_path / "resumed"
    reference = run_learner(cfg, "safe", 4, 0, continuous, device="cpu")
    interrupted = run_learner(cfg, "safe", 4, 0, resumed, device="cpu", stop_after=1)
    assert interrupted is None
    assert (resumed / "checkpoint.pt").exists()
    restored = run_learner(cfg, "safe", 4, 0, resumed, device="cpu")
    left = torch.load(continuous / "model.pt", map_location="cpu", weights_only=True)
    right = torch.load(resumed / "model.pt", map_location="cpu", weights_only=True)
    for module in left["modules"]:
        for name, tensor in left["modules"][module].items():
            assert torch.equal(tensor, right["modules"][module][name]), (module, name)
    for metric in ("cost_percent", "energy_mJ_device_slot", "bfsr_percent", "final_lambda"):
        assert restored[metric] == reference[metric]
