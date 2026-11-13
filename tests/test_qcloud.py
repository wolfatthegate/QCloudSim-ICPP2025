"""
Smoke and regression tests for the QCloud simulator.

Run from the repository root:  python -m pytest -q
"""
import os
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from QCloud import (  # noqa: E402
    IBM_Brussels, IBM_Kawasaki, IBM_Kyiv, IBM_Quebec, IBM_Strasbourg,
    ParallelBroker, QCloudSimEnv,
)
from QCloud.rl_utils import action_to_allocation, build_observation  # noqa: E402

CALI = os.path.join("QCloud", "calibration", "03_2025")
JOBS_20 = os.path.join(REPO_ROOT, "synth_job_batches", "20-large-circuits.csv")


def make_devices():
    """The five IBM Eagle r3 devices used in the ICPP 2025 experiments."""
    return [
        IBM_Strasbourg(env=None, name="ibm_strasbourg", cali_filepath=os.path.join(CALI, "ibm_strasbourg_calibrations_2025-03-29T05_16_26Z.csv"), printlog=False),
        IBM_Brussels(env=None, name="ibm_brussels", cali_filepath=os.path.join(CALI, "ibm_brussels_calibrations_2025-03-30T02_20_13Z.csv"), printlog=False),
        IBM_Kyiv(env=None, name="ibm_kyiv", cali_filepath=os.path.join(CALI, "ibm_kyiv_calibrations_2025-03-30T02_24_45Z.csv"), printlog=False),
        IBM_Quebec(env=None, name="ibm_quebec", cali_filepath=os.path.join(CALI, "ibm_quebec_calibrations_2025-03-30T02_51_33Z.csv"), printlog=False),
        IBM_Kawasaki(env=None, name="ibm_kawasaki", cali_filepath=os.path.join(CALI, "ibm_kawasaki_calibrations_2025-03-30T03_15_05Z.csv"), printlog=False),
    ]


def run_sim(mode, file_path=JOBS_20):
    env = QCloudSimEnv(devices=make_devices(), broker_class=ParallelBroker,
                       job_feed_method="dispatcher", file_path=file_path,
                       allocation_mode=mode, printlog=False)
    env.run()
    return env


# ---------------------------------------------------------------- rl_utils

def test_allocation_matches_legacy_rounding_when_feasible():
    action = np.array([0.15, 0.21, 0.20, 0.20, 0.24])
    ratios = action / (action.sum() + 1e-6)
    legacy = [int(round(r * 188)) for r in ratios]
    legacy[0] += 188 - sum(legacy)
    assert action_to_allocation(action, 188, capacities=[127] * 5) == legacy


@pytest.mark.parametrize("action", [
    [1.0, 0.0, 0.0, 0.0, 0.0],        # everything on one device -> exceeds capacity
    [0.0, 0.0, 0.0, 0.0, 0.0],        # degenerate all-zero action
    [0.6, 0.6, 0.0, 0.0, 0.0],
    [-1.0, 2.0, 0.5, 0.0, 0.1],       # negative weights are clipped
])
def test_allocation_is_feasible(action):
    alloc = action_to_allocation(action, 250, capacities=[127] * 5)
    assert sum(alloc) == 250
    assert all(0 <= a <= 127 for a in alloc)


def test_allocation_rejects_impossible_job():
    with pytest.raises(ValueError):
        action_to_allocation([1, 1], 300, capacities=[127, 127])


def test_observation_shape_and_range():
    devices = make_devices()
    obs = build_observation(30, devices)
    assert obs.shape == (16,)
    assert obs[0] == pytest.approx(30 / 50)
    assert np.all(obs[1::3] == 1.0)  # every device fully free at start (127/127)


# -------------------------------------------------------------- simulation

@pytest.mark.parametrize("mode", ["speed", "fidelity", "fair"])
def test_heuristic_modes_complete(mode):
    env = run_sim(mode)
    records = env.job_records_manager.get_job_records()
    assert len(records) == 20
    for rec in records.values():
        assert 0 < rec["fidelity"] <= 1


def test_rl_mode_completes_and_loads_model_once():
    pytest.importorskip("stable_baselines3")
    env = run_sim("rlbase")
    records = env.job_records_manager.get_job_records()
    assert len(records) == 20
    assert all(0 < r["fidelity"] <= 1 for r in records.values())
    assert env.qcloud._rl_model is not None  # cached on the QCloud instance


def test_generator_mode_runs():
    env = QCloudSimEnv(devices=make_devices(), broker_class=ParallelBroker,
                       job_feed_method="generator", allocation_mode="speed", printlog=False)
    env.run(until=50)
    assert len(env.job_records_manager.get_job_records()) > 0


def test_maintenance_enabled_does_not_crash():
    devices = make_devices()
    for d in devices:
        d.maintenance_switch = True
    env = QCloudSimEnv(devices=devices, broker_class=ParallelBroker,
                       job_feed_method="dispatcher", file_path=JOBS_20,
                       allocation_mode="speed", printlog=False)
    env.run(until=2000)


def test_default_calibration_file_resolves_from_any_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    device = IBM_Kyiv(env=None, printlog=False)  # no cali_filepath -> package default
    assert device.error_score > 0
