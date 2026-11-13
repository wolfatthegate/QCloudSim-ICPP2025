# rl_utils.py
"""
Helpers shared by the RL training environment (QCloudGymEnv) and the RL
allocation strategy in QCloud, so that training and inference build the
observation and convert actions to qubit allocations in exactly the same way.
"""

import numpy as np

MAX_DEVICES = 5
JOB_QUBIT_NORM = 50.0  # job size normalisation used when the PPO model was trained
CLOPS_NORM = 1e6


def build_observation(num_qubits, devices, max_devices=MAX_DEVICES):
    """
    Observation = [job_qubits / 50] + [free_qubits / capacity, error_score, clops / 1e6] per device.

    Missing devices (fewer than ``max_devices``) are zero-padded.
    """
    state = [num_qubits / JOB_QUBIT_NORM]
    for i in range(max_devices):
        if i < len(devices):
            device = devices[i]
            state.append(device.container.level / device.number_of_qubits)
            state.append(device.error_score)
            state.append(device.clops / CLOPS_NORM)
        else:
            state.extend([0.0, 0.0, 0.0])
    return np.array(state, dtype=np.float32)


def action_to_allocation(action, total_qubits, capacities=None):
    """
    Convert a policy action (non-negative weights, one per device) into integer
    qubit counts that sum to ``total_qubits``.

    The rounding matches the original implementation (round each share, put the
    rounding difference on device 0). If that result is infeasible -- a negative
    count, or more qubits than a device physically has -- it falls back to a
    largest-remainder split that respects ``capacities``.
    """
    action = np.clip(np.asarray(action, dtype=np.float64), 0.0, None)
    n = len(action)
    ratios = action / (action.sum() + 1e-6) if action.sum() > 0 else np.full(n, 1.0 / n)

    allocations = [int(round(r * total_qubits)) for r in ratios]
    allocations[0] += total_qubits - sum(allocations)

    caps = list(capacities) if capacities is not None else [total_qubits] * n
    if all(0 <= a <= c for a, c in zip(allocations, caps)):
        return allocations

    if total_qubits > sum(caps):
        raise ValueError(f"Job needs {total_qubits} qubits but devices only have {sum(caps)} in total.")

    # Largest-remainder split, then move overflow from full devices to devices with spare room.
    raw = ratios * total_qubits
    allocations = [min(int(np.floor(x)), c) for x, c in zip(raw, caps)]
    order = np.argsort(-(raw - np.floor(raw)))  # biggest fractional part first
    remaining = total_qubits - sum(allocations)
    while remaining > 0:
        progressed = False
        for i in order:
            if remaining == 0:
                break
            if allocations[i] < caps[i]:
                allocations[i] += 1
                remaining -= 1
                progressed = True
        if not progressed:  # unreachable given the capacity check above
            raise ValueError("Could not place all qubits within device capacities.")
    return allocations
