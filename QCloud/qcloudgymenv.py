import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

class QCloudGymEnv(gym.Env):
    """
    An environment for quantum job allocation.
    The state consists of normalized job parameters and device features,
    and the action is a vector of allocation fractions for a fixed maximum number of devices.
    """
    
    metadata = {"render_modes": ["human"], "render_fps": None}
    
    def __init__(self, devices, job_records_manager, printlog=True):
        super(QCloudGymEnv, self).__init__()
        self.devices = devices  # List of device objects from your QCloud simulation
        self.job_records_manager = job_records_manager
        self.printlog = printlog

        self.MAX_DEVICES = 5  # Maximum number of devices considered
        
        # Define the observation space: one job parameter plus for each device three features:
        # container level, error score, and CLOPS.
        self.observation_space = spaces.Box(
            low=0,
            high=1, 
            shape=(1 + self.MAX_DEVICES * 3,),  # 1 for job param, 3 per device
            dtype=np.float32
        )
        
        # Define the action space as continuous allocation fractions for each device.
        self.action_space = spaces.Box(
            low=0,
            high=1,
            shape=(self.MAX_DEVICES,),
            dtype=np.float32
        )
        self.job = None

    def reset(self, *, seed: int = None, options: dict = None):
        # Optionally set a random seed for reproducibility.
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Create a new job with randomized parameters.
        self.job = {
            "num_qubits": random.randint(10, 50),
            "depth": random.uniform(1, 10),
            "num_shots": random.randint(1000, 10000),
            "two_qubits": random.randint(5, 20)
        }
        # Build the state vector.
        # Normalize the job's num_qubits by 50.
        state = [self.job["num_qubits"] / 50.0]
        
        # For each device (up to MAX_DEVICES), extract and normalize device features.
        for i in range(self.MAX_DEVICES):
            if i < len(self.devices):
                device = self.devices[i]
                # Normalize container level; adjust divisor from 100.0 to 150.0 to ensure value ≤1.
                state.append(device.container.level / 127.0)
                # Assuming device.error_score is already between 0 and 1.
                state.append(device.error_score)
                # Normalize CLOPS; adjust divisor based on its expected maximum value.
                state.append(device.clops / 1e6)
            else:
                # Pad missing devices with zeros.
                state.extend([0, 0, 0])
                
        return np.array(state, dtype=np.float32), {}

    def step(self, action):
        # Normalize the action to obtain allocation fractions.
        allocation_ratios = action / (np.sum(action) + 1e-6)
        allocations = []
        total_qubits = self.job["num_qubits"]
        for ratio in allocation_ratios:
            allocated = int(round(ratio * total_qubits))
            allocations.append(allocated)
        # Adjust to ensure the total matches job["num_qubits"].
        diff = total_qubits - sum(allocations)
        allocations[0] += diff  # Simple correction
        
        # Simulate the allocation by computing fidelities.
        fidelities = []
        for i, device in enumerate(self.devices[:self.MAX_DEVICES]):
            allocated_qubits = allocations[i]
            if allocated_qubits <= 0:
                fidelity = 0
            else:
                # Fidelity computation as in your QCloud methods.
                single_qubit_fidelity = (1 - device.avg_single_qubit_error) ** self.job["depth"]
                readout_fidelity = (1 - device.avg_readout_error) ** ((self.job["num_qubits"]) ** 0.5)
                two_qubit_fidelity = (1 - device.avg_two_qubit_error) ** ((self.job["two_qubits"]) ** 0.25)
                fidelity = single_qubit_fidelity * readout_fidelity * two_qubit_fidelity
            fidelities.append(fidelity)
        
        # Use the average fidelity as the reward.
        avg_fidelity = np.mean(fidelities)
        reward = avg_fidelity
        
        # In this simple environment, the episode ends after one allocation.
        terminated = True   # Natural terminal condition.
        truncated = False   # No external truncation.
        info = {"allocations": allocations, "fidelity": avg_fidelity}
        
        # For the next episode, return a fresh observation.
        observation, _ = self.reset()
        return observation, reward, terminated, truncated, info