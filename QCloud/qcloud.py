# qcloud.py

from QCloud.dependencies import *
import numpy as np
import random
import math
from stable_baselines3 import PPO  # Replace with PPO/SAC/TD3 if using continuous action spaces


class QCloud:
    def __init__(self, env, devices, job_records_manager, allocation_mode="fast", printlog=True):
        """
        Initializes the QCloud class.

        Parameters:
        - env: The SimPy environment.
        - devices: The list of quantum devices in the cloud.
        """
        self.env = env
        self.devices = devices
        self.job_records = {}  # Dictionary to track job lifecycle events
        self.job_records_manager = job_records_manager
        # self.error_aware = False
        self.printlog = printlog

        # Mapping strategies to functions
        self.allocation_strategies = {
            "speed": self.fast_allocate_large_job,
            "fidelity": self.smart_allocate_large_job, 
            "fair": self.balanced_allocate_large_job, 
            "rlbase": self.rl_allocate_large_job
        }

        # Set the allocation mode
        if allocation_mode in self.allocation_strategies:
            self.allocation_function = self.allocation_strategies[allocation_mode]
        else:
            raise ValueError(f"Invalid allocation mode: {allocation_mode}. Choose from {list(self.allocation_strategies.keys())}.")        

    def allocate_job(self, job, devices):
        """
        Dynamically calls the selected job allocation function.

        Parameters:
        - job: The QJob object representing the job.
        - devices: List of quantum devices.
        """
        return self.allocation_function(job, devices)
    
    def log_job_event(self, job_id, event_type, timestamp):
        """
        Logs a job event with a timestamp.

        Parameters:
        - job_id: The ID of the job.
        - event_type: The type of event ('arrival', 'start', or 'finish').
        """
        if job_id not in self.job_records:
            self.job_records[job_id] = {}
        self.job_records[job_id][event_type] = timestamp

    def get_event_logger(self):
        """
        Returns a callback function for logging job events.
        """
        return self.log_job_event

    def device_comm(self, job, device1, device2, qubits_required, feedback=False):
        comm_time = qubits_required * 0.02  # Adjust communication latency
        if self.printlog:
            print(f"{self.env.now:.2f}: Communication between {device1.name} and {device2.name} started.")
        
        self.job_records_manager.log_job_event(job.job_id, 'comm_time', round(comm_time,4))
        
        yield self.env.timeout(comm_time)  # Simulate delay

        if feedback:
            # Example: Simulate a feedback-dependent operation
            if self.printlog:
                print(f"{self.env.now:.2f}: Measurement feedback received. Adjusting circuit on {device2.name}.")
            yield self.env.timeout(0.02)  # Simulate classical computation delay
        if self.printlog:
            print(f"{self.env.now:.2f}: Communication between {device1.name} and {device2.name} finished.")
 
    def calculate_process_time(self, device, job):
        """
        Calculate processing time considering IBM-specific metrics.
        """
        M = 100
        K = 10
        S = job.num_shots
        D = math.log2(device.qvol)

        return  M * K * S * D / device.clops / 60
    
    def fast_allocate_large_job(self, job, devices):
        """
        Allocate a large job across multiple devices with error-aware or error-agnostic scheduling.

        Parameters:
        - job: The QJob object representing the job.
        - devices: List of quantum devices.
        """
        if self.printlog:
            for d in devices: 
                print(f"Available qubits for {d.name}: {d.container.level}, CLOPS: {d.clops}, ERR: {d.error_score}")

        # Step 1: Identify eligible devices
        eligible_devices = []
        for device in devices:
            if device.container.level >= job.num_qubits / len(devices):
                selected_subgraph = select_vertices_fast(device, job.num_qubits // len(devices), job.job_id)
                if selected_subgraph:                    
                    eligible_devices.append((device, selected_subgraph, device.error_score, device.clops))

        # Step 2: Ensure sufficient devices are available
        while len(eligible_devices) < 2:
            if self.printlog:
                print(f"{self.env.now:.2f}: Insufficient connected devices to allocate job #{job.job_id}. Retrying...")
            yield self.env.timeout(1)
            eligible_devices = []
            for device in devices:
                if device.container.level >= job.num_qubits / len(devices):
                    selected_subgraph = select_vertices_fast(device, job.num_qubits // len(devices), job.job_id)
                    if selected_subgraph:
                        eligible_devices.append((device, selected_subgraph, device.error_score, device.clops))


        # Step 3: Select devices to allocate
        split_qubits = job.num_qubits // len(eligible_devices)
        remainder_qubits = job.num_qubits % len(eligible_devices)
        allocated_devices = []
        
        # Step 4: Split job across selected devices optimally
        for idx, (device, subgraph, error_score, clops) in enumerate(eligible_devices):
            allocated_qubits = split_qubits + (1 if idx < remainder_qubits else 0)
            with device.resource.request(priority=2) as req:
                yield device.container.get(allocated_qubits)
                allocated_devices.append((device, allocated_qubits, subgraph, clops))
                if self.printlog:
                    print(f"{self.env.now:.2f}: Job #{job.job_id} allocated {allocated_qubits} qubits on {device.name} (error score: {error_score:.4f}).")

                # Log per-device allocation start
                self.job_records_manager.log_job_event(job.job_id, 'devc_proc', round(self.env.now, 4))

        # Step 5: process inter-device communication by pairing devices 
        for i in range(len(allocated_devices) - 1):
            device1, qubits1, _, _ = allocated_devices[i]
            device2, qubits2, _, _ = allocated_devices[i + 1]
            yield self.env.process(self.device_comm(job, device1, device2, qubits1 + qubits2))

        # Step 6: Process communication time  
        process_times = [self.calculate_process_time(device, job) for device, _, _, _ in allocated_devices]
        yield self.env.timeout(max(process_times))  # Max execution time across devices

        # Step 7: Release resources after completion
        for device, allocated_qubits, _, _ in allocated_devices:
            yield device.container.put(allocated_qubits)
            self.job_records_manager.log_job_event(job.job_id, 'devc_finish', round(self.env.now, 4))
            if self.printlog:
                print(f"{self.env.now:.2f}: Job #{job.job_id} completed on {device.name}.")
             
        # Step 8: Compute Fidelity
        fidelities = []
        for device, _, _, _ in allocated_devices:  
            single_qubit_fidelity = (1 - device.avg_single_qubit_error) ** job.depth             
            readout_fidelity = (1 - device.avg_readout_error) ** ((job.num_qubits // len(allocated_devices)) ** 0.5 ) 
            two_qubit_fidelity = (1 - device.avg_two_qubit_error) ** ((job.two_qubits // len(allocated_devices)) ** 0.25 )
            two_qubit_fidelity = 1
            device_fidelity = single_qubit_fidelity * readout_fidelity * two_qubit_fidelity
            fidelities.append(device_fidelity)

        # Step 9: Compute final fidelity with communication penalty
        avg_fidelity = sum(fidelities) / len(fidelities) if fidelities else -1.0
        num_connections = len(allocated_devices) - 1  # Number of times devices communicate
        communication_penalty = 0.94 ** num_connections  # Decay for each communication
        final_fidelity = avg_fidelity * communication_penalty  # Final fidelity
        
        self.job_records_manager.log_job_event(job.job_id, 'fidelity', round(final_fidelity,4))
 


    def smart_allocate_large_job(self, job, devices):
        """
        Allocate a large job across multiple devices using either a simple or error-aware allocation strategy.

        Parameters:
        - job: The QJob object representing the job.
        - devices: List of quantum devices.
        """
        if self.printlog:
            for d in devices: 
                print(f"Available qubits for {d.name}: {d.container.level}, CLOPS: {d.clops}, ERR: {d.error_score}")

        # Step 1: Identify eligible devices
        eligible_devices = []
        for device in devices:
            if device.container.level >= job.num_qubits / len(devices):
                selected_subgraph = select_vertices_fast(device, job.num_qubits // len(devices), job.job_id)
                if selected_subgraph:   
                    eligible_devices.append((device, selected_subgraph, device.error_score, device.clops))
        
        # Step 2: Ensure sufficient devices are available
        while len(eligible_devices) < 2:
            if self.printlog:
                print(f"{self.env.now:.2f}: Insufficient connected devices to allocate job #{job.job_id}. Retrying...")
            yield self.env.timeout(1)
            eligible_devices = []
            for device in devices:
                if device.container.level >= job.num_qubits / len(devices):
                    selected_subgraph = select_vertices_fast(device, job.num_qubits // len(devices), job.job_id)
                    if selected_subgraph:
                        eligible_devices.append((device, selected_subgraph, device.error_score, device.clops))

        # Sort by lowest error score first then minimize the device usage
        eligible_devices.sort(key=lambda x: x[2])  

        num_devices_to_use = min(len(eligible_devices), math.ceil(job.num_qubits / max(d.container.level for d, _, _, _ in eligible_devices)))
        selected_devices = eligible_devices[:num_devices_to_use]  # Pick best devices

        # Step 3: Select devices to allocate
        split_qubits = job.num_qubits // num_devices_to_use
        remainder_qubits = job.num_qubits % num_devices_to_use
        allocated_devices = []
    
        # Step 4: Split job across selected devices optimally
        for idx, (device, subgraph, error_score, clops) in enumerate(selected_devices):
            allocated_qubits = split_qubits + (1 if idx < remainder_qubits else 0)
            with device.resource.request(priority=2) as req:
                yield device.container.get(allocated_qubits)
                allocated_devices.append((device, allocated_qubits, subgraph, clops))
                if self.printlog:
                    print(f"{self.env.now:.2f}: Job #{job.job_id} allocated {allocated_qubits} qubits on {device.name} (error score: {error_score:.4f}).")

                # Log per-device allocation start
                self.job_records_manager.log_job_event(job.job_id, 'devc_proc', round(self.env.now, 4))

        # Step 5: process inter-device communication by pairing devices 
        for i in range(len(allocated_devices) - 1):
            device1, qubits1, _, _ = allocated_devices[i]
            device2, qubits2, _, _ = allocated_devices[i + 1]
            yield self.env.process(self.device_comm(job, device1, device2, qubits1 + qubits2))

        # Step 6: Process communication time  
        process_times = [self.calculate_process_time(device, job) for device, _, _, _ in allocated_devices]
        yield self.env.timeout(max(process_times))

        # Step 7: Release resources after completion
        for device, allocated_qubits, _, _ in allocated_devices:
            yield device.container.put(allocated_qubits)
            self.job_records_manager.log_job_event(job.job_id, 'devc_finish', round(self.env.now, 4))
            if self.printlog:
                print(f"{self.env.now:.2f}: Job #{job.job_id} completed on {device.name}.")

        # Step 8: Compute Fidelity
        fidelities = []
        for device, _, _, _ in allocated_devices:
            single_qubit_fidelity = (1 - device.avg_single_qubit_error) ** job.depth
            readout_fidelity = (1 - device.avg_readout_error) ** ((job.num_qubits // len(allocated_devices)) ** 0.5 )
            two_qubit_fidelity = (1 - device.avg_two_qubit_error) ** ((job.two_qubits // len(allocated_devices)) ** 0.25 )
            two_qubit_fidelity = 1
            device_fidelity = single_qubit_fidelity * readout_fidelity * two_qubit_fidelity
            fidelities.append(device_fidelity)

        # Step 9: Compute final fidelity with communication penalty
        avg_fidelity = sum(fidelities) / len(fidelities) if fidelities else -1.0
        num_connections = len(allocated_devices) - 1  
        communication_penalty = 0.94 ** num_connections  
        final_fidelity = avg_fidelity * communication_penalty  

        self.job_records_manager.log_job_event(job.job_id, 'fidelity', round(final_fidelity,4))

    def balanced_allocate_large_job(self, job, devices, max_retries=5):
        retries = 0
        if self.printlog:
            for d in devices: 
                print(f"Available qubits for {d.name}: {d.container.level}, CLOPS: {d.clops}, ERR: {d.error_score}")

        # Identify eligible devices with a retry mechanism up to max_retries
        eligible_devices = []
        # shut off the retries
        while len(eligible_devices) < 2: # and retries < max_retries:
            eligible_devices = []
            for device in devices:
                if device.container.level >= job.num_qubits / len(devices):
                    selected_subgraph = select_vertices_fast(device, job.num_qubits // len(devices), job.job_id)
                    if selected_subgraph:
                        # Calculate a weighted score that benefits high speed (CLOPS) and low error scores
                        # Avoid division by zero by adding a small epsilon value
                        weight = device.clops / (device.error_score + 1e-6)
                        eligible_devices.append((device, selected_subgraph, device.error_score, device.clops, weight))
            if len(eligible_devices) < 2:
                if self.printlog:
                    print(f"{self.env.now:.2f}: Insufficient connected devices for job #{job.job_id} in balanced mode. Retrying...")
                yield self.env.timeout(1)
                retries += 1

        if len(eligible_devices) < 2:
            raise Exception(f"Job #{job.job_id} could not be allocated after {max_retries} retries in balanced mode.")

        # Sort devices by computed weight (higher is better)
        eligible_devices.sort(key=lambda x: x[4], reverse=True)

        # Allocate qubits on the selected devices as in fast/smart methods
        num_devices = len(eligible_devices)
        split_qubits = job.num_qubits // num_devices
        remainder_qubits = job.num_qubits % num_devices
        allocated_devices = []

        for idx, (device, subgraph, error_score, clops, weight) in enumerate(eligible_devices):
            allocated_qubits = split_qubits + (1 if idx < remainder_qubits else 0)
            with device.resource.request(priority=2) as req:
                yield device.container.get(allocated_qubits)
                allocated_devices.append((device, allocated_qubits, subgraph, clops))
                if self.printlog:
                    print(f"{self.env.now:.2f}: Balanced mode – Job #{job.job_id} allocated {allocated_qubits} qubits on {device.name} (ERR: {error_score:.4f}, weight: {weight:.2f}).")
                self.job_records_manager.log_job_event(job.job_id, 'devc_proc', round(self.env.now, 4))

        # Process inter-device communication and processing just like in fast/smart
        for i in range(len(allocated_devices) - 1):
            device1, q1, _, _ = allocated_devices[i]
            device2, q2, _, _ = allocated_devices[i + 1]
            yield self.env.process(self.device_comm(job, device1, device2, q1 + q2))

        process_times = [self.calculate_process_time(device, job) for device, _, _, _ in allocated_devices]
        yield self.env.timeout(max(process_times))

        for device, allocated_qubits, _, _ in allocated_devices:
            yield device.container.put(allocated_qubits)
            self.job_records_manager.log_job_event(job.job_id, 'devc_finish', round(self.env.now, 4))
            if self.printlog:
                print(f"{self.env.now:.2f}: Balanced mode – Job #{job.job_id} completed on {device.name}.")

        # Fidelity computations (same as before)
        fidelities = []
        for device, _, _, _ in allocated_devices:
            single_qubit_fidelity = (1 - device.avg_single_qubit_error) ** job.depth
            readout_fidelity = (1 - device.avg_readout_error) ** ((job.num_qubits // len(allocated_devices)) ** 0.5)
            two_qubit_fidelity = (1 - device.avg_two_qubit_error) ** ((job.two_qubits // len(allocated_devices)) ** 0.25)
            two_qubit_fidelity = 1
            device_fidelity = single_qubit_fidelity * readout_fidelity * two_qubit_fidelity
            fidelities.append(device_fidelity)

        avg_fidelity = sum(fidelities) / len(fidelities) if fidelities else -1.0
        num_connections = len(allocated_devices) - 1  
        communication_penalty = 0.94 ** num_connections  
        final_fidelity = avg_fidelity * communication_penalty  
        self.job_records_manager.log_job_event(job.job_id, 'fidelity', round(final_fidelity, 4))
        
    def rl_allocate_large_job(self, job, devices):
        MAX_DEVICES = 5
        # Construct the state vector similar to the one used in your Gym environment.
        # Ensure the normalization factors here match those in the training environment.
        state = [job.num_qubits / 50.0]
        for i in range(MAX_DEVICES):
            if i < len(devices):
                device = devices[i]
                # Adjust normalization: using 150.0 instead of 100.0 if needed.
                state.append(device.container.level / 150.0)
                state.append(device.error_score)
                state.append(device.clops / 1e6)  # Adjust based on maximum expected value.
            else:
                state.extend([0, 0, 0])
        state = np.array(state, dtype=np.float32)

        # Load your trained RL model.
        # Consider loading this once (e.g., during __init__) and reusing it for performance.
        model = PPO.load("qcloud_ppo_model-new")

        # Get the agent's action.
        action, _ = model.predict(state, deterministic=True)

        # Normalize action into allocation ratios.
        allocation_ratios = action / (np.sum(action) + 1e-6)

        allocations = []
        for ratio in allocation_ratios:
            allocated = int(round(ratio * job.num_qubits))
            allocations.append(allocated)

        # Adjust to ensure total allocation equals job.num_qubits.
        diff = job.num_qubits - sum(allocations)
        allocations[0] += diff  # simple correction for rounding

        # Now follow the steps of allocation: request resources, etc.
        allocated_devices = []
        for idx, device in enumerate(devices[:MAX_DEVICES]):
            allocated_qubits = allocations[idx]
            if allocated_qubits > 0:
                # For asynchronous operation in a simulation, consider yielding this request.
                with device.resource.request(priority=2) as req:
                    while device.container.level < allocated_qubits:
                        if self.printlog:
                            print(f"{self.env.now:.2f}: Waiting for {allocated_qubits} qubits on {device.name} (available: {device.container.level})")
                        yield self.env.timeout(1)

                    # 3. Actually take the qubits (will now succeed immediately)
                    yield device.container.get(allocated_qubits)

                    # 4. Record the allocation
                    allocated_devices.append((device, allocated_qubits))
                    if self.printlog:
                        print(f"{self.env.now:.2f}: RL mode – Job #{job.job_id} allocated {allocated_qubits} qubits on {device.name}.")
                    self.job_records_manager.log_job_event(job.job_id, 'devc_proc', round(self.env.now, 4))

        # for d in devices: 
        #     print(f"device name: {d.name}, qubit available: {d.container.level}")
                    
        for i in range(len(allocated_devices) - 1):
            device1, q1 = allocated_devices[i]
            device2, q2 = allocated_devices[i + 1]
            yield self.env.process(self.device_comm(job, device1, device2, q1 + q2))

        process_times = [self.calculate_process_time(device, job) for device, _ in allocated_devices]
        yield self.env.timeout(max(process_times))

        for device, allocated_qubits in allocated_devices:
            yield device.container.put(allocated_qubits)
            self.job_records_manager.log_job_event(job.job_id, 'devc_finish', round(self.env.now, 4))
            if self.printlog:
                print(f"{self.env.now:.2f}: Job #{job.job_id} completed on {device.name}.")

        # Fidelity computations (same as before)
        fidelities = []
        for device, _ in allocated_devices:
            single_qubit_fidelity = (1 - device.avg_single_qubit_error) ** job.depth
            readout_fidelity = (1 - device.avg_readout_error) ** ((job.num_qubits // len(allocated_devices)) ** 0.5)
            two_qubit_fidelity = (1 - device.avg_two_qubit_error) ** ((job.two_qubits // len(allocated_devices)) ** 0.25)
            two_qubit_fidelity = 1 # supressing two qubits fidelity
            device_fidelity = single_qubit_fidelity * readout_fidelity * two_qubit_fidelity
            fidelities.append(device_fidelity)

        avg_fidelity = sum(fidelities) / len(fidelities) if fidelities else -1.0
        num_connections = len(allocated_devices) - 1  
        communication_penalty = 0.94 ** num_connections  
        final_fidelity = avg_fidelity * communication_penalty  
        self.job_records_manager.log_job_event(job.job_id, 'fidelity', round(final_fidelity, 4))