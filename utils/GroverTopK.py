from math import ceil, log2, pi, floor, asin, sqrt
from typing import List, Dict, Any, Optional, Tuple
from qiskit import QuantumCircuit
from qiskit.circuit.library import MCXGate
import qiskit
import qiskit_ibm_runtime
from qiskit_aer import Aer


class GroverTopK:
    """
    A class that implements Grover's algorithm to select top-k contexts based on similarity scores.
    
    This class uses quantum computing principles, specifically Grover's search algorithm,
    to efficiently find the top-k most relevant contexts from a list of results based on
    similarity scores.
    """

    def __init__(self, initial_threshold: float = 0.75, shots: int = 10000, top_k: int = 3):
        """
        Initialize the GroverTopK selector.

        Args:
            initial_threshold: Initial threshold for similarity scores (0.0 to 1.0)
            shots: Number of shots for the quantum circuit execution, 
            top_k: Number of top results to return (default: 3)

        Raises:
            ValueError: If threshold is not between 0 and 1 or shots is not positive
        """
        self.top_k = top_k
        self._validate_init_params(initial_threshold, shots)
        self.initial_threshold = initial_threshold
        self.shots = shots
        self.backend = Aer.get_backend('aer_simulator')

    def _validate_init_params(self, threshold: float, shots: int) -> None:
        """
        Validate initialization parameters.
        
        Args:
            threshold: Threshold value to validate
            shots: Number of shots to validate
            
        Raises:
            ValueError: If parameters are invalid
        """
        if not 0 <= threshold <= 1:
            raise ValueError("Threshold must be between 0 and 1")
        if not isinstance(shots, int) or shots <= 0:
            raise ValueError("Shots must be a positive integer")

    def _grover_oracle(self, marked_indices: List[int], n_qubits: int) -> QuantumCircuit:
        """
        Create a Grover oracle that marks the specified indices.
        
        Args:
            marked_indices: List of indices to mark in the oracle
            n_qubits: Number of qubits in the circuit
            
        Returns:
            QuantumCircuit: The oracle circuit
        """
        binary_states = [format(i, f"0{n_qubits}b") for i in marked_indices] # Convert indices to binary strings (e.g., 3 -> '011')
        qc = QuantumCircuit(n_qubits) # Create a quantum circuit with n_qubits
        for target in binary_states:
            rev_target = target[::-1] # Reverse the binary string to match qubit order
            zero_inds = [ind for ind in range(n_qubits) if rev_target[ind] == "0"] # Find indices where the target is '0'
            qc.x(zero_inds) # Apply X gates to those indices to flip them to |1> for MCX operation
            qc.append(MCXGate(n_qubits - 1), list(range(n_qubits))) # Apply multi-controlled X gate (MCX) to mark the state, also known as Toffoli gate
            qc.x(zero_inds) # Flip back the indices to their original state
        return qc

    def _diffusion_operator(self, n_qubits: int) -> QuantumCircuit:
        """
        Create a diffusion operator for Grover's algorithm.
        
        This operator implements the reflection about the mean amplitude,
        which is a key component of Grover's algorithm.
        
        Args:
            n_qubits: Number of qubits in the circuit
            
        Returns:
            QuantumCircuit: The diffusion operator circuit
        """
        qc = QuantumCircuit(n_qubits)
        qc.h(range(n_qubits)) # Apply Hadamard gates to all qubits, transforming |0> to superposition
        qc.x(range(n_qubits))
        qc.h(n_qubits - 1) # Apply Hadamard gate to the last qubit
        qc.append(MCXGate(n_qubits - 1), qargs=list(range(n_qubits))) # Apply multi-controlled X gate (Toffoli) to flip the last qubit
        qc.h(n_qubits - 1)
        qc.x(range(n_qubits))
        qc.h(range(n_qubits))
        return qc

    def _adjust_threshold(self, results: List[Dict[str, Any]], k: int) -> Tuple[float, List[int]]:
        """
        Dynamically adjust the threshold to get approximately k results.
        
        This method iteratively adjusts the threshold value until approximately k
        results have similarity scores above the threshold.
        
        Args:
            results: List of result dictionaries with similarity scores
            k: Desired number of results
            
        Returns:
            Tuple containing:
                - Final adjusted threshold
                - List of marked indices that satisfy the threshold
        """
        threshold = self.initial_threshold
        step = 0.1
        min_step = 0.001
        max_attempts = 100
        attempts = 0
        
        while attempts < max_attempts:
            marked_indices = [i for i, res in enumerate(results) 
                             if res.get("similarity_score", 0) > threshold]
            if len(marked_indices) > k:
                threshold += step
                step = max(step / 2, min_step)
            elif len(marked_indices) < k:
                threshold -= step
                step = max(step / 2, min_step)
            else:
                break
            attempts += 1
            
        return threshold, [i for i, res in enumerate(results) 
                          if res.get("similarity_score", 0) > threshold]

    def select(self, results: List[Dict[str, Any]], k: Optional[int] = None) -> Dict[str, Any]:
        """
        Select top-k contexts from the results list using Grover's algorithm.
        
        This method uses quantum computing to find the top-k results with similarity
        scores above a dynamically adjusted threshold.
        
        Args:
            results: List of result dictionaries, each containing at least a "similarity_score" key
            k: Number of top results to return (default: 3)
            
        Returns:
            Dict containing:
                - contexts: List of top-k result dictionaries
                - threshold: Final similarity score threshold used
                - qiskit_version: Version of Qiskit used
                - runtime_version: Version of Qiskit Runtime used
                
        Raises:
            ValueError: If input parameters are invalid
            TypeError: If input parameters have wrong types
        """
        self._validate_select_params(results, k)
        k = k if k is not None else self.top_k

        n_items = len(results)
        if n_items == 0 or k == 0:
            return {"contexts": [], "threshold": self.initial_threshold, 
                    "qiskit_version": qiskit.__version__, 
                    "runtime_version": qiskit_ibm_runtime.__version__}

        n_qubits = ceil(log2(n_items))
        
        # Adjust threshold and get marked indices
        threshold, marked_indices = self._adjust_threshold(results, k)
        
        if len(marked_indices) == 0:
            return {"contexts": [], "threshold": threshold, 
                    "qiskit_version": qiskit.__version__, 
                    "runtime_version": qiskit_ibm_runtime.__version__}

        # Create and run the Grover circuit
        oracle = self._grover_oracle(marked_indices, n_qubits)
        diffuser = self._diffusion_operator(n_qubits)

        optimal_num_iterations = floor(
            pi / (4 * asin(sqrt(len(marked_indices) / 2**n_qubits)))
        )
        
        qc = self._create_grover_circuit(n_qubits, oracle, diffuser, optimal_num_iterations)
        
        result = self.backend.run(qc, shots=self.shots).result()
        dist = result.get_counts()

        # Extract top-k results
        top_k_results = self._extract_top_k(dist, results, threshold, k)

        return {
            "contexts": top_k_results,
            "threshold": threshold,
            "qiskit_version": qiskit.__version__,
            "runtime_version": qiskit_ibm_runtime.__version__
        }
    
    def _create_grover_circuit(self, n_qubits: int, oracle: QuantumCircuit, 
                               diffuser: QuantumCircuit, iterations: int) -> QuantumCircuit:
        """
        Create the complete Grover circuit.
        
        Args:
            n_qubits: Number of qubits
            oracle: Oracle circuit
            diffuser: Diffusion operator circuit
            iterations: Number of Grover iterations
            
        Returns:
            QuantumCircuit: Complete Grover circuit with measurements
        """
        qc = QuantumCircuit(n_qubits)
        # Create even superposition of all basis states
        qc.h(range(n_qubits))
        for _ in range(iterations):
            qc.compose(oracle, inplace=True)
            qc.compose(diffuser, inplace=True)
        # Measure all qubits
        qc.measure_all()
        return qc
    
    def _extract_top_k(self, dist: Dict[str, int], results: List[Dict[str, Any]], 
                       threshold: float, k: int) -> List[Dict[str, Any]]:
        """
        Extract top-k results from measurement outcomes.
        
        Args:
            dist: Distribution of measurement outcomes
            results: Original results list
            threshold: Similarity score threshold
            k: Number of top results to extract
            
        Returns:
            List[Dict[str, Any]]: Top-k results
        """
        sorted_dist = sorted(dist.items(), key=lambda item: item[1], reverse=True)
        top_k_results = []
        
        for state, _ in sorted_dist:
            idx = int(state, 2)
            if (0 <= idx < len(results) and 
                results[idx].get("similarity_score", 0) > threshold):
                top_k_results.append(results[idx])
            if len(top_k_results) == k:
                break
                
        return top_k_results
    
    def _validate_select_params(self, results: List[Dict[str, Any]], k: int) -> None:
        """
        Validate parameters for the select method.
        
        Args:
            results: Results list to validate
            k: k value to validate
            
        Raises:
            ValueError: If parameters are invalid
            TypeError: If parameters have wrong types
        """
        if not isinstance(results, list):
            raise TypeError("Results must be a list")
        
        if not isinstance(k, int):
            raise TypeError("k must be an integer")
            
        if k < 0:
            raise ValueError("k must be non-negative")
            
        for i, result in enumerate(results):
            if not isinstance(result, dict):
                raise TypeError(f"Each result must be a dictionary (issue at index {i})")
            if "similarity_score" not in result:
                raise ValueError(f"Each result must have a 'similarity_score' key (missing at index {i})")
            if not isinstance(result.get("similarity_score"), (int, float)):
                raise TypeError(f"Similarity score must be a number (issue at index {i})")