"""
Graph comparison utilities for evaluating shared dependencies between graphs.
"""

from typing import Dict, Tuple, Set, Any, Union
from collections import defaultdict


class GraphComparison:
    """
    A class for comparing edge index dictionaries to evaluate shared dependencies
    between input and output nodes.
    
    The edge_index_dict is expected to have keys of the form:
    - (input, to, function)
    - (function, to, function) 
    - (function, to, output)
    """
    
    def __init__(self, reference_edge_index_dict: Dict[Tuple[str, str, str], Any]):
        """
        Initialize with a reference edge index dictionary.
        
        Args:
            reference_edge_index_dict: The baseline graph to compare against
        """
        self.reference_edge_index_dict = reference_edge_index_dict
        self.reference_dependencies = self._extract_dependencies(reference_edge_index_dict)
    
    def __call__(self, comparison_edge_index_dict: Dict[Tuple[str, str, str], Any]) -> Dict[str, Union[int, float, Set[Tuple[str, str]]]]:
        """
        Compare a new edge index dictionary against the reference.
        
        Args:
            comparison_edge_index_dict: The graph to compare against the reference
            
        Returns:
            Dictionary with comparison metrics including TP, FP, FN, and TN
        """
        comparison_dependencies = self._extract_dependencies(comparison_edge_index_dict)
        
        # Get all possible input-output pairs to calculate TNs
        all_possible_pairs = self._get_all_possible_pairs(self.reference_edge_index_dict, comparison_edge_index_dict)
        
        # Calculate metrics
        shared_dependencies = self.reference_dependencies.intersection(comparison_dependencies)
        
        true_positives = len(shared_dependencies)
        false_positives = len(comparison_dependencies - self.reference_dependencies)
        false_negatives = len(self.reference_dependencies - comparison_dependencies)
        
        # True negatives: pairs that are non-dependencies in both graphs
        all_dependencies = self.reference_dependencies.union(comparison_dependencies)
        true_negatives = len(all_possible_pairs - all_dependencies)
        
        total_reference = len(self.reference_dependencies)
        total_comparison = len(comparison_dependencies)
        total_possible = len(all_possible_pairs)
        
        return {
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'true_negatives': true_negatives,
            'total_reference_dependencies': total_reference,
            'total_comparison_dependencies': total_comparison,
            'total_possible_pairs': total_possible,
            'precision': true_positives / total_comparison if total_comparison > 0 else 0.0,
            'recall': true_positives / total_reference if total_reference > 0 else 0.0,
            'specificity': true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0.0,
            'accuracy': (true_positives + true_negatives) / total_possible if total_possible > 0 else 0.0,
            'shared_dependencies': shared_dependencies
        }
    
    def _extract_dependencies(self, edge_index_dict: Dict[Tuple[str, str, str], Any]) -> Set[Tuple[str, str]]:
        """
        Extract all input-to-output dependencies from the edge index dictionary.
        
        This method traces paths from input nodes to output nodes through function nodes
        to identify all dependencies between inputs and outputs.
        
        Args:
            edge_index_dict: Dictionary with edge information
            
        Returns:
            Set of (input_node, output_node) tuples representing dependencies
        """
        # Build adjacency lists for efficient path traversal
        graph = defaultdict(set)
        input_nodes = set()
        output_nodes = set()
        
        # Parse the edge index dictionary
        for (source_type, relation, target_type), edges in edge_index_dict.items():
            if hasattr(edges, 'numpy'):
                edges = edges.numpy()
            elif hasattr(edges, 'tolist'):
                edges = edges.tolist()
            
            # Handle different edge formats
            if len(edges) == 2:  # [source_indices, target_indices]
                source_indices, target_indices = edges
                for src, tgt in zip(source_indices, target_indices):
                    # Create node identifiers that include type information
                    src_node = f"{source_type}_{src}"
                    tgt_node = f"{target_type}_{tgt}"
                    
                    graph[src_node].add(tgt_node)
                    
                    if source_type == 'input':
                        input_nodes.add(src_node)
                    if target_type == 'output':
                        output_nodes.add(tgt_node)
        
        # Find all dependencies from inputs to outputs
        dependencies = set()
        
        for input_node in input_nodes:
            reachable_outputs = self._find_reachable_outputs(graph, input_node, output_nodes)
            for output_node in reachable_outputs:
                # Extract the actual node indices from the identifiers
                input_idx = input_node.split('_', 1)[1]
                output_idx = output_node.split('_', 1)[1]
                dependencies.add((input_idx, output_idx))
        
        return dependencies
    
    def _find_reachable_outputs(self, graph: Dict[str, Set[str]], start_node: str, 
                               output_nodes: Set[str]) -> Set[str]:
        """
        Find all output nodes reachable from a given start node using DFS.
        
        Args:
            graph: Adjacency list representation of the graph
            start_node: Starting node for the search
            output_nodes: Set of output nodes to look for
            
        Returns:
            Set of reachable output nodes
        """
        visited = set()
        reachable_outputs = set()
        
        def dfs(node):
            if node in visited:
                return
            visited.add(node)
            
            if node in output_nodes:
                reachable_outputs.add(node)
                return
            
            for neighbor in graph.get(node, set()):
                dfs(neighbor)
        
        dfs(start_node)
        return reachable_outputs
    
    def _get_all_possible_pairs(self, reference_edge_index_dict: Dict[Tuple[str, str, str], Any], 
                               comparison_edge_index_dict: Dict[Tuple[str, str, str], Any]) -> Set[Tuple[str, str]]:
        """
        Get all possible input-output pairs from both graphs.
        
        Args:
            reference_edge_index_dict: Reference graph
            comparison_edge_index_dict: Comparison graph
            
        Returns:
            Set of all possible (input, output) pairs
        """
        all_inputs = set()
        all_outputs = set()
        
        # Extract input and output nodes from both graphs
        for edge_dict in [reference_edge_index_dict, comparison_edge_index_dict]:
            for (source_type, relation, target_type), edges in edge_dict.items():
                if hasattr(edges, 'numpy'):
                    edges = edges.numpy()
                elif hasattr(edges, 'tolist'):
                    edges = edges.tolist()
                
                if len(edges) == 2:  # [source_indices, target_indices]
                    source_indices, target_indices = edges
                    
                    if source_type == 'input':
                        all_inputs.update(str(idx) for idx in source_indices)
                    if target_type == 'output':
                        all_outputs.update(str(idx) for idx in target_indices)
        
        # Generate all possible input-output pairs
        all_possible_pairs = set()
        for input_node in all_inputs:
            for output_node in all_outputs:
                all_possible_pairs.add((input_node, output_node))
        
        return all_possible_pairs
    
    def get_dependency_details(self, comparison_edge_index_dict: Dict[Tuple[str, str, str], Any]) -> Dict[str, Set]:
        """
        Get detailed information about the dependencies comparison.
        
        Args:
            comparison_edge_index_dict: The graph to compare against the reference
            
        Returns:
            Dictionary with detailed dependency sets
        """
        comparison_dependencies = self._extract_dependencies(comparison_edge_index_dict)
        
        return {
            'reference_dependencies': self.reference_dependencies,
            'comparison_dependencies': comparison_dependencies,
            'shared_dependencies': self.reference_dependencies.intersection(comparison_dependencies),
            'missing_dependencies': self.reference_dependencies - comparison_dependencies,
            'extra_dependencies': comparison_dependencies - self.reference_dependencies
        } 