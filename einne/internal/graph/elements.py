import collections
from dataclasses import dataclass, field
from typing import Any, Optional

import matplotlib.pyplot as plt
import networkx as nx
import torch


@dataclass
class Graph:
    computation_history: collections.deque = field(
        default_factory=lambda: collections.deque(maxlen=1000)
    )
    nx_graph: nx.DiGraph = field(default_factory=nx.DiGraph)  # NetworkX graph instance

    def add_node(self, node_id: str, node: "Node") -> None:
        """Add a node to the NetworkX graph."""
        self.nx_graph.add_node(node_id, node=node)  # Add node to NetworkX graph

    def add_edge(
        self, from_node: str, to_node: str, edge_type: str = "forward"
    ) -> None:
        """Add a directed edge between nodes on NetworkX graph."""
        self.nx_graph.add_edge(
            from_node, to_node, type=edge_type
        )  # Add edge to NetworkX graph

    def log_computation(self, node_id: str, computation_type: str, data: Any) -> None:
        """Log computation events in the graph."""
        self.computation_history.append(
            {"node_id": node_id, "type": computation_type, "data": data}
        )

    def visualize(self):
        """Enhanced visualization for clarity using networkx and matplotlib."""
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(self.nx_graph, seed=42)
        nx.draw_networkx_nodes(
            self.nx_graph,
            pos,
            node_size=2500,
            node_color="skyblue",
            label="Main Modules",
        )
        nx.draw_networkx_edges(self.nx_graph, pos, arrows=True)
        nx.draw_networkx_labels(
            self.nx_graph,
            pos,
            font_size=9,
            font_weight="bold",
            verticalalignment="center",
        )
        edge_labels = nx.get_edge_attributes(self.nx_graph, "type")
        nx.draw_networkx_edge_labels(
            self.nx_graph,
            pos,
            edge_labels=edge_labels,
            font_size=8,
            label_pos=0.5,
            verticalalignment="center_baseline",
        )
        plt.legend(loc="upper left")
        plt.title("Enhanced Graph Visualization with Submodules")
        plt.show()


class Node(torch.nn.Module):
    _graph: Optional[Graph]

    def __init__(self, name=None, threshold_steps=50, max_history: int = 100):
        super().__init__()
        self.name = name or self.__class__.__name__
        self._graph = None
        self._threshold_steps = threshold_steps
        self._init_history_buffers(max_history)
        self._init_hooks()
        self._init_step_counters()

    def _init_history_buffers(self, max_history: int) -> None:
        """Initialize history buffers for inputs, outputs, and gradients."""
        self._inputs = collections.deque(maxlen=max_history)
        self._outputs = collections.deque(maxlen=max_history)
        self._grad_inputs = collections.deque(maxlen=max_history)
        self._grad_outputs = collections.deque(maxlen=max_history)

    def _init_hooks(self) -> None:
        """Register hooks for forward and backward passes."""
        self.register_forward_hook(self._forward_hook)
        self.register_full_backward_hook(self._backward_hook)

    def _init_step_counters(self) -> None:
        """Initialize counters for forward and backward steps."""
        self._fwd_step = 0
        self._bck_step = 0

    def _forward_hook(self, module, inputs, outputs):
        """Hook to log forward pass data if the step threshold is met."""
        self._fwd_step += 1
        if self._should_log_step(self._fwd_step):
            self._log_forward(inputs, outputs)
            self._fwd_step = 0
        return outputs

    def _backward_hook(self, module, grad_inputs, grad_outputs):
        """Hook to log backward pass data if the step threshold is met."""
        self._bck_step += 1
        if self._should_log_step(self._bck_step):
            self._log_backward(grad_inputs, grad_outputs)
            self._bck_step = 0
        return grad_inputs

    def _should_log_step(self, step: int) -> bool:
        """Determine if the current step should trigger logging."""
        return step >= self._threshold_steps

    def _log_forward(self, inputs, outputs) -> None:
        """Log forward pass inputs and outputs."""
        self._inputs.append(self._detach_and_cpu(inputs))
        self._outputs.append(self._detach_and_cpu(outputs))
        if self._graph:
            self._graph.log_computation(
                self.name, "forward", {"inputs": inputs, "outputs": outputs}
            )

    def _log_backward(self, grad_inputs, grad_outputs) -> None:
        """Log backward pass gradients."""
        self._grad_inputs.append(self._detach_and_cpu(grad_inputs, allow_none=True))
        self._grad_outputs.append(self._detach_and_cpu(grad_outputs, allow_none=True))
        if self._graph:
            self._graph.log_computation(
                self.name,
                "backward",
                {"grad_inputs": grad_inputs, "grad_outputs": grad_outputs},
            )

    @staticmethod
    def _detach_and_cpu(tensors, allow_none=False):
        """Detach tensors from the computation graph and move them to CPU."""
        with torch.no_grad():
            if isinstance(tensors, torch.Tensor):
                return tensors.detach().cpu()
            return [
                (
                    t.detach().cpu()
                    if isinstance(t, torch.Tensor)
                    else (None if allow_none else t)
                )
                for t in tensors
            ]

    @property
    def graph(self) -> Optional[Graph]:
        """Get the graph associated with the node."""
        return self._graph

    @graph.setter
    def graph(self, new_graph: Graph) -> None:
        """Set a new graph and update the graph structure."""
        if new_graph is not self._graph:
            self._graph = new_graph
            self._update_graph_structure()

    def _update_graph_structure(self) -> None:
        """Update the graph with the node and its children."""
        if self._graph:
            self._graph.add_node(self.name, self)
            for name, child in self.named_children():
                if isinstance(child, Node):
                    child.graph = self._graph
                    self._graph.add_edge(self.name, child.name)

    def _reset(self) -> None:
        """Reset the graph structure."""
        self._update_graph_structure()

    def __call__(self, *args, first=False, **kwargs):
        """Override the call method to reset the graph and perform the forward pass."""
        if first and self._graph is None:
            self.graph = Graph()
        self._reset()
        return super().__call__(*args, **kwargs)

    def clear_history(self) -> None:
        """Clear the history buffers."""
        for buffer in [
            self._inputs,
            self._outputs,
            self._grad_inputs,
            self._grad_outputs,
        ]:
            buffer.clear()
