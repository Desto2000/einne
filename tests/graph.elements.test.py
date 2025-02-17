import gc
import unittest
import weakref
from collections import deque

import networkx as nx
import torch

from einne.internal.graph.elements import Graph, Node


class TestGraphElements(unittest.TestCase):
    def test_graph_initialization_default_values(self):
        graph = Graph()
        self.assertIsInstance(graph.computation_history, deque)
        self.assertEqual(graph.computation_history.maxlen, 1000)
        self.assertIsInstance(graph.nx_graph, nx.DiGraph)
        self.assertEqual(len(graph.nx_graph.nodes), 0)
        self.assertEqual(len(graph.nx_graph.edges), 0)

    def test_add_node(self):
        graph = Graph()
        test_node = Node(name="TestNode")
        test_node.forward = lambda x: x
        graph.add_node("TestNode", test_node)
        self.assertIn("TestNode", graph.nx_graph.nodes)
        self.assertEqual(graph.nx_graph.nodes["TestNode"]["node"], test_node)

    def test_add_edge(self):
        graph = Graph()
        node1 = Node(name="Node1")
        node1.forward = lambda x: x
        node2 = Node(name="Node2")
        node2.forward = lambda x: x
        graph.add_node("Node1", node1)
        graph.add_node("Node2", node2)
        graph.add_edge("Node1", "Node2", edge_type="custom")

        self.assertIn("Node1", graph.nx_graph.nodes)
        self.assertIn("Node2", graph.nx_graph.nodes)
        self.assertIn(("Node1", "Node2"), graph.nx_graph.edges)
        self.assertEqual(graph.nx_graph.edges[("Node1", "Node2")]["type"], "custom")

    def test_log_computation(self):
        graph = Graph()
        node_id = "test_node"
        computation_type = "forward"
        data = {"inputs": [1, 2, 3], "outputs": [4, 5, 6]}

        graph.log_computation(node_id, computation_type, data)

        self.assertEqual(len(graph.computation_history), 1)
        logged_event = graph.computation_history[0]
        self.assertEqual(logged_event["node_id"], node_id)
        self.assertEqual(logged_event["type"], computation_type)
        self.assertEqual(logged_event["data"], data)

    def test_node_initialization_with_custom_parameters(self):
        custom_name = "CustomNode"
        custom_threshold = 100
        node = Node(name=custom_name, logging=True, threshold_steps=custom_threshold)
        node.forward = lambda x: x

        self.assertEqual(node._name, custom_name)
        self.assertEqual(node._threshold_steps, custom_threshold)
        self.assertIsNone(node._graph)
        self.assertEqual(node._fwd_step, 0)
        self.assertEqual(node._bck_step, 0)
        self.assertIsInstance(node._inputs, deque)
        self.assertIsInstance(node._outputs, deque)
        self.assertIsInstance(node._grad_inputs, deque)
        self.assertIsInstance(node._grad_outputs, deque)

    def test_graph_property_propagation(self):
        parent_node = Node(name="ParentNode")
        parent_node.forward = lambda x: x
        child_node1 = Node(name="ChildNode")
        child_node1.forward = lambda x: x
        child_node2 = Node(name="ChildNode")
        child_node2.forward = lambda x: x
        parent_node.add_module("child1", child_node1)
        parent_node.add_module("child2", child_node2)

        new_graph = Graph()
        parent_node.graph = new_graph

        self.assertEqual(parent_node.graph, new_graph)
        self.assertEqual(child_node1.graph, new_graph)
        self.assertEqual(child_node2.graph, new_graph)
        self.assertIn(parent_node.name, new_graph.nx_graph.nodes)
        self.assertIn(child_node1.name, new_graph.nx_graph.nodes)
        self.assertIn(child_node2.name, new_graph.nx_graph.nodes)
        self.assertIn((parent_node.name, child_node1.name), new_graph.nx_graph.edges)
        self.assertIn((parent_node.name, child_node2.name), new_graph.nx_graph.edges)

    def test_clear_history(self):
        node = Node(name="TestNode", logging=True)
        node.forward = lambda x: x
        node._inputs.extend([torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])])
        node._outputs.extend([torch.tensor([7, 8, 9]), torch.tensor([10, 11, 12])])
        node._grad_inputs.extend(
            [torch.tensor([13, 14, 15]), torch.tensor([16, 17, 18])]
        )
        node._grad_outputs.extend(
            [torch.tensor([19, 20, 21]), torch.tensor([22, 23, 24])]
        )

        node.clear_history()

        self.assertEqual(len(node._inputs), 0)
        self.assertEqual(len(node._outputs), 0)
        self.assertEqual(len(node._grad_inputs), 0)
        self.assertEqual(len(node._grad_outputs), 0)

    def test_forward_hook_logging(self):
        node = Node(name="TestNode", logging=True, threshold_steps=5)
        node.forward = lambda x: x
        graph = Graph()
        node.graph = graph

        # Simulate forward passes
        for _ in range(6):
            node(torch.randn(1, 10))

        # Check if computation was logged
        self.assertEqual(len(graph.computation_history), 1)
        log_entry = graph.computation_history[0]
        self.assertEqual(log_entry["node_id"], node.name)
        self.assertEqual(log_entry["type"], "forward")
        self.assertIn("inputs", log_entry["data"])
        self.assertIn("outputs", log_entry["data"])

        # Check if inputs and outputs were stored
        self.assertEqual(len(node._inputs), 1)
        self.assertEqual(len(node._outputs), 1)

    def test_backward_hook_logging(self):
        node = Node(name="TestNode", logging=True, threshold_steps=5)
        node.forward = lambda x: x
        graph = Graph()
        node.graph = graph

        # Create a simple tensor and perform operations to trigger backward
        for _ in range(5):
            x = torch.randn(1, 10, requires_grad=True)
            y = node(x)
            loss = y.sum()
            loss.backward()

        # Check if computation was logged
        self.assertGreater(len(graph.computation_history), 0)
        last_computation = graph.computation_history[-1]
        self.assertEqual(last_computation["node_id"], node.name)
        self.assertEqual(last_computation["type"], "backward")
        self.assertIn("grad_inputs", last_computation["data"])
        self.assertIn("grad_outputs", last_computation["data"])

    def test_node_create_graph_on_first_call(self):
        node = Node(name="TestNode")
        node.forward = lambda x: x
        self.assertIsNone(node.graph)

        # Call the node with first=True
        node(torch.randn(1, 10), first=True)

        # Check if a new Graph instance was created
        self.assertIsNotNone(node.graph)
        self.assertIsInstance(node.graph, Graph)

        # Verify that the node was added to the graph
        self.assertIn(node.name, node.graph.nx_graph.nodes)
        self.assertEqual(node.graph.nx_graph.nodes[node.name]["node"], node)

    def test_node_del_removes_from_graph(self):
        graph = Graph()
        node = Node(name="TestNode")
        node.forward = lambda x: x
        node.graph = graph

        # Ensure the node is in the graph
        self.assertIn(node.name, graph.nx_graph.nodes)

        # Manually call __del__ method
        node.__del__()

        # Check if the node has been removed from the graph
        self.assertNotIn(node.name, graph.nx_graph.nodes)
        self.assertIsNone(node.graph)

    def test_node_del_handles_none_graph(self):
        node = Node(name="TestNode")
        node.forward = lambda x: x
        node.graph = None

        # Manually call __del__ method
        node.__del__()

        # No exception should be raised, and the method should complete successfully
        self.assertIsNone(node.graph)

    def test_node_del_called_on_garbage_collection(self):
        graph = Graph()
        node = Node(name="TestNode")
        node.forward = lambda x: x
        node.graph = graph
        node2 = Node(name="TestNode2")
        node2.forward = lambda x: x

        # Create a weak reference to the node
        weak_ref = weakref.ref(node2)

        node2(node(1))

        # Delete the strong reference to the node
        del node2

        # Force garbage collection
        gc.collect()

        # Check if the node has been garbage collected
        self.assertIsNone(weak_ref())

        # Verify that the node has been removed from the graph
        self.assertNotIn("TestNode", graph.nx_graph.nodes)

    def test_node_del_no_error_when_name_not_in_graph(self):
        graph = Graph()
        node = Node(name="TestNode")
        node.forward = lambda x: x
        node.graph = graph

        # Remove the node from the graph manually
        graph.remove_node(node.name)

        # Manually call __del__ method
        node.__del__()

        # Check if no exception was raised and the graph attribute is set to None
        self.assertIsNone(node.graph)


if __name__ == "__main__":
    unittest.main()
