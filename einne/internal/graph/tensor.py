import torch
from node import Node

class EinTensor(torch.Tensor, Node):
    node: Node
    def __new__(cls, *args, **kwargs):
        cls.node = Node()
        return super().__new__(cls, *args, **kwargs)


if __name__ == '__main__':
    x = torch.add(EinTensor([1, 2, 3]), torch.tensor([1, 2, 3]))
    print(x.node)
