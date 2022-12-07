import torch  # must import first!
import python_example


a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])

print(python_example.add(a, b))
