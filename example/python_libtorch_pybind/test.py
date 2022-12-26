import torch  # must import first!
import python_example


a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])

print(python_example.tensor_add(a, b))
# print(python_example.omp_add(0, 4))