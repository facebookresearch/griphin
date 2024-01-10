#  Copyright (c) Meta Platforms, Inc. and affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import torch  # must import first!
import python_example


a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])

print(python_example.tensor_add(a, b))
