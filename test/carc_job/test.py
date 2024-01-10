#  Copyright (c) Meta Platforms, Inc. and affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import os
import re


def extract_master_addr(nodelist_str):
    master_addr = nodelist_str.split(',')[0]
    match = re.match(r'(\w+-)\[(\w+)', master_addr)
    if match:
        master_addr = match.group(1) + match.group(2)
    return master_addr


nodelist_str = os.environ.get('SLURM_JOB_NODELIST')
print(nodelist_str)

machine = extract_master_addr(nodelist_str)
print(machine)
