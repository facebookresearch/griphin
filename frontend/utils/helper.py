#  Copyright (c) Meta Platforms, Inc. and affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from pathlib import Path
import os.path as osp
import re

import torch


def get_root_path():
    current_dir = Path(__file__)
    project_dir = [p for p in current_dir.parents if p.parts[-1] == 'graph_engine'][0]
    return project_dir


def get_data_path():
    return osp.join(get_root_path(), 'data')


def extract_master_addr(nodelist_str):
    master_addr = nodelist_str.split(',')[0]
    match = re.match(r'(\w+-)\[(\w+)', master_addr)
    if match:
        master_addr = match.group(1) + match.group(2)
    return master_addr


def extract_hostname_list(nodelist_str):
    """
    :param nodelist_str: e.g., "a02-[01,06,20],b05-14,e02-[42,50,57-62,65-69,79]"
    :return: hostname list
    """
    pattern = r'([a-z]\d{2})-\[(.*?)\]'
    matches = re.findall(pattern, nodelist_str)

    hostnames = []
    for prefix, values in matches:
        value_list = re.split(r'[,]', values)
        for value in value_list:
            if '-' in value:
                start, end = map(int, value.split('-'))
                for i in range(start, end + 1):
                    hostnames.append(f'{prefix}-{str(i).zfill(2)}')
            else:
                hostnames.append(f'{prefix}-{value}')

    # Handling single entries
    single_entries = re.findall(r'([a-z]\d{2}-\d{2})', nodelist_str)
    hostnames.extend(single_entries)

    return hostnames


def extract_core_global_ids(parts):
    part_core_global_ids = []
    for i in range(len(parts)):
        part = parts[i]  # parts is a dict
        core_mask = part.ndata['inner_node'].type(torch.bool)
        part_global_id = part.ndata['orig_id']
        part_core_global_ids.append(part_global_id[core_mask])
    return part_core_global_ids
