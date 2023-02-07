from pathlib import Path
import os.path as osp


def get_root_path():
    current_dir = Path(__file__)
    project_dir = [p for p in current_dir.parents if p.parts[-1] == 'graph_engine'][0]
    return project_dir


def get_data_path():
    return osp.join(get_root_path(), 'data')
