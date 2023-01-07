from pathlib import Path


def get_root_path():
    current_dir = Path(__file__)
    project_dir = [p for p in current_dir.parents if p.parts[-1] == 'graph_engine'][0]
    return project_dir
