import os
import shutil
from dataclasses import dataclass


@dataclass(frozen=True)
class TaskWorkspace:
    task_root: str
    input_path: str
    output_path: str


class WorkspaceManager:
    def __init__(self, root_dir: str):
        self._root_dir = root_dir

    def create(self, task_id: int, attempt: int) -> TaskWorkspace:
        task_root = os.path.join(self._root_dir, f'{task_id}_{attempt}')
        os.makedirs(task_root, exist_ok=True)
        input_path = os.path.join(task_root, 'input')
        output_path = os.path.join(task_root, 'output')
        os.makedirs(input_path, exist_ok=True)
        os.makedirs(output_path, exist_ok=True)
        return TaskWorkspace(task_root=task_root, input_path=input_path, output_path=output_path)

    def cleanup(self, workspace: TaskWorkspace) -> None:
        if os.path.exists(workspace.task_root):
            shutil.rmtree(workspace.task_root, ignore_errors=True)
