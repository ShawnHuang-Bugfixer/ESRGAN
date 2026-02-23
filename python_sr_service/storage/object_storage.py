from abc import ABC, abstractmethod
from typing import List


class ObjectStorage(ABC):
    @abstractmethod
    def download(self, object_key: str, local_path: str) -> None:
        pass

    @abstractmethod
    def upload(self, local_path: str, object_key: str) -> None:
        pass

    @abstractmethod
    def exists(self, object_key: str) -> bool:
        pass

    @abstractmethod
    def delete(self, object_key: str) -> None:
        pass

    @abstractmethod
    def list_objects(self, prefix: str) -> List[str]:
        pass
