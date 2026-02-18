from threading import Lock
from typing import ClassVar


class SingletonMeta(type):
    """
    Thread-safe Singleton metaclass.
    """

    _instances: ClassVar = {}
    _lock: Lock = Lock()

    def __call__(cls, *args, **kwargs):
        with cls._lock:
            if cls not in cls._instances:
                instance = super().__call__(*args, **kwargs)
                cls._instances[cls] = instance

        return cls._instances[cls]
