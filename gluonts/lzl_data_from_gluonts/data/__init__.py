from .data_utils import whetherInputInList

__all__ = [
    'whetherInputInList'
]
for item in __all__:
    if hasattr(item, "__module__"):
        setattr(item, "__module__", __name__)