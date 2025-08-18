import hashlib
from typing import Union

def stable_label_id(val: Union[str, int, float]) -> Union[int, float]:
    """
    统一将标签转为可训练的数值：
    - str -> 稳定 hash（不受 PYTHONHASHSEED 影响）
    - int/float -> 原样数值
    """
    if isinstance(val, str):
        h = hashlib.md5(val.encode("utf-8")).hexdigest()
        return int(h, 16) % (2**31 - 1)
    if isinstance(val, (int,)):
        return int(val)
    return float(val)
