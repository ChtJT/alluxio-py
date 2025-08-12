from typing import TypedDict, Literal, Mapping


class ArrowResult(TypedDict):
    kind: Literal["arrow"]
    splits: Mapping[str, str]  # split -> 保存到磁盘的目录
