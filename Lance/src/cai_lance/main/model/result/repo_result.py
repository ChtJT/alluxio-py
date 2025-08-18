from typing import TypedDict, Literal


class RepoResult(TypedDict):
    kind: Literal["repo"]
    path: str  # 根目录