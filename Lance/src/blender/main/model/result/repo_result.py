from typing import TypedDict, Literal


class RepoResult(TypedDict):
    kind: Literal["repo"]
    path: str  # root_path