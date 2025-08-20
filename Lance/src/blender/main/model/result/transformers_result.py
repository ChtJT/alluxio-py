from typing import Any, TypedDict, Literal


class TransformersResult(TypedDict):
    kind: Literal["transformers"]
    model: Any
    tokenizer: Any