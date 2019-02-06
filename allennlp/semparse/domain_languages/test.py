
from typing import Dict, List, NamedTuple, Set, Tuple

class Row(NamedTuple):
    # Maps column names to cell values.
    values: Dict[str, str]


x = {"abc": "def"}

print(Row(x))

