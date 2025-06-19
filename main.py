import torch
from typing import List
from torch import Tensor


def example():
    # Let 'a' be the layer's raw output
    a = torch.randn([1, 20])

    # First part is the high-level action
    # Further parts are the low-level choices
    parts = [p.softmax(dim=1) for p in a.split([3, 5, 4, 8], dim=1)]

    # High level action modulates low-level actions
    hl = parts[0].split(1, dim=1)
    low_parts = [h * p for h, p in zip(hl, parts[1:])]

    final = torch.cat(low_parts, dim=1)


class TwoLevelLinear(torch.nn.Linear):

    def __init__(self, in_features: int, low_level_classes: List[int]):
        if len(low_level_classes) < 2:
            raise ValueError("This class only makes sense with at least two high-level options.")
        self.num_high_level = len(low_level_classes)
        self.low_level_nums = [n for n in low_level_classes]
        self.part_sizes = [len(low_level_classes)] + [n for n in low_level_classes]
        out_features = self.num_high_level + sum(self.low_level_nums)
        super().__init__(in_features, out_features)

    def forward(self, input: Tensor) -> Tensor:
        raw_output = super().forward(input)
        parts = [p.softmax(dim=1) for p in raw_output.split(self.part_sizes, dim=1)]
        high_level_gates = parts[0].split(1, dim=1)
        low_level_probs = [h * p for h, p in zip(high_level_gates, parts[1:])]
        return torch.cat(low_level_probs, dim=1)
