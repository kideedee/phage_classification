from typing import Any

import torch
from torch_geometric.data import Data


class BipartiteData(Data):
    def _add_other_feature(self, other_feature):
        self.other_feature = other_feature

    # def __inc__(self, key, value):
    #     if key == 'edge_index':
    #         return torch.tensor([[self.x_src.size(0)], [self.x_dst.size(0)]])
    #     else:
    #         return super(BipartiteData, self).__inc__(key, value)

    def __inc__(self, key: str, value: Any, *args, **kwargs) -> Any:
        if key == 'edge_index':
            return torch.tensor([[self.x_src.size(0)], [self.x_dst.size(0)]])
        else:
            return super(BipartiteData, self).__inc__(key, value)
