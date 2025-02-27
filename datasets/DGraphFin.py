import os.path as osp
from typing import Callable, Optional

import numpy as np
import torch

from torch_geometric.data import Data, InMemoryDataset, extract_zip


class DGraphFin(InMemoryDataset):
    r"""The DGraphFin networks from the
        `"DGraph: A Large-Scale Financial Dataset for Graph Anomaly Detection"
        <https://arxiv.org/abs/2207.03579>`_ paper.
        It is a directed, unweighted dynamic graph consisting of millions of
        nodes and edges, representing a realistic user-to-user social network
        in financial industry.
        Node represents a Finvolution user, and an edge from one
        user to another means that the user regards the other user
        as the emergency contact person. Each edge is associated with a
        timestamp ranging from 1 to 821 and a type of emergency contact
        ranging from 0 to 11.


        Args:
            root (str): Root directory where the dataset should be saved.
            edge_window_size (int, optional): The window size for grouping edges. (default: :obj:'7' weekly)
            transform (callable, optional): A function/transform that takes in an
                :obj:`torch_geometric.data.Data` object and returns a transformed
                version. The data object will be transformed before every access.
                (default: :obj:`None`)
            pre_transform (callable, optional): A function/transform that takes in
                an :obj:`torch_geometric.data.Data` object and returns a
                transformed version. The data object will be transformed before
                being saved to disk. (default: :obj:`None`)
            force_reload (bool, optional): Whether to re-process the dataset.
                (default: :obj:`False`)

        **STATS:**

        .. list-table::
            :widths: 10 10 10 10
            :header-rows: 1

            * - #nodes
              - #edges
              - #features
              - #classes
            * - 3,700,550
              - 4,300,999
              - 17
              - 2
        """

    url = "https://dgraph.xinye.com"

    def __init__(self,
                 root: str,
                 edge_window_size: int = 7,
                 num_windows: int = 3,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 force_reload: bool = False
                 ) -> None:
        self.edge_window_size = edge_window_size
        self.num_windows = num_windows
        super().__init__(root, transform, pre_transform,
                         force_reload=force_reload)
        self.load(self.processed_paths[0])

    def download(self) -> None:
        raise RuntimeError(
            f"Dataset not found. Please download '{self.raw_file_names}' from "
            f"'{self.url}' and move it to '{self.raw_dir}'")

    @property
    def raw_file_names(self) -> str:
        return 'DGraphFin.zip'

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    @property
    def num_nodes(self) -> int:
        return 3_700_550

    @property
    def num_node_features(self) -> int:
        return 17

    @property
    def num_classes(self) -> int:
        return 2

    def process(self) -> None:
        extract_zip(self.raw_paths[0], self.raw_dir, log=False)
        path = osp.join(self.raw_dir, "dgraphfin.npz")

        with np.load(path) as loader:
            x = loader['x']
            y = loader['y']
            edge_index_np = loader['edge_index']
            edge_timestamp_np = loader['edge_timestamp']
            edge_type_np = loader['edge_type']
            max_timestamps = np.max(edge_timestamp_np)
            data_list = []
            for timestamp in range(1, max_timestamps, self.edge_window_size):
                edge_mask = (timestamp >= edge_timestamp_np) & (
                        edge_timestamp_np < timestamp + self.edge_window_size)
                filtered_edge_index = edge_index_np[edge_mask]
                filtered_edge_type = edge_type_np[edge_mask]
                data = Data()
                data.x = torch.tensor(x, dtype=torch.float)
                data.y = torch.tensor(y, dtype=torch.long)
                edge_index = torch.tensor(filtered_edge_index, dtype=torch.long)
                data.edge_index = edge_index.t()
                data.edge_attr = torch.tensor(filtered_edge_type, dtype=torch.long),
                available_node = torch.unique(edge_index)
                data.num_nodes = available_node.size(0)
                node_mask = torch.zeros(x.shape[0], dtype=torch.bool)
                filtered_indices = available_node[(data.y[available_node] == 0) | (data.y[available_node] == 1)]
                node_mask[filtered_indices] = True
                data.node_mask = node_mask
                data_list.append(data)
                if len(data_list) >= self.num_windows:
                    break
        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]
        self.save(data_list, self.processed_paths[0])
