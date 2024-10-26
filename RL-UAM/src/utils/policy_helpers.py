import torch
from typing import Dict, List, Union
import pandas as pd

def create_vertiport_edge_index(vertiport_ids: List, vertiport_distances: pd.DataFrame) -> Union[torch.Tensor, torch.Tensor]:
    """
    Creates the edge index for the vertiport graph.
    :param vertiport_distances: (pd.DataFrame) The vertiport distances
    :return: (torch.Tensor) The edge index
    """
    vertiport_indices = {vertiport_id: idx for idx, vertiport_id in enumerate(vertiport_ids)}
    edges = []
    distances = []
    for _, row in vertiport_distances.iterrows():
        origin_idx = vertiport_indices[row['origin_vertiport_id']]
        dest_idx = vertiport_indices[row['destination_vertiport_id']]
        edges.append((origin_idx, dest_idx))
        distances.append(row['distance_miles'])

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(distances, dtype=torch.float).unsqueeze(-1)
    
    return edge_index, edge_attr


def create_aircraft_edge_index(n_aircraft: int) -> Union[torch.Tensor, torch.Tensor]:
    """
    Creates the edge index for the aircraft graph.
    :param n_aircraft: (int) The number of aircraft
    :return: (torch.Tensor) The edge index
    """
    edges = [(i, j) for i in range(n_aircraft) for j in range(n_aircraft) if i != j]
    distances = [1 for _ in edges]  # A fully connected graph with equal distances
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(distances, dtype=torch.float).unsqueeze(-1)
    return edge_index, edge_attr