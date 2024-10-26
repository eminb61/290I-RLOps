from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
from typing import Dict, List
import gymnasium as gym
from torch_geometric.data import Data, Batch


class CustomGATFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, 
                 observation_space: gym.spaces.Box,
                 n_vertiports: int, 
                 n_aircraft: int, 
                 n_vertiport_state_variables: int,
                 n_aircraft_state_variables: int,
                 n_environmental_state_variables: int,
                 n_additional_state_variables: int,
                 vertiport_edge_index: torch.Tensor, 
                 vertiport_edge_attr: torch.Tensor, 
                 aircraft_edge_index: torch.Tensor, 
                 aircraft_edge_attr: torch.Tensor,
                 vertiport_out_channels: int, 
                 aircraft_out_channels: int,
                 hidden_channels: int,
                 num_gat_layers: int,
                 heads: int):
                
        self.n_vertiports = n_vertiports
        self.n_aircraft = n_aircraft
        self.n_vertiport_state_variables = n_vertiport_state_variables
        self.n_aircraft_state_variables = n_aircraft_state_variables
        self.n_environmental_state_variables = max(0, n_environmental_state_variables)
        self.n_additional_state_variables = n_additional_state_variables

        self.vertiport_edge_index = vertiport_edge_index
        self.vertiport_edge_attr = vertiport_edge_attr
        self.aircraft_edge_index = aircraft_edge_index
        self.aircraft_edge_attr = aircraft_edge_attr

        # Calculate the total size of the observation space
        total_obs_size = (n_vertiports * n_vertiport_state_variables +
                          n_aircraft * n_aircraft_state_variables +
                          self.n_environmental_state_variables + 
                          self.n_additional_state_variables)

        # Ensure that the observation_space matches our calculated size
        assert observation_space.shape[0] == total_obs_size, f"Observation space size {observation_space.shape[0]} does not match expected size {total_obs_size}"

        
        # Additional states in the observation space excluding the graph data
        features_dim = (vertiport_out_channels * self.n_vertiports) +\
            (aircraft_out_channels * self.n_aircraft) +\
                  self.n_environmental_state_variables +\
                      self.n_additional_state_variables

        super(CustomGATFeatureExtractor, self).__init__(observation_space, features_dim)


        from src.rl_models.sb3.sb3_contrib_local.encoders.gat_encoder import GATv2Encoder
        self.vertiport_encoder = GATv2Encoder(in_channels=self.n_vertiport_state_variables, 
                                              hidden_channels=hidden_channels,
                                              out_channels=vertiport_out_channels,
                                              num_layers=num_gat_layers,
                                              heads=heads,
                                              edge_dim=self.vertiport_edge_attr.shape[1])
        
        self.aircraft_encoder = GATv2Encoder(in_channels=self.n_aircraft_state_variables, 
                                             hidden_channels=hidden_channels,
                                             out_channels=aircraft_out_channels,
                                             num_layers=num_gat_layers,
                                             heads=heads,
                                             edge_dim=self.aircraft_edge_attr.shape[1])

    def forward(self, observations: torch.Tensor) -> torch.Tensor:

        if observations.dim() == 1:
            observations = observations.unsqueeze(0)            

        # Extract graph data for vertiports and aircraft from the observations
        vertiport_obs, aircraft_obs, env_obs, additional_obs = self.reshape_observation(observations)

        # Process vertiport data
        vertiport_graphs = [Data(x=vp, edge_index=self.vertiport_edge_index, edge_attr=self.vertiport_edge_attr) 
                            for vp in vertiport_obs]
        batched_vertiport_graphs = Batch.from_data_list(vertiport_graphs)
        vertiport_embeddings = self.vertiport_encoder(batched_vertiport_graphs)

        # Process aircraft data
        aircraft_graphs = [Data(x=ac, edge_index=self.aircraft_edge_index, edge_attr=self.aircraft_edge_attr) 
                           for ac in aircraft_obs]
        batched_aircraft_graphs = Batch.from_data_list(aircraft_graphs)
        aircraft_embeddings = self.aircraft_encoder(batched_aircraft_graphs)

        # Reshape embeddings
        vertiport_embeddings = vertiport_embeddings.view(observations.shape[0], self.n_vertiports, -1)
        aircraft_embeddings = aircraft_embeddings.view(observations.shape[0], self.n_aircraft, -1)

        # Flatten embeddings
        vertiport_embeddings_flat = vertiport_embeddings.view(observations.shape[0], -1)
        aircraft_embeddings_flat = aircraft_embeddings.view(observations.shape[0], -1)


        if env_obs.numel() == 0:
            env_obs = torch.zeros((observations.shape[0], 0))  # Make env_data an empty tensor if no env variables

        if additional_obs.numel() == 0:
            additional_obs = torch.zeros((observations.shape[0], 0))  # Similar for additional_data

        combined_embeddings = torch.cat([vertiport_embeddings_flat, 
                                         aircraft_embeddings_flat, 
                                         env_obs, 
                                         additional_obs], dim=1)        
        
        return combined_embeddings

    def reshape_observation(self, observations: torch.Tensor):
        batch_size = observations.shape[0]
        total_vertiport_obs = self.n_vertiports * self.n_vertiport_state_variables
        total_aircraft_obs = self.n_aircraft * self.n_aircraft_state_variables

        vertiport_data = observations[:, :total_vertiport_obs].view(batch_size, self.n_vertiports, self.n_vertiport_state_variables)
        aircraft_data = observations[:, total_vertiport_obs:total_vertiport_obs + total_aircraft_obs].view(batch_size, self.n_aircraft, self.n_aircraft_state_variables)

        env_start_idx = total_vertiport_obs + total_aircraft_obs
        env_data = observations[:, env_start_idx:env_start_idx + self.n_environmental_state_variables]
        
        additional_start_idx = env_start_idx + self.n_environmental_state_variables
        additional_data = observations[:, additional_start_idx:]

        return vertiport_data, aircraft_data, env_data, additional_data