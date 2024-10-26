import pandas as pd
from typing import List, Tuple

def create_fully_connected_vertiport_network(fato_coordinates: List[Tuple[float, float]], 
                                             vertiport_ids: List[str], 
                                             save_path: str,
                                             network_simulation: bool) -> pd.DataFrame:
    """
    Creates a fully connected vertiport network with given coordinates.
    :param fato_coordinates: list of vertiport coordinates
    ex: 
    fato_coordinates = [
    (34.052110, -118.252680),
    (33.767700, -118.199550),
    (33.94305010754884, -118.40678502696238)
    ]
    :param vertiport_ids: list of vertiport locations
    ex:
    vertiport_ids = ['LADT', 'LongBeach', 'LAX']
    :param save_path: path to save the network
    :param network_simulation: if True, it creates a fully connected network. If False, it creates a unidirectional network.

    :return: pd.DataFrame of vertiport network

        o_vert_lat	o_vert_lon	d_vert_lat	d_vert_lon	origin_vertiport_id	destination_vertiport_id
    0	34.05211	-118.252680	33.76770	-118.199550	LADT	LongBeach
    1	34.05211	-118.252680	33.94305	-118.406785	LADT	LAX
    2	33.76770	-118.199550	34.05211	-118.252680	LongBeach	LADT
    3	33.76770	-118.199550	33.94305	-118.406785	LongBeach	LAX
    4	33.94305	-118.406785	34.05211	-118.252680	LAX	LADT
    5	33.94305	-118.406785	33.76770	-118.199550	LAX	LongBeach
    """
    data = []

    if network_simulation:
        for i in range(len(fato_coordinates)):
            for j in range(len(fato_coordinates)):
                if i != j:
                    data.append([
                        fato_coordinates[i][0], fato_coordinates[i][1], 
                        fato_coordinates[j][0], fato_coordinates[j][1], 
                        vertiport_ids[i], vertiport_ids[j]
                    ])
    else:
        for i in range(1, len(fato_coordinates)):
            data.append([
                fato_coordinates[0][0], fato_coordinates[0][1], 
                fato_coordinates[i][0], fato_coordinates[i][1], 
                vertiport_ids[0], vertiport_ids[i]
            ])

    vertiport_network = pd.DataFrame(data, columns=[
        'o_vert_lat', 'o_vert_lon', 'd_vert_lat', 'd_vert_lon', 
        'origin_vertiport_id', 'destination_vertiport_id'
    ])
    
    if network_simulation:
        vertiport_network.to_csv(save_path, index=False)
    else:
        vertiport_network.to_csv(f'{save_path}/{len(fato_coordinates)}_destination_single_vertiport.csv', index=False)
    
    return vertiport_network
