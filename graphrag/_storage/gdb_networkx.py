import html
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Union, cast, List
import networkx as nx
import numpy as np
import asyncio
import xml.etree.ElementTree
import time

from .._utils import logger
from ..base import (
    BaseGraphStorage,
    SingleCommunitySchema,
)
from ..prompt import GRAPH_FIELD_SEP


@dataclass
class NetworkXStorage(BaseGraphStorage):
    @staticmethod
    def load_nx_graph(file_name) -> nx.Graph:
        if os.path.exists(file_name):
            try:
                # Try to read and parse the GraphML file
                graph = nx.read_graphml(file_name)
                if graph is not None:
                    logger.info(f"Successfully loaded graph file: {file_name}")
                    # Convert JSON strings in node attributes back to lists
                    for node, data in graph.nodes(data=True):
                        for key, value in data.items():
                            if isinstance(value, str):
                                try:
                                    # Try to parse JSON, if successful and it's a list, convert it
                                    parsed_value = json.loads(value)
                                    if isinstance(parsed_value, list):
                                        data[key] = parsed_value
                                except json.JSONDecodeError:
                                    # Not a valid JSON string, keep as is
                                    pass
                    
                    # Convert JSON strings in edge attributes back to lists
                    for u, v, data in graph.edges(data=True):
                        for key, value in data.items():
                            if isinstance(value, str):
                                try:
                                    parsed_value = json.loads(value)
                                    if isinstance(parsed_value, list):
                                        data[key] = parsed_value
                                except json.JSONDecodeError:
                                    pass
                return graph
            except (xml.etree.ElementTree.ParseError, ValueError, IOError) as e:
                # Catch XML parsing errors, value errors, and IO errors
                logger.error(f"Failed to load graph file: {file_name}, error: {str(e)}")
                logger.info("Creating a new empty graph")
                # File exists but may be corrupted, backing up and creating a new file
                if os.path.getsize(file_name) > 0:  # If the file is not empty, create a backup
                    backup_file = f"{file_name}.bak.{int(time.time())}"
                    try:
                        import shutil
                        shutil.copy2(file_name, backup_file)
                        logger.info(f"Backup of corrupted file created: {backup_file}")
                    except Exception as backup_error:
                        logger.warning(f"Failed to create backup file: {str(backup_error)}")
                return nx.Graph()
        
        # Graph file does not exist, creating a new graph
        logger.info(f"Graph file does not exist: {file_name}, creating a new graph")
        return nx.Graph()

    @staticmethod
    def write_nx_graph(graph: nx.Graph, file_name):
        try:
            logger.info(
                f"Writing graph with {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges"
            )
            # Create a copy of the graph to avoid modifying the original
            graph_copy = graph.copy()
            
            # Convert lists in node attributes to JSON strings
            for node, data in graph_copy.nodes(data=True):
                for key, value in data.items():
                    if isinstance(value, list):
                        data[key] = json.dumps(value)
            
            # Convert lists in edge attributes to JSON strings
            for u, v, data in graph_copy.edges(data=True):
                for key, value in data.items():
                    if isinstance(value, list):
                        data[key] = json.dumps(value)
            
            # Ensure the target directory exists
            os.makedirs(os.path.dirname(os.path.abspath(file_name)), exist_ok=True)
            
            # Use a temporary file to ensure successful write before replacing the original file
            temp_file = f"{file_name}.temp"
            nx.write_graphml(graph_copy, temp_file)
            
            # If write is successful, replace the original file
            import shutil
            shutil.move(temp_file, file_name)
            
            logger.info(f"Successfully wrote graph file: {file_name}")
        except Exception as e:
            logger.error(f"Failed to write graph file: {file_name}, error: {str(e)}")
            # If the temporary file exists but the write failed, try to delete it
            temp_file = f"{file_name}.temp"
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except:
                    pass
            # Do not throw an exception, allow the program to continue

    @staticmethod
    def stable_largest_connected_component(graph: nx.Graph) -> nx.Graph:
        """Refer to https://github.com/microsoft/graphrag/index/graph/utils/stable_lcc.py
        Return the largest connected component of the graph, with nodes and edges sorted in a stable way.
        """
        from graspologic.utils import largest_connected_component

        graph = graph.copy()
        graph = cast(nx.Graph, largest_connected_component(graph))
        node_mapping = {node: html.unescape(node.upper().strip()) for node in graph.nodes()}  # type: ignore
        graph = nx.relabel_nodes(graph, node_mapping)
        return NetworkXStorage._stabilize_graph(graph)

    @staticmethod
    def _stabilize_graph(graph: nx.Graph) -> nx.Graph:
        """Refer to https://github.com/microsoft/graphrag/index/graph/utils/stable_lcc.py
        Ensure an undirected graph with the same relationships will always be read the same way.
        """
        fixed_graph = nx.DiGraph() if graph.is_directed() else nx.Graph()

        sorted_nodes = graph.nodes(data=True)
        sorted_nodes = sorted(sorted_nodes, key=lambda x: x[0])

        fixed_graph.add_nodes_from(sorted_nodes)
        edges = list(graph.edges(data=True))

        if not graph.is_directed():

            def _sort_source_target(edge):
                source, target, edge_data = edge
                if source > target:
                    temp = source
                    source = target
                    target = temp
                return source, target, edge_data

            edges = [_sort_source_target(edge) for edge in edges]

        def _get_edge_key(source: Any, target: Any) -> str:
            return f"{source} -> {target}"

        edges = sorted(edges, key=lambda x: _get_edge_key(x[0], x[1]))

        fixed_graph.add_edges_from(edges)
        return fixed_graph

    def __post_init__(self):
        self._graphml_xml_file = os.path.join(
            self.global_config["working_dir"], f"graph_{self.namespace}.graphml"
        )
        preloaded_graph = NetworkXStorage.load_nx_graph(self._graphml_xml_file)
        if preloaded_graph is not None:
            logger.info(
                f"Loaded graph from {self._graphml_xml_file} with {preloaded_graph.number_of_nodes()} nodes, {preloaded_graph.number_of_edges()} edges"
            )
        self._graph = preloaded_graph or nx.Graph()
        self._clustering_algorithms = {
            "leiden": self._leiden_clustering,
        }
        self._node_embed_algorithms = {
            "node2vec": self._node2vec_embed,
        }

    async def index_done_callback(self):
        NetworkXStorage.write_nx_graph(self._graph, self._graphml_xml_file)

    async def has_node(self, node_id: str) -> bool:
        return self._graph.has_node(node_id)

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        return self._graph.has_edge(source_node_id, target_node_id)

    async def get_node(self, node_id: str) -> Union[dict, None]:
        return self._graph.nodes.get(node_id)
    
    async def get_all_nodes(self) -> dict[str, dict]:
        """
        Get all nodes and their data from the graph
        
        Returns:
            A dictionary containing all node IDs and their data
        """
        return dict(self._graph.nodes(data=True))
    
    async def get_nodes_batch(self, node_ids: list[str]) -> dict[str, Union[dict, None]]:
        return await asyncio.gather(*[self.get_node(node_id) for node_id in node_ids])

    async def node_degree(self, node_id: str) -> int:
        # [numberchiffre]: node_id not part of graph returns `DegreeView({})` instead of 0
        return self._graph.degree(node_id) if self._graph.has_node(node_id) else 0

    async def node_degrees_batch(self, node_ids: List[str]) -> List[str]:
        return await asyncio.gather(*[self.node_degree(node_id) for node_id in node_ids])

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        return (self._graph.degree(src_id) if self._graph.has_node(src_id) else 0) + (
            self._graph.degree(tgt_id) if self._graph.has_node(tgt_id) else 0
        )

    async def edge_degrees_batch(self, edge_pairs: list[tuple[str, str]]) -> list[int]:
        return await asyncio.gather(*[self.edge_degree(src_id, tgt_id) for src_id, tgt_id in edge_pairs])

    async def get_edge(
        self, source_node_id: str, target_node_id: str
    ) -> Union[dict, None]:
        return self._graph.edges.get((source_node_id, target_node_id))

    async def get_edges_batch(
        self, edge_pairs: list[tuple[str, str]]
    ) -> list[Union[dict, None]]:
        return await asyncio.gather(*[self.get_edge(source_node_id, target_node_id) for source_node_id, target_node_id in edge_pairs])

    async def get_node_edges(self, source_node_id: str):
        if self._graph.has_node(source_node_id):
            return list(self._graph.edges(source_node_id))
        return None

    async def get_nodes_edges_batch(
        self, node_ids: list[str]
    ) -> list[list[tuple[str, str]]]:
        return await asyncio.gather(*[self.get_node_edges(node_id) for node_id
        in node_ids])

    async def upsert_node(self, node_id: str, node_data: dict[str, str]):
        self._graph.add_node(node_id, **node_data)

    async def upsert_nodes_batch(self, nodes_data: list[tuple[str, dict[str, str]]]):
        await asyncio.gather(*[self.upsert_node(node_id, node_data) for node_id, node_data in nodes_data])

    async def upsert_edge(
        self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]
    ):
        self._graph.add_edge(source_node_id, target_node_id, **edge_data)

    async def upsert_edges_batch(
        self, edges_data: list[tuple[str, str, dict[str, str]]]
    ):
        await asyncio.gather(*[self.upsert_edge(source_node_id, target_node_id, edge_data) 
                for source_node_id, target_node_id, edge_data in edges_data])
        
    async def clustering(self, algorithm: str):
        if algorithm not in self._clustering_algorithms:
            raise ValueError(f"Clustering algorithm {algorithm} not supported")
        await self._clustering_algorithms[algorithm]()

    async def community_schema(self) -> dict[str, SingleCommunitySchema]:
        results = defaultdict(
            lambda: dict(
                level=None,
                title=None,
                edges=set(),
                nodes=set(),
                chunk_ids=set(),
                occurrence=0.0,
                sub_communities=[],
            )
        )
        max_num_ids = 0
        levels = defaultdict(set)
        for node_id, node_data in self._graph.nodes(data=True):
            if "clusters" not in node_data:
                continue
            clusters = json.loads(node_data["clusters"])
            this_node_edges = self._graph.edges(node_id)

            for cluster in clusters:
                level = cluster["level"]
                cluster_key = str(cluster["cluster"])
                levels[level].add(cluster_key)
                results[cluster_key]["level"] = level
                results[cluster_key]["title"] = f"Cluster {cluster_key}"
                results[cluster_key]["nodes"].add(node_id)
                results[cluster_key]["edges"].update(
                    [tuple(sorted(e)) for e in this_node_edges]
                )
                results[cluster_key]["chunk_ids"].update(
                    node_data["source_id"].split(GRAPH_FIELD_SEP)
                )
                max_num_ids = max(max_num_ids, len(results[cluster_key]["chunk_ids"]))

        ordered_levels = sorted(levels.keys())
        for i, curr_level in enumerate(ordered_levels[:-1]):
            next_level = ordered_levels[i + 1]
            this_level_comms = levels[curr_level]
            next_level_comms = levels[next_level]
            # compute the sub-communities by nodes intersection
            for comm in this_level_comms:
                results[comm]["sub_communities"] = [
                    c
                    for c in next_level_comms
                    if results[c]["nodes"].issubset(results[comm]["nodes"])
                ]

        for k, v in results.items():
            v["edges"] = list(v["edges"])
            v["edges"] = [list(e) for e in v["edges"]]
            v["nodes"] = list(v["nodes"])
            v["chunk_ids"] = list(v["chunk_ids"])
            v["occurrence"] = len(v["chunk_ids"]) / max_num_ids
        return dict(results)

    def _cluster_data_to_subgraphs(self, cluster_data: dict[str, list[dict[str, str]]]):
        for node_id, clusters in cluster_data.items():
            self._graph.nodes[node_id]["clusters"] = json.dumps(clusters)

    async def _leiden_clustering(self):
        from graspologic.partition import hierarchical_leiden

        graph = NetworkXStorage.stable_largest_connected_component(self._graph)
        community_mapping = hierarchical_leiden(
            graph,
            max_cluster_size=self.global_config["max_graph_cluster_size"],
            random_seed=self.global_config["graph_cluster_seed"],
        )

        node_communities: dict[str, list[dict[str, str]]] = defaultdict(list)
        __levels = defaultdict(set)
        for partition in community_mapping:
            level_key = partition.level
            cluster_id = partition.cluster
            node_communities[partition.node].append(
                {"level": level_key, "cluster": cluster_id}
            )
            __levels[level_key].add(cluster_id)
        node_communities = dict(node_communities)
        __levels = {k: len(v) for k, v in __levels.items()}
        logger.info(f"Each level has communities: {dict(__levels)}")
        self._cluster_data_to_subgraphs(node_communities)

    async def embed_nodes(self, algorithm: str) -> tuple[np.ndarray, list[str]]:
        if algorithm not in self._node_embed_algorithms:
            raise ValueError(f"Node embedding algorithm {algorithm} not supported")
        return await self._node_embed_algorithms[algorithm]()

    def get_graph(self):
        """
        Return the internal NetworkX graph object
        
        Returns:
            NetworkX graph object
        """
        return self._graph

    async def _node2vec_embed(self):
        from graspologic import embed

        embeddings, nodes = embed.node2vec_embed(
            self._graph,
            **self.global_config["node2vec_params"],
        )

        nodes_ids = [self._graph.nodes[node_id]["id"] for node_id in nodes]
        return embeddings, nodes_ids
