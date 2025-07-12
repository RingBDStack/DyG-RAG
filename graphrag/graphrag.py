import asyncio
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from functools import partial
from typing import Callable, Dict, List, Optional, Type, Union, cast, Tuple
import torch
import time
import tiktoken
import glob
import json
import re
import networkx as nx
import copy
import random
import math
from collections import Counter, defaultdict
from pathlib import Path
from datetime import datetime
from nano_vectordb import NanoVectorDB
import numpy as np
from copy import deepcopy

from .prompt import PROMPTS, GRAPH_FIELD_SEP


from ._llm import (
    amazon_bedrock_embedding,
    create_amazon_bedrock_complete_function,
    gpt_4o_complete,
    gpt_4o_mini_complete,
    openai_embedding,
    azure_gpt_4o_complete,
    azure_openai_embedding,
    azure_gpt_4o_mini_complete,
)
from ._op import (
    chunking_by_token_size,
    extract_events,
    get_chunks,
    normalize_timestamp,
)
from ._storage import (
    JsonKVStorage,
    NetworkXStorage,
    NanoVectorDBStorage,
    TimestampEnhancedVectorStorage,
)
from ._utils import (
    EmbeddingFunc,
    compute_mdhash_id,
    limit_async_func_call,
    convert_response_to_json,
    always_get_an_event_loop,
    logger,
    compute_args_hash,
    truncate_list_by_token_size,
)
from .base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    StorageNameSpace,
    QueryParam,
)

# Add cross-encoder imports
try:
    from sentence_transformers import CrossEncoder
    CROSSENCODER_AVAILABLE = True
except ImportError:
    CROSSENCODER_AVAILABLE = False
    logger.warning("sentence-transformers not available. Cross-encoder reranking will be disabled.")


@dataclass
class GraphRAG:
    working_dir: str = field(
        default_factory=lambda: f"./dyg_rag_cache_{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}"
    )

    # text chunking
    chunk_func: Callable[
        [
            list[list[int]],
            List[str],
            tiktoken.Encoding,
            Optional[int],
            Optional[int],
        ],
        List[Dict[str, Union[str, int]]],
    ] = chunking_by_token_size
    chunk_token_size: int = 1200
    chunk_overlap_token_size: int = 64
    tiktoken_model_name: str = "cl100k_base"

    # dynamic event extraction
    event_extract_max_gleaning: int = 1
    if_wri_ents: bool = False

    # Dynamic Graph (DyG) parameters
    ent_factor: float = 0.3  # Shared entity weight cal
    ent_ratio: float = 0.75  # Entity weight proportion in composite score
    time_ratio: float = 0.25  # Time weight proportion in composite score
    decay_rate: float = 0.02  # Time decay factor
    time_weight: float = 0.1  # time retrieval weight
    time_factor: float = 1.0  # Maximum weight for time-based relationship scoring
    max_links: int = 6  # Max relationships to create for each event.

    # --- Dynamic Query: Graph Traversal ---
    # Settings for how we explore the event graph during a query.
    enable_graph_traversal: bool = True  # To enable or disable the graph walk.
    walk_depth: int = 2  # How many steps to take from a seed event.
    walk_nodes: int = 5  # Max nodes to collect from the walk around a single seed.
    walk_n: int = 3  # How many random walks to start from each seed.

    # --- Dynamic Query: Reranking ---
    # TODO: Unify all these reranking configs. Maybe a single dictionary or a dedicated config object?
    # BM25 Reranking (a classic keyword-based method)
    enable_bm25_reranking: bool = False  # Whether to enable BM25 reranking
    bm25_k1: float = 1.5  # BM25 k1 parameter (term frequency saturation)
    bm25_b: float = 0.75  # BM25 b parameter (length normalization)
    bm25_weight: float = 0.7  # Weight of BM25 score in composite ranking
    entity_match_weight: float = 0.3  # Weight of entity matching in composite ranking

    # Cross-encoder Reranking (a more powerful neural method)
    enable_ce_rerank: bool = True  # Whether to enable cross-encoder reranking (alternative to BM25)
    ce_model: str = "cross-encoder/ms-marco-TinyBERT-L-2-v2"  # Cross-encoder model name
    ce_max_len: int = 512  # Maximum sequence length for cross-encoder
    ce_batch_size: int = 16  # Batch size for cross-encoder inference
    ce_device: str = "auto"  # Device for cross-encoder ('cpu', 'cuda', 'auto')
    ce_weight: float = 0.8  # Weight of cross-encoder score in composite ranking
    ce_ent_weight: float = 0.2  # Weight of entity matching in composite ranking (when using cross-encoder)

    # Cross-encoder optimization parameters
    ce_cache_size: int = 1000  # Maximum cache size for query-document pairs
    ce_truncate_len: int = 400  # Truncate text length before max_length tokenization
    ce_dynamic_batch: bool = True  # Enable dynamic batch sizing based on GPU memory
    ce_min_batch: int = 8  # Minimum batch size
    ce_max_batch: int = 64  # Maximum batch size
    ce_early_stop: float = 0.9  # A threshold to stop reranking early if we find a great match.

    # --- Model Configuration ---
    # A central place for all our AI models.
    model_path: str = "./models"

    # NER model parameters
    ner_model_name: str = "dslim_bert_base_ner"  # NER model name (will be combined with model_path)
    ner_device: str = "cuda:0"  # Device for NER model
    ner_batch_size: int = 32  # Batch size for NER processing

    # Entity matching parameters
    enable_fuzzy_entity_matching: bool = True  # Whether to enable fuzzy entity matching
    fuzzy_match_threshold: float = 0.8  # Fuzzy matching similarity threshold (0-1)


    # text embedding
    embedding_func: EmbeddingFunc = field(default_factory=lambda: openai_embedding)
    embedding_batch_num: int = 32
    embedding_func_max_async: int = 32
    query_better_than_threshold: float = 0.2

    # LLM
    using_azure_openai: bool = False
    using_amazon_bedrock: bool = False
    best_model_id: str = "us.anthropic.claude-3-sonnet-20240229-v1:0"
    cheap_model_id: str = "us.anthropic.claude-3-haiku-20240307-v1:0"
    best_model_func: callable = gpt_4o_complete
    best_model_max_token_size: int = 32768
    best_model_max_async: int = 16
    cheap_model_func: callable = gpt_4o_mini_complete
    cheap_model_max_token_size: int = 32768
    cheap_model_max_async: int = 16


    # storage
    key_string_value_json_storage_cls: Type[BaseKVStorage] = JsonKVStorage
    vector_db_storage_cls: Type[BaseVectorStorage] = NanoVectorDBStorage
    vector_db_storage_cls_kwargs: dict = field(default_factory=dict)
    graph_storage_cls: Type[BaseGraphStorage] = NetworkXStorage
    enable_llm_cache: bool = True
    enable_timestamp_encoding: bool = True  # Whether to use timestamp encoding in vector embeddings
    timestamp_dim: int = 16  # Dimension for timestamp encoding, must be a multiple of 4

    # extension
    always_create_working_dir: bool = True
    addon_params: dict = field(default_factory=dict)
    convert_response_to_json_func: callable = convert_response_to_json

    # Dynamic query timeline events parameter  
    if_timeline_events: bool = True  # Whether to include timeline events in dynamic query (default: True, set to False for ablation study)

    random_seed: int = 42  # Default random seed

    def get_config_dict(self):
        """Return a configuration dictionary without large model objects."""
        config = {}

        for k, v in self.__dict__.items():
            config[k] = v

        # These are runtime objects, not config, so we exclude them.
        keys_to_exclude = [
            "event_dynamic_graph",
            "events_vdb"
            "full_docs",
            "text_chunks",
            "llm_response_cache"
        ]

        for key in keys_to_exclude:
            if key in config:
                config.pop(key)

        return config

    def __post_init__(self):
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        
        _print_config = ",\n  ".join([f"{k} = {v}" for k, v in asdict(self).items()])
        logger.debug(f"GraphRAG init with param:\n\n  {_print_config}\n")

        self._original_vector_db_storage_cls = self.vector_db_storage_cls

        if self.enable_timestamp_encoding:
            self.vector_db_storage_cls = TimestampEnhancedVectorStorage
            if "timestamp_dim" not in self.vector_db_storage_cls_kwargs:
                self.vector_db_storage_cls_kwargs["timestamp_dim"] = self.timestamp_dim
            logger.info(
                f"Enabled timestamp encoding with dimension {self.timestamp_dim}"
            )

        if self.using_azure_openai:
            if self.best_model_func == gpt_4o_complete:
                self.best_model_func = azure_gpt_4o_complete
            if self.cheap_model_func == gpt_4o_mini_complete:
                self.cheap_model_func = azure_gpt_4o_mini_complete
            if self.embedding_func == openai_embedding:
                self.embedding_func = azure_openai_embedding
            logger.info(
                "Switched the default openai funcs to Azure OpenAI if you didn't set any of it"
            )

        if self.using_amazon_bedrock:
            self.best_model_func = create_amazon_bedrock_complete_function(
                self.best_model_id
            )
            self.cheap_model_func = create_amazon_bedrock_complete_function(
                self.cheap_model_id
            )
            self.embedding_func = amazon_bedrock_embedding
            logger.info("Switched the default openai funcs to Amazon Bedrock")

        if not os.path.exists(self.working_dir) and self.always_create_working_dir:
            logger.info(f"Creating working directory {self.working_dir}")
            os.makedirs(self.working_dir)
            
        config_dict = self.get_config_dict()

        self.full_docs = self.key_string_value_json_storage_cls(
            namespace="full_docs", global_config=config_dict
        )

        self.text_chunks = self.key_string_value_json_storage_cls(
            namespace="text_chunks", global_config=config_dict
        )

        self.llm_response_cache = (
            self.key_string_value_json_storage_cls(
                namespace="llm_response_cache", global_config=config_dict
            )
            if self.enable_llm_cache
            else None
        )

        self.event_dynamic_graph = self.graph_storage_cls(
            namespace="event_dynamic_graph", global_config=config_dict
        )
        
        self.embedding_func = limit_async_func_call(self.embedding_func_max_async)(
            self.embedding_func
        )
        
        self.events_vdb = (
            self.vector_db_storage_cls(
                namespace="events",
                global_config=config_dict,
                embedding_func=self.embedding_func,
                meta_fields={"event_id", "timestamp", "sentence"},
            )
        )
        

        self.best_model_func = limit_async_func_call(self.best_model_max_async)(
            partial(self.best_model_func, hashing_kv=self.llm_response_cache)
        )
        self.cheap_model_func = limit_async_func_call(self.cheap_model_max_async)(
            partial(self.cheap_model_func, hashing_kv=self.llm_response_cache)
        )

        # --- Initialize Cross-Encoder ---
        # FIXME: This whole block is a bit chunky. Could be moved to a helper function,
        # like `_initialize_reranker` or something similar.
        self.cross_encoder = None
        if self.enable_ce_rerank:
            if not CROSSENCODER_AVAILABLE:
                logger.warning(
                    "Cross-encoder reranking is enabled but sentence-transformers is not available. Falling back to BM25."
                )
                self.enable_ce_rerank = False
                self.enable_bm25_reranking = True
            else:
                try:
                    model_name = self.ce_model

                    # Prefer a local model if it exists, otherwise pull from HuggingFace.
                    local_model_path = (
                        Path(self.model_path) / model_name.replace("/", "_")
                    )
                    model_path = model_name

                    if local_model_path.exists():
                        config_file = local_model_path / "config.json"
                        model_files = list(
                            local_model_path.glob("*.bin")
                        ) + list(local_model_path.glob("*.safetensors")) + list(local_model_path.glob("pytorch_model.bin"))

                        if config_file.exists() and (
                            model_files
                            or any(local_model_path.glob("model.safetensors"))
                        ):
                            model_path = str(local_model_path)
                            logger.info(
                                f"Loading cross-encoder from validated local path: {model_path}"
                            )
                        else:
                            logger.warning(
                                f"Local model path exists but missing required files (config.json or model files). Using original model name: {model_name}"
                            )
                    else:
                        logger.info(
                            f"Local model path not found. Using original model name: {model_name}"
                        )

                    # Automatically select the best device (CUDA if available).
                    device = self.ce_device
                    if device == "auto":
                        try:
                            import torch

                            device = "cuda" if torch.cuda.is_available() else "cpu"
                        except ImportError:
                            device = "cpu"

                    self.cross_encoder = CrossEncoder(
                        model_name=model_path,
                        max_length=self.ce_max_len,
                        device=device,
                    )
                    logger.info(
                        f"Cross-encoder initialized successfully on device: {device}"
                    )

                    # Initialize our little optimization cache for reranking.
                    self.cross_encoder_cache = {}
                    logger.info(
                        f"Cross-encoder optimization enabled with cache size: {self.ce_cache_size}"
                    )

                    # It doesn't make much sense to use both rerankers, so disable BM25.
                    if self.enable_bm25_reranking:
                        logger.info(
                            "Cross-encoder enabled. BM25 reranking will be disabled unless explicitly kept enabled."
                        )
                        self.enable_bm25_reranking = False

                except Exception as e:
                    logger.error(
                        f"Failed to initialize cross-encoder model {model_name}: {e}"
                    )
                    logger.info("Falling back to BM25 reranking")
                    self.enable_ce_rerank = False
                    self.enable_bm25_reranking = True
                    self.cross_encoder = None

        # Log the dynamic event graph statistics after everything is loaded
        self._log_event_graph_stats()
        
        # Also log general storage statistics for a complete overview
        self._log_storage_overview()

    def log_data_statistics(self):
        """
        Manually log all data statistics including event graph, vector database, and storage.
        This can be called by users to get a current overview of the loaded data.
        """
        logger.info("=== DyG-RAG Data Statistics ===")
        self._log_event_graph_stats()
        self._log_storage_overview()
        logger.info("=== End Data Statistics ===")

    def _log_event_graph_stats(self):
        """
        Log statistics about the loaded event dynamic graph and vector database.
        This gives us a quick overview of what we're working with.
        """
        try:
            # Log event dynamic graph statistics
            if hasattr(self.event_dynamic_graph, '_graph') and self.event_dynamic_graph._graph:
                graph = self.event_dynamic_graph._graph
                num_nodes = graph.number_of_nodes()
                num_edges = graph.number_of_edges()
                
                if num_nodes > 0:
                    # Calculate some basic graph metrics for insight
                    avg_degree = (2 * num_edges) / num_nodes if num_nodes > 0 else 0
                    
                    logger.info(f"Event Dynamic Graph loaded successfully:")
                    logger.info(f"  • Nodes (events): {num_nodes:,}")
                    logger.info(f"  • Edges (relationships): {num_edges:,}")
                    logger.info(f"  • Average node degree: {avg_degree:.2f}")
                    
                    # If the graph is reasonably sized, show density
                    if num_nodes > 1:
                        max_edges = num_nodes * (num_nodes - 1) // 2  # For undirected graph
                        density = num_edges / max_edges if max_edges > 0 else 0
                        logger.info(f"  • Graph density: {density:.4f}")
                else:
                    logger.info("Event Dynamic Graph is empty (no events loaded yet)")
            else:
                logger.warning("Event Dynamic Graph is not properly initialized")
                
            # Log events vector database statistics
            self._log_events_vdb_stats()
                
        except Exception as e:
            logger.warning(f"Could not retrieve event graph statistics: {e}")
            
    def _log_events_vdb_stats(self):
        """
        Log statistics about the events vector database.
        """
        try:
            if hasattr(self.events_vdb, '_client') and self.events_vdb._client:
                # For NanoVectorDB storage
                client = self.events_vdb._client
                
                # Try to get the count of documents in different ways
                doc_count = 0
                
                # Method 1: Check if the client has a direct count method
                if hasattr(client, '__storage') and isinstance(client.__storage, dict):
                    storage = client.__storage
                    
                    # Check different possible storage structures
                    if 'matrix' in storage and storage['matrix'] is not None:
                        if hasattr(storage['matrix'], 'shape'):
                            doc_count = storage['matrix'].shape[0]
                        elif isinstance(storage['matrix'], list):
                            doc_count = len(storage['matrix'])
                    elif 'data' in storage and storage['data'] is not None:
                        if isinstance(storage['data'], list):
                            doc_count = len(storage['data'])
                
                # Method 2: Try alternative ways to get count
                if doc_count == 0:
                    # Check if there's a method to get current count
                    if hasattr(client, 'get_current_count'):
                        doc_count = client.get_current_count()
                    elif hasattr(client, 'count'):
                        doc_count = client.count()
                    elif hasattr(client, '__len__'):
                        doc_count = len(client)
                
                # Log the results
                if doc_count > 0:
                    logger.info(f"Events Vector Database loaded successfully:")
                    logger.info(f"  • Documents (events): {doc_count:,}")
                    
                    # Show embedding dimension if available
                    if hasattr(client, 'vec_dim'):
                        logger.info(f"  • Embedding dimension: {client.vec_dim}")
                    elif hasattr(self.events_vdb, 'enhanced_dim'):
                        # For TimestampEnhancedVectorStorage
                        logger.info(f"  • Enhanced embedding dimension: {self.events_vdb.enhanced_dim}")
                        if hasattr(self.events_vdb, 'original_dim') and hasattr(self.events_vdb, 'timestamp_dim'):
                            logger.info(f"    - Semantic: {self.events_vdb.original_dim}, Timestamp: {self.events_vdb.timestamp_dim}")
                else:
                    logger.info("Events Vector Database is empty (no events indexed yet)")
            else:
                logger.warning("Events Vector Database is not properly initialized")
        except Exception as e:
            logger.warning(f"Could not retrieve events vector database statistics: {e}")

    def _log_storage_overview(self):
        """
        Log statistics about the general storage (full_docs, text_chunks, llm_response_cache).
        """
        try:
            logger.info("--- Storage Overview ---")
            
            # Full docs storage statistics
            if hasattr(self.full_docs, '_data'):
                logger.info(f"Full Docs Storage: {len(self.full_docs._data):,} documents")
            else:
                logger.info("Full Docs Storage: not initialized or unavailable")
                
            # Text chunks storage statistics
            if hasattr(self.text_chunks, '_data'):
                logger.info(f"Text Chunks Storage: {len(self.text_chunks._data):,} chunks")
            else:
                logger.info("Text Chunks Storage: not initialized or unavailable")
                
            # LLM response cache statistics
            if self.llm_response_cache and hasattr(self.llm_response_cache, '_data'):
                logger.info(f"LLM Response Cache: {len(self.llm_response_cache._data):,} cached responses")
            else:
                logger.info("LLM Response Cache: disabled or unavailable")
                
            logger.info("--- End Storage Overview ---")
        except Exception as e:
            logger.warning(f"Could not retrieve storage overview: {e}")

    def insert(self, string_or_strings):
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.ainsert(string_or_strings))

    def query(self, query: str, param: QueryParam = QueryParam()):
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.aquery(query, param))

    async def aquery(self, query: str, param: QueryParam = QueryParam()):
        if param.mode == "dynamic":
            response = await self.dynamic_query(
                query,
                param
            )
        else:
            raise ValueError(f"Unknown mode {param.mode}")
        await self._query_done()
        return response

    async def dynamic_query(self, query: str, param: QueryParam):
        logger.info(f"Executing dynamic event query: {query}")
        
        time_constraints, entities = await self.parse_query_time_and_entities(
            query, param, self.llm_response_cache
        )
        logger.info(f"Extracted time constraints: {time_constraints}")
        logger.info(f"Extracted entities: {entities}")
        
        topk1 = param.topk1 or 2000  
        et_top_k = param.et_top_k or 20   
        found_results = []
        
        

        if hasattr(self.events_vdb, 'time_weighted_query'):
            # logger.info(f"Using time_weighted_query with topk1={topk1}")

            start_time = time_constraints.get("start_time")
            end_time = time_constraints.get("end_time")
            current_timestamp = datetime.now().strftime("%Y-%m-%d")

            if not start_time:
                start_time = "0"
                # logger.info("start_time is empty, setting to '0'")

            if not end_time:
                end_time = current_timestamp
                # logger.info(f"end_time is empty, setting to current time: {end_time}")

            all_time_weighted_results = []

            try:
                # logger.info(f"Querying with start_time: {start_time}, query: {query}")
                start_results = await self.events_vdb.time_weighted_query(
                    query=query, 
                    timestamp=start_time,
                    time_weight=getattr(self, 'time_weight', 1.0),
                    top_k=topk1 
                )
                if start_results:
                    all_time_weighted_results.extend(start_results)
                    logger.info(f"Found {len(start_results)} events using start_time query")
                else:
                    logger.info("No results from start_time query")
            except Exception as e:
                logger.warning(f"Error in time_weighted_query with start_time {start_time}: {e}")
            
            if end_time != start_time:
                try:
                    # logger.info(f"Querying with end_time: {end_time}, query: {query}")
                    end_results = await self.events_vdb.time_weighted_query(
                        query=query,
                        timestamp=end_time,
                        time_weight=getattr(self, 'time_weight', 1.0),
                        top_k=topk1 
                    )
                    if end_results:
                        all_time_weighted_results.extend(end_results)
                        logger.info(f"Found {len(end_results)} events using end_time query")
                    else:
                        logger.info("No results from end_time query")
                except Exception as e:
                    logger.warning(f"Error in time_weighted_query with end_time {end_time}: {e}")
            else:
                logger.warning("start_time and end_time are the same, skipping duplicate query")
            
            # Deduplicate results by event ID
            if all_time_weighted_results:
                seen_ids = set()
                deduplicated_results = []
                for result in all_time_weighted_results:
                    event_id = result.get('id')
                    if event_id and event_id not in seen_ids:
                        seen_ids.add(event_id)
                        deduplicated_results.append(result)
                    elif not event_id: 
                        deduplicated_results.append(result)
                
                found_results = deduplicated_results
                logger.info(f"After deduplication: {len(found_results)} unique events")
            else:
                logger.warning("No events found using time_weighted_query")
                try:
                    fallback_results = await self.events_vdb.query(query, top_k=topk1)
                    if fallback_results:
                        found_results = fallback_results
                        # logger.info(f"Fallback to regular query, found {len(found_results)} events")
                except Exception as e:
                    logger.warning(f"Error in fallback regular query: {e}")
                    
        else:
            # logger.info("events_vdb doesn't support time_weighted_query, using regular query")
            regular_results = await self.events_vdb.query(query, top_k=topk1)
            if regular_results:
                found_results = regular_results
                # logger.info(f"Found {len(found_results)} events using regular query")
        
        top_k_seed_events = []
        if found_results:
            logger.info(f"Starting cross-encoder filtering to select top_{et_top_k} seed events from {len(found_results)} candidates")
            
            if (self.enable_ce_rerank or self.enable_bm25_reranking):
                if self.enable_ce_rerank and self.cross_encoder:
                    logger.info("Using cross-encoder for seed selection")
                    reranked_candidates = await self.rerank_with_cross_encoder(
                        events=found_results,
                        query=query,
                        entities=entities,
                        time_constraints=time_constraints
                    )
                elif self.enable_bm25_reranking:
                    logger.info("Using BM25 for seed selection")  
                    reranked_candidates = await self.rerank_with_bm25(
                        events=found_results,
                        query=query,
                        entities=entities,
                        time_constraints=time_constraints
                    )
                else:
                    reranked_candidates = found_results
                
                top_k_seed_events = reranked_candidates[:et_top_k]
                logger.info(f"Selected top_{et_top_k} events as seeds for graph traversal")
            else:
                sorted_candidates = sorted(found_results, key=lambda x: x.get('distance', 1.0))
                top_k_seed_events = sorted_candidates[:et_top_k]
                logger.info(f"No reranking enabled, selected top_{et_top_k} events by distance")
        else:
            logger.info("No filtered candidates available for seed selection")

        logger.info(f"Starting graph traversal from {len(top_k_seed_events)} seed events")
        
        seed_event_ids = []
        for result in top_k_seed_events:
            event_id = result.get('id')
            if event_id:
                seed_event_ids.append(event_id)
        
        # logger.info(f"Found {len(seed_event_ids)} valid seed event IDs for graph traversal")
        
        graph_traversed_event_ids = []
        if (self.enable_graph_traversal and seed_event_ids and self.event_dynamic_graph):
            try:
                graph_traversed_event_ids = await self._random_walk_graph_traversal(
                    event_graph=self.event_dynamic_graph,
                    seed_event_ids=seed_event_ids,
                    max_depth=self.walk_depth,
                    max_nodes_per_seed=self.walk_nodes,
                    num_walks=self.walk_n
                )
                logger.info(f"Graph traversal found {len(graph_traversed_event_ids)} total events")
            except Exception as e:
                logger.info(f"Error during graph traversal: {e}")
                graph_traversed_event_ids = []
        elif not self.enable_graph_traversal:
            logger.info("Graph traversal is disabled in configuration")
        else:
            logger.info("Skipping graph traversal: no valid seeds or graph not available")
        
        final_results = []
        if graph_traversed_event_ids:
            # logger.info("Retrieving full event data for all traversed events...")
            
            if self.event_dynamic_graph:
                try:
                    events_data = await self.event_dynamic_graph.get_nodes_batch(graph_traversed_event_ids)
                    
                    for i, event_data in enumerate(events_data):
                        if event_data is not None:
                            event_id = graph_traversed_event_ids[i]
                            final_results.append(
                                {
                                    "id": event_id,
                                    "timestamp": event_data.get("timestamp", "unknown"),
                                    "sentence": event_data.get("sentence", ""),
                                    "context": event_data.get("context", ""),
                                    "entities_involved": event_data.get(
                                        "entities_involved", []
                                    ),
                                    "source_id": event_data.get(
                                        "source_id", ""
                                    ),  # This should now be available
                                    "distance": 0.0,  # No distance in graph retrieval
                                }
                            )

                    logger.info(
                        f"Retrieved {len(final_results)} complete events from event_dynamic_graph"
                    )

                except Exception as e:
                    logger.error(f"Error retrieving events from event_dynamic_graph: {e}")
                    final_results = []
            else:
                logger.warning("event_dynamic_graph not available")
                final_results = []
        else:
            logger.info("No events found through graph traversal")
        
        # logger.info(f"Using {len(final_results)} events from graph traversal for context building")
        
        sorted_final_results = sorted(final_results, key=lambda x: x.get('timestamp', '0'))
        
        question = query

        from .prompt import PROMPTS, GRAPH_FIELD_SEP
        from collections import Counter
        
        source_id_frequency = Counter()

        for i, event in enumerate(final_results):
            event_source_id = event.get('source_id', '')
            logger.debug(f"Event {i+1}: ID={event.get('id', 'N/A')}, source_id='{event_source_id}'")
            
            if event_source_id:
                event_source_ids = event_source_id.split(GRAPH_FIELD_SEP)
                for sid in event_source_ids:
                    sid = sid.strip()
                    if sid:
                        source_id_frequency[sid] += 1
        
        if source_id_frequency:
            sorted_source_ids = [sid for sid, freq in source_id_frequency.most_common()]
            limited_source_ids = sorted_source_ids[:param.top_k]
            
            logger.info(f"Found {len(source_id_frequency)} unique source_ids")
            # logger.info(f"Top {len(limited_source_ids)} source_ids by frequency: {dict(source_id_frequency.most_common(param.top_k))}")
        else:
            limited_source_ids = []
            logger.warning("No valid source_ids found in events")
        
        try:
            retrieved_chunks = await self.text_chunks.get_by_ids(limited_source_ids)
            logger.info(f"Retrieved {len(retrieved_chunks)} chunks from text_chunks")

            for i, chunk in enumerate(retrieved_chunks):
                if chunk is not None:
                    chunk_keys = list(chunk.keys()) if isinstance(chunk, dict) else "Not a dict"
                    has_content = 'content' in chunk if isinstance(chunk, dict) else False
                    content_length = len(chunk.get('content', '')) if isinstance(chunk, dict) and chunk.get('content') else 0
                    logger.debug(f"Chunk {i+1}: keys={chunk_keys}, has_content={has_content}, content_length={content_length}")
            
            valid_chunks = []
            for chunk in retrieved_chunks:
                if chunk is not None and isinstance(chunk, dict):
                    content = chunk.get('content', '').strip()
                    if content:
                        valid_chunks.append(content)
            
            logger.debug(f"Extracted {len(valid_chunks)} valid chunks with content")
            
            truncated_chunks = truncate_list_by_token_size(
                valid_chunks,
                key=lambda x: x,  # Since valid_chunks is list of strings, use identity function
                max_token_size=param.max_token_for_text_unit
            )
            logger.debug(f"After token truncation: {len(truncated_chunks)} chunks remain")
            
            events_section = ""
            if final_results and self.if_timeline_events:
                events_section = self.build_time_CoT(
                    events=final_results,
                    query=question,
                    time_constraints=time_constraints,
                    entities=entities
                )
            elif not self.if_timeline_events:
                logger.info("Timeline events disabled for ablation study - skipping events section construction")
            
            chunks_section = ""
            if truncated_chunks:
                for i, chunk_content in enumerate(truncated_chunks):
                    chunks_section += f"## Chunk {i+1}\n{chunk_content}\n\n"
            else:
                chunks_section = "No relevant chunks found."
            if self.if_timeline_events:
                context = PROMPTS["dynamic_QA"].format(
                    question=question,
                    events_data=events_section,
                    chunks_data=chunks_section
                )
                logger.debug("Using dynamic_QA prompt template with events and chunks")
            else:
                # Ablation study: exclude timeline events - use dynamic_QA_wo_timeline prompt
                context = PROMPTS["dynamic_QA_wo_timeline"].format(
                    question=question,
                    chunks_data=chunks_section
                )
                logger.debug("Using dynamic_QA_wo_timeline prompt template with chunks only (ablation study)")
        except Exception as e:
            logger.error(f"Error retrieving chunks: {e}")
            # Fall back to original method if chunk retrieval fails
            events_section = ""
            if final_results:
                events_section = self.build_time_CoT(
                    events=final_results,
                    query=question,
                    time_constraints=time_constraints,
                    entities=entities
                )
            
            context = PROMPTS["short_answer"].format(
                content_data=events_section, 
                response_type=param.response_type
            )
        
        response = await self.best_model_func(context)
        return response

    async def ainsert(self, string_or_strings):
        await self._insert_start()
        try:
            # ---------- data preprocessing
            if isinstance(string_or_strings, str):
                string_or_strings = [string_or_strings]
            # ---------- new docs
            new_docs = {
                compute_mdhash_id(c.strip(), prefix="doc-"): {"content": c.strip()}
                for c in string_or_strings
            }
            _add_doc_keys = await self.full_docs.filter_keys(list(new_docs.keys()))
            new_docs = {k: v for k, v in new_docs.items() if k in _add_doc_keys}
            if not len(new_docs):
                logger.warning(f"All docs are already in the storage")
                await self._insert_done()
                return
            logger.info(f"[New Docs] inserting {len(new_docs)} docs")

            # ---------- chunking
            inserting_chunks = get_chunks(
                new_docs=new_docs,
                chunk_func=self.chunk_func,
                overlap_token_size=self.chunk_overlap_token_size,
                max_token_size=self.chunk_token_size,
            )

            _add_chunk_keys = await self.text_chunks.filter_keys(
                list(inserting_chunks.keys())
            )
            inserting_chunks = {
                k: v for k, v in inserting_chunks.items() if k in _add_chunk_keys
            }
            if not len(inserting_chunks):
                logger.warning(f"All chunks are already in the storage")
                await self._insert_done()
                return
            logger.info(f"[New Chunks] inserting {len(inserting_chunks)} chunks")

            # ---------- event extraction
            logger.info("[Event Extraction]...")
            maybe_new_dyg, extraction_stats = await extract_events(
                inserting_chunks,
                dyg_inst=self.event_dynamic_graph,
                events_vdb = self.events_vdb,
                global_config={**self.get_config_dict(), "events_vdb": self.events_vdb},
                using_amazon_bedrock=self.using_amazon_bedrock,
            )
            self.extraction_stats = extraction_stats
            
            # update dynamic event graph
            if maybe_new_dyg is not None:
                self.event_dynamic_graph = maybe_new_dyg
                logger.info(f"Updated dynamic event graph")
            else:
                logger.warning("No new events found")
                await self._insert_done()
                return
            
            torch.cuda.empty_cache()
            time.sleep(2)
           
            # ---------- commit upsertings
            await self.full_docs.upsert(new_docs)
            await self.text_chunks.upsert(inserting_chunks)
        finally:
            await self._insert_done()

    async def _insert_start(self):
        tasks = []
        for storage_inst in [
            self.full_docs,
            self.text_chunks,
            self.events_vdb,
            self.event_dynamic_graph,
        ]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_start_callback())
        await asyncio.gather(*tasks)

    async def _insert_done(self):
        tasks = []
        for storage_inst in [
            self.full_docs,
            self.text_chunks,
            self.llm_response_cache,
            self.event_dynamic_graph,
            self.events_vdb,
        ]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks)

    async def _query_done(self):
        tasks = []
        for storage_inst in [self.llm_response_cache]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks)

    def build_time_CoT(self, events, query=None, time_constraints=None, entities=None):
        if not events:
            return ""

        static_events = []
        timestamped_events = []
        
        for event in events:
            timestamp = event.get('timestamp', 'static')
            if timestamp == 'static' or not timestamp or timestamp == 'unknown':
                static_events.append(event)
            else:
                timestamped_events.append(event)
        
        timestamped_events.sort(key=lambda x: x.get('timestamp', '0'))
        
        sorted_events = static_events + timestamped_events

        context_parts = []

        for i, event in enumerate(sorted_events):
            event_time = event.get('timestamp', 'static')
            sentence = event.get('sentence', '')
            context = event.get('context', '') # Get the context field
            
            if event_time == 'static' or not event_time or event_time == 'unknown':
                time_display = 'Static'
            else:
                time_display = event_time

            event_text = f"Event #{i+1} [{time_display}]: {sentence}"
            # if context: # Add context if it exists
            #     event_text += f"\n  Context: {context}"
            context_parts.append(event_text)

        return "\n".join(context_parts)

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            # Don't deepcopy things that can't or shouldn't be, like model functions.
            if k in ["cheap_model_func", "best_model_func", "cross_encoder"]:
                setattr(result, k, v)
            else:
                setattr(result, k, deepcopy(v, memo))
        return result

    async def _random_walk_graph_traversal(
        self,
        event_graph,
        seed_event_ids,
        max_depth=3,
        max_nodes_per_seed=10,
        num_walks=5,
    ):
        """
        Explore the event graph starting from a set of seed events.
        This helps discover related context that wasn't found in the initial vector search.
        """
        if not seed_event_ids:
            return []

        all_nodes = await event_graph.get_all_nodes()
        if not all_nodes:
            logger.info("Graph is empty")
            return []

        valid_seeds = [
            seed_id
            for seed_id in seed_event_ids
            if await event_graph.has_node(seed_id)
        ]
        if not valid_seeds:
            logger.info("No valid seed events found in the graph")
            return []

        logger.info(
            f"Starting random walk traversal from {len(valid_seeds)} seed events"
        )

        all_related_events = set()

        for seed_event_id in valid_seeds:
            seed_related_events = set()
            seed_related_events.add(seed_event_id)

            # For each seed, start multiple random walks.
            for walk_num in range(num_walks):
                current_node = seed_event_id
                walk_path = [current_node]

                for step in range(max_depth):
                    neighbors = []
                    weights = []
                    node_edges = await event_graph.get_node_edges(current_node)

                    if node_edges:
                        for edge in node_edges:
                            neighbor_node = (
                                edge[1] if edge[0] == current_node else edge[0]
                            )

                            edge_data = await event_graph.get_edge(edge[0], edge[1])
                            if (
                                edge_data
                                and edge_data.get("relation_type")
                                == "event_temporal_proximity"
                            ):
                                neighbors.append(neighbor_node)
                                weights.append(edge_data.get("weight", 1.0))

                    # Don't walk back to a node we just visited in this path.
                    available_indices = [
                        i for i, n in enumerate(neighbors) if n not in walk_path
                    ]
                    if not available_indices:
                        break
                    
                    available_weights = [weights[i] for i in available_indices]
                    available_neighbors = [neighbors[i] for i in available_indices]
                    
                    total_weight = sum(available_weights)
                    if total_weight > 0:
                        probabilities = [w / total_weight for w in available_weights]
                        current_node = random.choices(
                            available_neighbors, weights=probabilities, k=1
                        )[0]
                    else:
                        current_node = random.choice(available_neighbors)
                    
                    walk_path.append(current_node)
                    seed_related_events.add(current_node)
                    
                    if len(seed_related_events) >= max_nodes_per_seed:
                        break

                await asyncio.sleep(0)  # Yield control to the event loop.

                if len(seed_related_events) >= max_nodes_per_seed:
                    break

            all_related_events.update(seed_related_events)
            logger.info(
                f"Seed {seed_event_id}: collected {len(seed_related_events)} related events"
            )

        related_event_list = list(all_related_events)
        logger.info(
            f"Random walk traversal completed: {len(related_event_list)} total unique events found"
        )

        return related_event_list

    def tokenize_text(self, text: str) -> List[str]:
        if not text:
            return []
        
        # Convert to lowercase and split by whitespace and punctuation
        import string
        text = text.lower()
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        # Split and filter empty strings
        tokens = [token for token in text.split() if token.strip()]
        return tokens

    def build_bm25_corpus(self, events: List[Dict]) -> tuple[List[List[str]], Dict[str, int]]:
        corpus = []
        event_id_to_index = {}
        
        for idx, event in enumerate(events):
            text_content = ""
            
            if event.get('sentence'):
                text_content = event.get('sentence', '')
            elif event.get('content'):
                # Fallback to content if sentence is not available
                text_content = event.get('content', '')
            else:
                # Last resort: empty text
                text_content = ""
            
            # Tokenize the text
            tokens = self.tokenize_text(text_content.strip())
            corpus.append(tokens)
            
            # Map event ID to corpus index
            event_id = event.get('id')
            if event_id:
                event_id_to_index[event_id] = idx
        
        return corpus, event_id_to_index

    def calculate_bm25_scores(self, query: str, corpus: List[List[str]], 
                            k1: float = 1.5, b: float = 0.75) -> List[float]:
        if not corpus:
            return []
        
        query_tokens = self.tokenize_text(query)
        if not query_tokens:
            return [0.0] * len(corpus)
        
        N = len(corpus)
        doc_freqs = defaultdict(int)
        doc_lengths = []
        
        for doc in corpus:
            doc_lengths.append(len(doc))
            unique_terms = set(doc)
            for term in unique_terms:
                doc_freqs[term] += 1
        
        avgdl = sum(doc_lengths) / len(doc_lengths) if doc_lengths else 0
        
        scores = []
        for doc_idx, doc in enumerate(corpus):
            score = 0.0
            doc_length = doc_lengths[doc_idx]
            
            term_freqs = Counter(doc)
            
            for term in query_tokens:
                if term in term_freqs:
                    tf = term_freqs[term]
                    
                    df = doc_freqs[term]
                    
                    idf = math.log((N - df + 0.5) / (df + 0.5))
                    
                    numerator = idf * tf * (k1 + 1)
                    denominator = tf + k1 * (1 - b + b * (doc_length / avgdl))
                    
                    score += numerator / denominator
            
            scores.append(score)
        
        return scores

    def calculate_entity_match_score(self, event_content: str, entities: List[str]) -> float:
        if not entities or not event_content:
            return 0.0
        
        content_lower = event_content.lower()
        matched_entities = 0
        
        for entity in entities:
            if self._is_entity_matched(entity, content_lower):
                matched_entities += 1
        
        return matched_entities / len(entities)
    
    def _is_entity_matched(self, entity: str, content_lower: str) -> bool:
        entity_lower = entity.lower().strip()
        
        if entity_lower in content_lower:
            return True
        
        entity_words = [word.strip() for word in entity_lower.split() if len(word.strip()) > 2]
        
        if len(entity_words) > 1:
            for word in entity_words:
                pattern = r'\b' + re.escape(word) + r'\b'
                if re.search(pattern, content_lower):
                    return True
        
        if ' ' in entity_lower:
            last_name = entity_words[-1] if entity_words else ""
            if len(last_name) > 2:
                pattern = r'\b' + re.escape(last_name) + r'\b'
                if re.search(pattern, content_lower):
                    return True
        
        entity_initials = ''.join([word[0] for word in entity_words if word])
        if len(entity_initials) > 1:
            pattern = r'\b' + re.escape(entity_initials) + r'\b'
            if re.search(pattern, content_lower):
                return True
        
        if hasattr(self, 'enable_fuzzy_entity_matching') and self.enable_fuzzy_entity_matching:
            threshold = getattr(self, 'fuzzy_match_threshold', 0.8)
            return self._fuzzy_entity_match(entity_lower, content_lower, threshold)
        
        return False
    
    def _fuzzy_entity_match(self, entity: str, content: str, threshold: float = 0.8) -> bool:
        try:
            from difflib import SequenceMatcher
        except ImportError:
            return False
        
        content_words = content.split()
        
        for word in content_words:
            if len(word) > 2:
                similarity = SequenceMatcher(None, entity.lower(), word.lower()).ratio()
                if similarity >= threshold:
                    return True
        
        entity_words = entity.split()
        if len(entity_words) > 1:
            content_words = content.split()
            for i in range(len(content_words) - len(entity_words) + 1):
                content_phrase = ' '.join(content_words[i:i+len(entity_words)])
                similarity = SequenceMatcher(None, entity.lower(), content_phrase.lower()).ratio()
                if similarity >= threshold:
                    return True
        
        return False

    async def rerank_with_bm25(self, events: List[Dict], query: str, 
                             entities: List[str], time_constraints: Dict) -> List[Dict]:
        if not events or not self.enable_bm25_reranking:
            return events
        
        logger.info(f"Starting BM25 reranking for {len(events)} events")
        
        corpus, event_id_to_index = self.build_bm25_corpus(events)
        
        bm25_scores = self.calculate_bm25_scores(
            query=query, 
            corpus=corpus, 
            k1=self.bm25_k1, 
            b=self.bm25_b
        )
        
        if bm25_scores and max(bm25_scores) > 0:
            max_score = max(bm25_scores)
            bm25_scores = [score / max_score for score in bm25_scores]
        
        scored_events = []
        for idx, event in enumerate(events):
            bm25_score = bm25_scores[idx] if idx < len(bm25_scores) else 0.0
            
            entity_score = self.calculate_entity_match_score(
                event.get('sentence', '') or event.get('content', ''), entities
            )
            
            composite_score = (
                bm25_score * self.bm25_weight +
                entity_score * self.entity_match_weight
            )
            
            event['_bm25_score'] = bm25_score
            event['_entity_score'] = entity_score
            event['_composite_score'] = composite_score
            
            scored_events.append(event)
        
        reranked_events = sorted(scored_events, 
                               key=lambda x: x['_composite_score'], 
                               reverse=True)
        
        logger.info(f"BM25 reranking completed")
        
        return reranked_events

    async def rerank_with_cross_encoder(self, events: List[Dict], query: str, 
                                      entities: List[str], time_constraints: Dict) -> List[Dict]:
        if not events or not self.enable_ce_rerank or not self.cross_encoder:
            return events
        
        logger.info(f"Starting cross-encoder reranking for {len(events)} events")
        
        return await self._rerank_with_cross_encoder(events, query, entities, time_constraints)
        

    
    async def _rerank_with_cross_encoder(self, events: List[Dict], query: str, 
                                                 entities: List[str], time_constraints: Dict) -> List[Dict]:
          
        query_doc_pairs = []
        cache_keys = []
        cached_scores = []
        uncached_indices = []
        
        # Truncate inputs to fit the model's context window.
        truncated_query = (
            query[: self.ce_truncate_len]
            if len(query) > self.ce_truncate_len
            else query
        )

        for idx, event in enumerate(events):
            event_text = event.get('sentence', '') or event.get('content', '')
            
            if event_text and len(event_text) > self.ce_truncate_len:
                event_text = event_text[:self.ce_truncate_len]
            
            cache_key = f"{hash(truncated_query)}_{hash(event_text)}"
            cache_keys.append(cache_key)
            
            if hasattr(self, 'cross_encoder_cache') and cache_key in self.cross_encoder_cache:
                cached_scores.append(self.cross_encoder_cache[cache_key])
            else:
                cached_scores.append(None)
                uncached_indices.append(idx)
                if event_text:
                    query_doc_pairs.append([truncated_query, event_text.strip()])
                else:
                    query_doc_pairs.append([truncated_query, ""])

        logger.info(
            f"Cache hit rate: {(len(events) - len(uncached_indices)) / len(events) * 100:.1f}% ({len(events) - len(uncached_indices)}/{len(events)})"
        )

        cross_encoder_scores = [0.0] * len(events)
        
        if query_doc_pairs:
            try:
                batch_size = self._get_optimal_batch_size(len(query_doc_pairs))
                logger.info(f"Using dynamic batch size: {batch_size}")
                
                uncached_scores = []
                for i in range(0, len(query_doc_pairs), batch_size):
                    batch_pairs = query_doc_pairs[i:i + batch_size]
                    batch_scores = self.cross_encoder.predict(batch_pairs)
                    
                    if hasattr(batch_scores, 'tolist'):
                        batch_scores = batch_scores.tolist()
                    elif not isinstance(batch_scores, list):
                        batch_scores = [float(score) for score in batch_scores]
                    
                    uncached_scores.extend(batch_scores)
                    
                    await asyncio.sleep(0)
                
                uncached_idx = 0
                for idx in range(len(events)):
                    if cached_scores[idx] is not None:
                        cross_encoder_scores[idx] = cached_scores[idx]
                    else:
                        score = uncached_scores[uncached_idx] if uncached_idx < len(uncached_scores) else 0.0
                        cross_encoder_scores[idx] = score
                        
                        if hasattr(self, 'cross_encoder_cache'):
                            if len(self.cross_encoder_cache) >= self.ce_cache_size:
                                oldest_key = next(iter(self.cross_encoder_cache))
                                del self.cross_encoder_cache[oldest_key]
                            self.cross_encoder_cache[cache_keys[idx]] = score

                        uncached_idx += 1
                        
            except Exception as e:
                logger.error(f"Error during cross-encoder prediction: {e}")
                cross_encoder_scores = [0.5] * len(events)  # Assign a neutral score
        else:
            for idx in range(len(events)):
                if cached_scores[idx] is not None:
                    cross_encoder_scores[idx] = cached_scores[idx]
        
        if cross_encoder_scores and len(cross_encoder_scores) > 0:
            min_score = min(cross_encoder_scores)
            max_score = max(cross_encoder_scores)
            if max_score > min_score:
                cross_encoder_scores = [
                    (score - min_score) / (max_score - min_score) 
                    for score in cross_encoder_scores
                ]
            else:
                cross_encoder_scores = [0.5] * len(cross_encoder_scores)
        
        if cross_encoder_scores and self.ce_early_stop > 0:
            max_score = max(cross_encoder_scores)
            if max_score >= self.ce_early_stop:
                logger.info(f"Early stopping triggered: top score {max_score:.3f} >= {self.ce_early_stop}")
        
        scored_events = []
        for idx, event in enumerate(events):
            cross_encoder_score = cross_encoder_scores[idx] if idx < len(cross_encoder_scores) else 0.0
            
            # Entity matching score
            entity_score = self.calculate_entity_match_score(
                event.get('sentence', '') or event.get('content', ''), entities
            )
            
            composite_score = (
                cross_encoder_score * self.ce_weight +
                entity_score * self.ce_ent_weight
            )
            
            event['_cross_encoder_score'] = cross_encoder_score
            event['_entity_score'] = entity_score
            event['_composite_score'] = composite_score
            
            scored_events.append(event)
        
        reranked_events = sorted(scored_events, 
                               key=lambda x: x['_composite_score'], 
                               reverse=True)
        
        logger.info(f"Optimized cross-encoder reranking completed")
        
        return reranked_events

    def _get_optimal_batch_size(self, num_pairs: int) -> int:
        """
        Calculate optimal batch size based on workload and GPU memory
        """
        if not self.ce_dynamic_batch:
            return self.ce_batch_size
        
        # Start with base batch size
        optimal_size = self.ce_batch_size
        
        # Adjust based on workload size
        if num_pairs < 32:
            optimal_size = min(num_pairs, self.ce_min_batch)
        elif num_pairs > 200:
            optimal_size = min(self.ce_max_batch, optimal_size * 2)
        
        # Check GPU memory if available
        try:
            import torch
            if torch.cuda.is_available() and self.ce_device != "cpu":
                # Get GPU memory info
                current_memory = torch.cuda.memory_allocated()
                total_memory = torch.cuda.get_device_properties(0).total_memory
                memory_usage_ratio = current_memory / total_memory
                
                if memory_usage_ratio > 0.8:
                    optimal_size = max(self.ce_min_batch, optimal_size // 2)
                elif memory_usage_ratio < 0.4:
                    optimal_size = min(self.ce_max_batch, optimal_size * 1.5)
        except:
            pass
        
        return max(self.ce_min_batch, min(self.ce_max_batch, int(optimal_size)))
    
    async def parse_query_time_and_entities(
        self,
        query: str,
        query_param: QueryParam,
        hashing_kv: BaseKVStorage | None = None,
    ) -> tuple[dict, list[str]]:

        if hashing_kv is None:
            hashing_kv = self.llm_response_cache
            
        # Check if time constraints and entities are already provided in query parameters
        if hasattr(query_param, 'time_constraints') and hasattr(query_param, 'entities'):
            if query_param.time_constraints and query_param.entities:
                return query_param.time_constraints, query_param.entities
            
        global_config = self.get_config_dict()
        time_constraints, entities = await self.aextract_time_and_entities(
            query, query_param, global_config, hashing_kv
        )
        return time_constraints, entities

    async def aextract_time_and_entities(
        self,
        query: str,
        param: QueryParam,
        global_config: dict,
        hashing_kv: BaseKVStorage | None = None,
    ) -> tuple[dict, list[str]]:

        args_hash = compute_args_hash(param.mode, query, "time_entities")
        cached_response = None
        quantized = False
        min_val = 0
        max_val = 0
        
        if hashing_kv and hashing_kv.global_config.get("enable_llm_cache"):
            try:
                cached_item = await hashing_kv.get(args_hash)
                if cached_item and (cached_item.get("cache_type") == "time_entities"):
                    cached_response = cached_item.get("content")
                    quantized = cached_item.get("quantized", False)
                    min_val = cached_item.get("min_val", 0)
                    max_val = cached_item.get("max_val", 0)
                    logger.info(f"Using cached time and entities data for query: {query[:50]}...")
            except Exception as e:
                logger.warning(f"Error retrieving from cache: {e}")
        
        if cached_response:
            try:
                parsed_data = json.loads(cached_response)
                return parsed_data["time_constraints"], parsed_data["entities"]
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(
                    f"Invalid cache format for time constraints and entities: {e}, proceeding with extraction"
                )
        
        example_number = global_config.get("addon_params", {}).get("example_number", None)
        examples = ""
        if "time_entity_extraction_examples" in PROMPTS:
            if example_number and example_number < len(PROMPTS["time_entity_extraction_examples"]):
                examples = "\n".join(
                    PROMPTS["time_entity_extraction_examples"][: int(example_number)]
                )
            else:
                examples = "\n".join(PROMPTS["time_entity_extraction_examples"])
        
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        te_prompt = PROMPTS["time_entity_extraction"]
        te_prompt = te_prompt.replace("{query}", query)
        te_prompt = te_prompt.replace("{examples}", examples)
        te_prompt = te_prompt.replace("{current_date}", current_date)
        
        use_model_func = global_config.get("best_model_func")
        
        llm_kwargs = global_config.get("special_community_report_llm_kwargs", {})
        if not llm_kwargs:
            llm_kwargs = {"response_format": {"type": "json_object"}}
        
        result = await use_model_func(te_prompt, **llm_kwargs)
        
        try:
            parsed_data = json.loads(result)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", result, re.DOTALL)
            if not match:
                logger.error("No JSON-like structure found in the LLM response.")
                return {"start_time": None, "end_time": None}, []
            try:
                parsed_data = json.loads(match.group(0))
            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing error: {e}")
                return {"start_time": None, "end_time": None}, []
        
        time_constraints = parsed_data.get("time_constraints", {"start_time": None, "end_time": None})
        entities = parsed_data.get("entities", [])
        
        if time_constraints.get("start_time"):
            time_constraints["start_time"] = normalize_timestamp(time_constraints["start_time"])
        if time_constraints.get("end_time"):
            time_constraints["end_time"] = normalize_timestamp(time_constraints["end_time"])
        

        if hashing_kv and hashing_kv.global_config.get("enable_llm_cache"):
            cache_data = {
                "time_constraints": time_constraints,
                "entities": entities,
            }
            await hashing_kv.upsert({
                args_hash: {
                    "content": json.dumps(cache_data),
                    "prompt": query,
                    "quantized": quantized,
                    "min_val": min_val,
                    "max_val": max_val,
                    "mode": param.mode,
                    "cache_type": "time_entities",
                }
            })
        
        return time_constraints, entities
