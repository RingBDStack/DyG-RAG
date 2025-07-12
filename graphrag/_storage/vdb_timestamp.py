import numpy as np
import math
import datetime
from dataclasses import dataclass
from dateutil import parser as date_parser
from typing import List, Dict, Any, Optional, Union, Tuple
import os
import calendar

from .vdb_nanovectordb import NanoVectorDBStorage, NanoVectorDB
from .._utils import logger


def encode_timestamp(timestamp: str, dim: int = 8) -> np.ndarray:
    """
    Encodes an ISO format timestamp (YYYY-MM-DD, YYYY-MM, YYYY) or "static" into a fixed-dimension embedding vector.
    This function uses Fourier feature mapping to convert the timestamp into a normalized continuous value,
    which is then encoded using a combination of sine and cosine functions at different frequencies.
    This method aims to ensure that temporally adjacent timestamps have more similar embedding representations.

    Args:
        timestamp: An ISO format timestamp string (e.g., "2023", "2023-06", "2023-06-15") or the special value "static".
        dim: The dimension of the output embedding, which must be a multiple of 4 (e.g., 8, 12, 16).
             Fourier feature mapping requires the dimension to be even, and this condition satisfies that requirement.

    Returns:
        np.ndarray: A numpy array of shape (dim,) representing the timestamp's embedding.
                    Returns a zero vector if the timestamp cannot be parsed or is "static".
    """
    if timestamp == "static" or not timestamp:
        return np.zeros(dim)
    
    if dim % 4 != 0:
        raise ValueError("Timestamp encoding dimension must be a multiple of 4")
    
    year, month, day = 0, 0, 0
    precision_level = -1  # 0: YYYY, 1: YYYY-MM, 2: YYYY-MM-DD / full
    
    try:
        if len(timestamp) == 4:  # YYYY
            year = int(timestamp)
            month, day = 1, 1  # Default for YYYY precision
            precision_level = 0
        elif len(timestamp) == 7:  # YYYY-MM
            year_str, month_str = timestamp.split('-')
            year, month = int(year_str), int(month_str)
            if not (1 <= month <= 12):
                raise ValueError("Month out of range.")
            day = 1  # Default for YYYY-MM precision
            precision_level = 1
        elif len(timestamp) == 10:  # YYYY-MM-DD
            year_str, month_str, day_str = timestamp.split('-')
            year, month, day = int(year_str), int(month_str), int(day_str)
            datetime.datetime(year, month, day)  # Validates date components
            precision_level = 2
        else:
            # Try to parse other formats
            dt = date_parser.parse(timestamp)
            year, month, day = dt.year, dt.month, dt.day
            precision_level = 2  # Assume full precision from generic parser
    except (ValueError, TypeError, AttributeError):
        return np.zeros(dim)  # Parsing failed
    
    # Construct continuous time value (t_val)
    t_val: float
    if precision_level == 0:  # YYYY
        t_val = float(year)
    elif precision_level == 1:  # YYYY-MM
        t_val = float(year) + (float(month) - 0.5) / 12.0
    elif precision_level == 2:  # YYYY-MM-DD or parsed
        day_of_year = datetime.datetime(year, month, day).timetuple().tm_yday
        days_in_year = 366.0 if calendar.isleap(year) else 365.0
        t_val = float(year) + (float(day_of_year) - 0.5) / days_in_year
    else: # Should not happen if parsing logic is correct and leads to return np.zeros(dim)
        return np.zeros(dim)
    
    # Normalize t_val (e.g., mapping years 1900-2100 to approx [0,1])
    # This range matches the year normalization in the original code.
    t_norm = (t_val - 1900.0) / 200.0
    
    # Apply Fourier Feature Mapping
    encoding = np.zeros(dim)
    num_freq_bands = dim // 2  # dim is already validated to be even (multiple of 4)
    
    for k in range(num_freq_bands):
        freq_val = (2.0 ** k) * math.pi * t_norm
        encoding[2 * k] = math.sin(freq_val)
        encoding[2 * k + 1] = math.cos(freq_val)
        
    return encoding


@dataclass
class TimestampEnhancedVectorStorage(NanoVectorDBStorage):
    """
    Extends NanoVectorDBStorage to support time-aware vector storage.
    Concatenates timestamp encodings with semantic embeddings to enable time-aware similarity search.
    """
    timestamp_dim: int = 16  # Dimension of the timestamp encoding
    
    def __post_init__(self):
        # Ensure the timestamp dimension is a multiple of 4
        if self.timestamp_dim % 4 != 0:
            self.timestamp_dim = (self.timestamp_dim // 4) * 4
            logger.warning(f"Adjusted timestamp_dim to {self.timestamp_dim} to ensure it's a multiple of 4")
            
        # Do not call the parent class initializer, directly implement the required initialization logic
        # Get the embedding dimension and add the timestamp dimension
        embed_dim = getattr(self.embedding_func, 'embedding_dim', 1536)
        self.original_dim = embed_dim
        self.enhanced_dim = self.original_dim + self.timestamp_dim
        
        # The following logic is copied from the parent class, but uses enhanced_dim instead of the original embed_dim
        self._client_file_name = os.path.join(
            self.global_config["working_dir"], f"vdb_{self.namespace}.json"
        )
        self._max_batch_size = self.global_config["embedding_batch_num"]
        
        try:
            logger.info(f"Initializing TimestampEnhancedVectorDB with embedding dimension {self.enhanced_dim}")
            
            # Create the client directly with the enhanced dimension
            self._client = NanoVectorDB(
                self.enhanced_dim, storage_file=self._client_file_name
            )
            
            # Verify that the storage matrix is correctly initialized
            if hasattr(self._client, '__storage') and 'matrix' in self._client.__storage:
                if self._client.__storage['matrix'] is None or len(self._client.__storage['matrix']) == 0:
                    logger.warning("Vector database matrix is empty, it will be initialized on first upsert")
            
            self.cosine_better_than_threshold = self.global_config.get(
                "query_better_than_threshold", self.cosine_better_than_threshold
            )
        except Exception as e:
            logger.error(f"Error initializing TimestampEnhancedVectorStorage: {e}")
            # Ensure a valid client is created, using the correct dimension even if an error occurs
            self._client = NanoVectorDB(
                self.enhanced_dim, storage_file=self._client_file_name
            )
            self.cosine_better_than_threshold = self.global_config.get(
                "query_better_than_threshold", self.cosine_better_than_threshold
            )
            
        logger.info(f"Initialized TimestampEnhancedVectorStorage with original_dim={self.original_dim}, "
                   f"timestamp_dim={self.timestamp_dim}, enhanced_dim={self.enhanced_dim}")
    
    async def upsert(self, data: Dict[str, Dict[str, Any]]) -> bool:
        logger.info(f"Inserting {len(data)} vectors with timestamp enhancement to {self.namespace}")
        if not len(data):
            logger.warning("You insert an empty data to vector DB")
            return []
            
        # Extract metadata and content
        list_data = [
            {
                "__id__": k,
                **{k1: v1 for k1, v1 in v.items() if k1 in self.meta_fields},
            }
            for k, v in data.items()
        ]
        contents = [v["content"] for v in data.values()]
        
        # Create batches based on configuration to reduce peak GPU memory usage
        batches = [
            contents[i : i + self._max_batch_size]
            for i in range(0, len(contents), self._max_batch_size)
        ]
        
        # Process in batches to reduce peak GPU memory usage
        logger.debug(f"Processing {len(batches)} embedding batches with size up to {self._max_batch_size}")
        embeddings_list = []
        for batch in batches:
            emb = await self.embedding_func(batch)
            embeddings_list.append(emb)
            # Clear CUDA cache to prevent OOM
            try:
                import torch, gc
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
            except ImportError:
                pass
        
        # Concatenate all embeddings into a single array
        embeddings = np.concatenate(embeddings_list)
        
        # Get and encode the timestamp for each document
        timestamp_embeddings = []
        for doc_id, doc in data.items():
            timestamp = doc.get("timestamp", "static")
            ts_encoding = encode_timestamp(timestamp, self.timestamp_dim)
            timestamp_embeddings.append(ts_encoding)
        
        # Concatenate timestamp encodings with semantic embeddings
        timestamp_embeddings = np.array(timestamp_embeddings)
        enhanced_embeddings = np.hstack((embeddings, timestamp_embeddings))
        
        # Add the enhanced embeddings to the metadata
        for i, d in enumerate(list_data):
            d["__vector__"] = enhanced_embeddings[i]
            
        # Insert into the vector database
        results = self._client.upsert(datas=list_data)
        return results
    
    async def query(self, query: str, top_k: int = 5, time_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        # Get the query embedding
        query_embedding = await self.embedding_func([query])
        query_embedding = query_embedding[0]
        
        # Create timestamp encoding (defaults to all zeros, meaning time is not considered)
        if time_filter:
            ts_encoding = encode_timestamp(time_filter, self.timestamp_dim)
        else:
            ts_encoding = np.zeros(self.timestamp_dim)
        
        # Concatenate the query embedding and timestamp encoding
        enhanced_query = np.hstack((query_embedding, ts_encoding))
        
        # Execute the query
        results = self._client.query(
            query=enhanced_query,
            top_k=top_k,
            better_than_threshold=self.cosine_better_than_threshold,
        )
        
        # Process the results
        results = [
            {**dp, "id": dp["__id__"], "distance": dp["__metrics__"]} for dp in results
        ]
        return results
    
    async def time_weighted_query(self, query: str, timestamp: str, time_weight: Optional[float] = None, top_k: int = 5) -> List[Dict[str, Any]]:
         # Get the query embedding
        query_embedding = await self.embedding_func([query])
        query_embedding = query_embedding[0]
        
        # Get the timestamp encoding
        ts_encoding = encode_timestamp(timestamp, self.timestamp_dim)
        
        # Apply the time weight
        actual_time_weight = time_weight if time_weight is not None else 1.0
        scaled_ts_encoding = ts_encoding * actual_time_weight
        
        # Concatenate the semantic embedding and the weighted timestamp embedding
        enhanced_query = np.hstack((query_embedding, scaled_ts_encoding))
        
        # Execute the query
        results = self._client.query(
            query=enhanced_query,
            top_k=top_k,
            better_than_threshold=self.cosine_better_than_threshold,
        )
        
        # Process the results
        results = [
            {**dp, "id": dp["__id__"], "distance": dp["__metrics__"]} 
            for dp in results
        ]
        return results 