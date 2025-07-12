import asyncio
import os
from dataclasses import dataclass
import numpy as np
from nano_vectordb import NanoVectorDB
from typing import List, Optional

from .._utils import logger
from ..base import BaseVectorStorage


@dataclass
class NanoVectorDBStorage(BaseVectorStorage):
    cosine_better_than_threshold: float = 0.2

    def __post_init__(self):
        self._client_file_name = os.path.join(
            self.global_config["working_dir"], f"vdb_{self.namespace}.json"
        )
        self._max_batch_size = self.global_config["embedding_batch_num"]
        
        try:
            # Try to get the embedding dimension
            embed_dim = getattr(self.embedding_func, 'embedding_dim', 1536)
            logger.info(f"Initializing NanoVectorDB with embedding dimension {embed_dim}")
            
            # Create client
            self._client = NanoVectorDB(
                embed_dim, storage_file=self._client_file_name
            )
            
            # Verify that the storage matrix is correctly initialized
            if hasattr(self._client, '__storage') and 'matrix' in self._client.__storage:
                if self._client.__storage['matrix'] is None or len(self._client.__storage['matrix']) == 0:
                    logger.warning("Vector database matrix is empty, it will be initialized on first upsert")
            
            self.cosine_better_than_threshold = self.global_config.get(
                "query_better_than_threshold", self.cosine_better_than_threshold
            )
        except Exception as e:
            logger.error(f"Error initializing NanoVectorDBStorage: {e}")
            self._client = NanoVectorDB(
                1536, storage_file=self._client_file_name
            )
            self.cosine_better_than_threshold = self.global_config.get(
                "query_better_than_threshold", self.cosine_better_than_threshold
            )

    async def upsert(self, data: dict[str, dict]):
        logger.info(f"Inserting {len(data)} vectors to {self.namespace}")
        if not len(data):
            logger.warning("You insert an empty data to vector DB")
            return []
            
        try:
            # Extract metadata and content
            list_data = [
                {
                    "__id__": k,
                    **{k1: v1 for k1, v1 in v.items() if k1 in self.meta_fields},
                }
                for k, v in data.items()
            ]
            contents = [v["content"] for v in data.values()]
            
            # Create optimized batches based on configured batch size
            batches = [
                contents[i : i + self._max_batch_size]
                for i in range(0, len(contents), self._max_batch_size)
            ]
            
            # Process batches sequentially to reduce peak GPU memory usage
            logger.debug(f"Processing {len(batches)} embedding batches of size up to {self._max_batch_size} sequentially")
            embeddings_list = []
            for batch in batches:
                try:
                    emb = await self.embedding_func(batch)
                    if emb is None or len(emb) == 0:
                        logger.error(f"Embedding function returned empty result for batch of size {len(batch)}")
                        continue
                    embeddings_list.append(emb)
                    # Clear CUDA cache to prevent OOM
                    try:
                        import torch, gc
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        gc.collect()
                    except ImportError:
                        pass
                except Exception as batch_err:
                    logger.error(f"Error processing batch: {batch_err}")
                    continue
            
            # Check if there are any valid embeddings
            if not embeddings_list:
                logger.error("No valid embeddings were generated for any batch")
                return []
            
            # Combine all embeddings into a single array
            embeddings = np.concatenate(embeddings_list)
            
            # Validate embedding dimension
            expected_dim = getattr(self.embedding_func, 'embedding_dim', None)
            actual_dim = embeddings.shape[1] if len(embeddings.shape) > 1 else None
            
            if expected_dim is not None and actual_dim is not None and actual_dim != expected_dim:
                logger.warning(f"Embedding dimension mismatch: expected {expected_dim}, got {actual_dim}")
            
            # Attach embeddings to the metadata
            for i, d in enumerate(list_data):
                if i < len(embeddings):
                    d["__vector__"] = embeddings[i]
                else:
                    logger.error(f"Index {i} out of bounds for embeddings with length {len(embeddings)}")
                    # Skip out-of-bounds index
                    continue
            
            # Get valid data (filter out data without vectors)
            valid_list_data = [d for d in list_data if "__vector__" in d]
            if not valid_list_data:
                logger.error("No valid data with vectors to insert")
                return []
            
            # Get client vector dimension
            client_dim = getattr(self._client, 'vec_dim', None)
            
            # If client dimension does not match vector dimension, reinitialize the client
            first_vector_dim = len(valid_list_data[0]["__vector__"])
            if client_dim is not None and client_dim != first_vector_dim:
                logger.warning(f"Vector dimension mismatch: client expects {client_dim}, got {first_vector_dim}")
                logger.info(f"Reinitializing vector database with new dimension: {first_vector_dim}")
                self._client = NanoVectorDB(
                    first_vector_dim, storage_file=self._client_file_name
                )
            
            # Insert into vector database with error handling
            try:
                results = self._client.upsert(datas=valid_list_data)
                return results
            except IndexError as e:
                # Usually a matrix size issue, attempting to fix
                logger.error(f"IndexError during upsert: {e}")
                logger.info("Attempting to fix and reinitialize vector database")
                
                # Recreate the client
                self._client = NanoVectorDB(
                    first_vector_dim, storage_file=self._client_file_name
                )
                
                # Retry insertion
                try:
                    results = self._client.upsert(datas=valid_list_data)
                    return results
                except Exception as retry_e:
                    logger.error(f"Retry failed: {retry_e}")
                    return []
                    
        except Exception as e:
            logger.error(f"Error during vector database upsert: {e}")
            return []

    async def query(self, query: str, top_k=5):
        try:
            embedding = await self.embedding_func([query])
            embedding = embedding[0]
            
            # Validate query vector dimension
            client_dim = getattr(self._client, 'vec_dim', None)
            if client_dim is not None and len(embedding) != client_dim:
                logger.warning(f"Query vector dimension {len(embedding)} doesn't match database dimension {client_dim}")
                # Resize vector to match
                embedding = np.resize(embedding, (client_dim,))
            
            results = self._client.query(
                query=embedding,
                top_k=top_k,
                better_than_threshold=self.cosine_better_than_threshold,
            )
            results = [
                {**dp, "id": dp["__id__"], "distance": dp["__metrics__"]} for dp in results
            ]
            return results
        except Exception as e:
            logger.error(f"Error during vector query: {e}")
            return []

    async def get_by_id(self, id_: str) -> Optional[dict]:
        """
        Get a single vector data by ID, including text content
        
        Args:
            id_: ID to get
            
        Returns:
            Optional[dict]: A dictionary containing text content and metadata, or None if the ID does not exist
        """
        try:
            # Method 1: Try using the get method (if it exists)
            if hasattr(self._client, 'get'):
                try:
                    result = self._client.get([id_])
                    if result and len(result) > 0:
                        data = result[0]
                        if data:
                            # Build return data
                            result_dict = {
                                "id": data.get("__id__", id_),
                                "distance": 0.0,  # get method does not return distance
                            }
                            
                            # Add all metadata fields
                            for k, v in data.items():
                                if k not in ["__id__", "__vector__", "__metrics__"]:
                                    result_dict[k] = v
                            
                            return result_dict
                except Exception as e:
                    logger.debug(f"get method failed for ID {id_}: {e}")
            
            # Method 2: Fallback to query method
            embed_dim = getattr(self.embedding_func, 'embedding_dim', 1536)
            
            # Use query method to search all data
            results = self._client.query(
                query=np.zeros(embed_dim),  # Use zero vector as query
                top_k=1000,  # Get more results to ensure the target ID is found
                better_than_threshold=0.0  # Do not use similarity threshold
            )
            
            # Find matching ID in results
            for item in results:
                if item.get("__id__") == id_:
                    # Build return data, including all original fields
                    result = {
                        "id": item["__id__"],
                        "distance": item.get("__metrics__", 0.0),
                    }
                    
                    # Add all metadata fields, including content etc.
                    for k, v in item.items():
                        if k not in ["__id__", "__vector__", "__metrics__"]:
                            result[k] = v
                    
                    return result
            return None
        except Exception as e:
            logger.error(f"Error during get_by_id for ID {id_}: {e}")
            return None

    async def get_by_ids(self, ids: List[str]) -> List[dict]:
        """
        Get vector data in batch by a list of IDs
        
        Args:
            ids: List of IDs to get
            
        Returns:
            List[dict]: A list of dictionaries containing vector data, with None for non-existent IDs
        """
        try:
            # Method 1: Try using the get method (if it exists)
            if hasattr(self._client, 'get'):
                try:
                    records = self._client.get(ids)
                    results = []
                    
                    # Create a mapping from ID to record
                    id_to_record = {record.get("__id__"): record for record in records if record}
                    
                    # Return results in the order of input IDs
                    for id_ in ids:
                        if id_ in id_to_record:
                            record = id_to_record[id_]
                            # Build return data
                            result_dict = {
                                "id": record["__id__"],
                                "distance": 0.0,  # get method does not return distance
                            }
                            
                            # Add all metadata fields, including content etc.
                            for k, v in record.items():
                                if k not in ["__id__", "__vector__", "__metrics__"]:
                                    result_dict[k] = v
                            
                            results.append(result_dict)
                        else:
                            results.append(None)
                    
                    return results
                except Exception as e:
                    logger.error(f"get method failed: {e}")
            
            # Method 2: Fallback to calling get_by_id one by one
            results = []
            for id_ in ids:
                result = await self.get_by_id(id_)
                results.append(result)
                # Allow other async tasks to run
                await asyncio.sleep(0)
            return results
        except Exception as e:
            logger.error(f"Error during get_by_ids: {e}")
            return [None] * len(ids)

    async def index_done_callback(self):
        try:
            self._client.save()
        except Exception as e:
            logger.error(f"Error during index_done_callback: {e}")
