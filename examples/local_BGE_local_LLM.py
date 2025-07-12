from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import os, json, time, asyncio, logging, re
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer, models
from openai import AsyncOpenAI
import aiohttp.client_exceptions
import torch, gc

from graphrag import GraphRAG, QueryParam
from graphrag.base import BaseKVStorage
from graphrag._utils import compute_args_hash, logger

from graphrag import GraphRAG, QueryParam
import json
from tqdm import tqdm
from pathlib import Path

WORK_DIR = Path("work_dir")
WORK_DIR.mkdir(exist_ok=True)
CORPUS_FILE = Path("demo/Corpus.json")

from graphrag._utils import compute_args_hash, logger

import tiktoken
tiktoken.get_encoding("cl100k_base")

logging.basicConfig(level=logging.INFO)
logging.getLogger("DyG-RAG").setLevel(logging.INFO)

################################################################################
# 0. Configuration
################################################################################
def get_config_value(env_name: str, description: str, example: str = None) -> str:
    """Get configuration value from environment variable or user input."""
    value = os.getenv(env_name)
    if value:
        return value
    
    print(f"\n⚠️  Missing configuration: {env_name}")
    print(f"Description: {description}")
    if example:
        print(f"Example: {example}")
    
    while True:
        user_input = input(f"Please enter {env_name}: ").strip()
        if user_input:
            return user_input
        print("❌ Value cannot be empty. Please try again.")

print("🔧 Checking configuration...")
VLLM_BASE_URL = get_config_value(
    "VLLM_BASE_URL", 
    "Base URL for VLLM API service", 
    "http://127.0.0.1:8000/v1"
)

BEST_MODEL_NAME = get_config_value(
    "QWEN_BEST", 
    "Model name for the best/primary LLM", 
    "qwen-14b"
)

LOCAL_BGE_PATH = get_config_value(
    "LOCAL_BGE_PATH", 
    "Local path to BGE embedding model", 
    "/path/to/bge-m3"
)

OPENAI_API_KEY_FAKE = "EMPTY"

@dataclass
class EmbeddingFunc:
    embedding_dim: int
    max_token_size: int
    model: SentenceTransformer

    async def __call__(self, texts: List[str]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        loop = asyncio.get_event_loop()
        encode = lambda: self.model.encode(
            texts,
            batch_size=32,  # Smaller batch size to help with memory issues
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True
        )
        if loop.is_running():
            return await loop.run_in_executor(None, encode)
        else:
            return encode()
    
    # Make the model not serializable for GraphRAG initialization
    def __getstate__(self):
        state = self.__dict__.copy()
        state['model'] = None
        return state
    
    # Restore model reference during deserialization (will be None, but structure preserved)
    def __setstate__(self, state):
        self.__dict__.update(state)

def get_bge_embedding_func() -> EmbeddingFunc:
    gpu_count = torch.cuda.device_count()
    using_cuda = gpu_count > 0
    device = "cuda" if using_cuda else "cpu"

    model_kwargs = {}
    if gpu_count > 1:
        model_kwargs = {"device_map": "auto", "torch_dtype": torch.float16}

    st_model = SentenceTransformer(
        LOCAL_BGE_PATH,
        device=device,              
        trust_remote_code=True,
        model_kwargs=model_kwargs,
    )

    return EmbeddingFunc(
        embedding_dim=st_model.get_sentence_embedding_dimension(),
        max_token_size=8192,        # bge-m3 supports long context
        model=st_model,
    )
    
################################################################################
# 2. LLM call function (with cache)
################################################################################

def _build_async_client() -> AsyncOpenAI:
    return AsyncOpenAI(api_key=OPENAI_API_KEY_FAKE, base_url=VLLM_BASE_URL)

async def _chat_completion(model: str, messages: list[dict[str, str]], **kwargs) -> str:
    client = _build_async_client()
    response = await client.chat.completions.create(model=model, messages=messages, **kwargs)
    return response.choices[0].message.content

async def _llm_with_cache(
    prompt: str,
    *,
    model: str,
    system_prompt: str | None = None,
    history_messages: list[dict[str, str]] | None = None,
    hashing_kv: BaseKVStorage | None = None,
    **kwargs,
) -> str:
    """General LLM wrapper, supporting GraphRAG cache interface."""
    history_messages = history_messages or []
    msgs = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    msgs.extend(history_messages)
    msgs.append({"role": "user", "content": prompt})

    if hashing_kv is not None:
        args_hash = compute_args_hash(model, msgs)
        cached = await hashing_kv.get_by_id(args_hash)
        if cached is not None:
            return cached["return"]

    answer = await _chat_completion(model=model, messages=msgs, **kwargs)

    if hashing_kv is not None:
        await hashing_kv.upsert({args_hash: {"return": answer, "model": model}})
        await hashing_kv.index_done_callback()
    return answer

async def best_model_func(prompt: str, system_prompt: str | None = None, history_messages: list[dict[str,str]] | None = None, **kwargs) -> str:
    return await _llm_with_cache(
        prompt,
        model=BEST_MODEL_NAME,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )

embedding_func = get_bge_embedding_func()
model_ref = embedding_func.model
embedding_func.model = None 

def read_json_file(fp: Path):
    with fp.open(encoding="utf-8") as f:
        return json.load(f)

graph_func = GraphRAG(
    working_dir=str(WORK_DIR),
    embedding_func=embedding_func,
    best_model_func=best_model_func,
    cheap_model_func=best_model_func,
    enable_llm_cache=True,
    best_model_max_token_size = 16384,
    cheap_model_max_token_size = 16384,
    model_path="./models",  
    ce_model="cross-encoder/ms-marco-TinyBERT-L-2-v2",  
    ner_model_name="dslim_bert_base_ner", 
)

embedding_func.model = model_ref

corpus_data = read_json_file(CORPUS_FILE)
total_docs = len(corpus_data)
logger.info(f"Start processing, total {total_docs} documents to process.")

all_docs = []
for idx, obj in enumerate(tqdm(corpus_data, desc="Loading docs", total=total_docs)):
    # Combine metadata with content
    enriched_content = f"Title: {obj['title']}\nDocument ID: {obj['doc_id']}\n\n{obj['context']}"
    all_docs.append(enriched_content)
 
graph_func.insert(all_docs)

print(graph_func.query("Where was Barbara Hammer educated after Mar 1962?", param=QueryParam(mode="dynamic")))
