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

# Check for OPENAI_API_KEY environment variable
def check_openai_api_key():
    """
    Check if OPENAI_API_KEY is set and not empty.
    If not set, allow user to input manually.
    Raises SystemExit if not properly configured.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    
    if api_key is None or not api_key.strip():
        print("❌ OPENAI_API_KEY environment variable is not set or empty.")
        print("\nOptions:")
        print("1. Set environment variable: export OPENAI_API_KEY='your-api-key-here'")
        print("2. Enter API key manually now (will be set for this session)")
        
        choice = input("\nWould you like to enter your API key manually? (y/N): ")
        if choice.lower() == 'y':
            manual_key = input("Please enter your OpenAI API key: ").strip()
            if not manual_key:
                print("❌ Error: No API key provided.")
                sys.exit(1)
            # Set the environment variable for this session
            os.environ["OPENAI_API_KEY"] = manual_key
            api_key = manual_key
            print("✅ API key has been set for this session.")
        else:
            print("❌ Cannot proceed without API key.")
            sys.exit(1)
    
    # Basic format validation (OpenAI keys typically start with 'sk-')
    if not api_key.startswith('sk-'):
        print("⚠️  Warning: OPENAI_API_KEY doesn't appear to be in the expected format (should start with 'sk-')")
        print(f"   Current key starts with: {api_key[:10]}...")
        response = input("Do you want to continue anyway? (y/N): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    print("✅ OPENAI_API_KEY is properly configured.")
    return True

# Check API key before proceeding
check_openai_api_key()

WORK_DIR = Path("work_dir")
WORK_DIR.mkdir(exist_ok=True)
CORPUS_FILE = Path("demo/Corpus.json")

logging.basicConfig(level=logging.INFO)
logging.getLogger("DyG-RAG").setLevel(logging.INFO)


def read_json_file(fp: Path):
    with fp.open(encoding="utf-8") as f:
        return json.load(f)

graph_func = GraphRAG(
    working_dir=str(WORK_DIR),
    best_model_max_token_size = 16384,
    cheap_model_max_token_size = 16384
)

# Read the JSON file
corpus_data = read_json_file(CORPUS_FILE)
total_docs = len(corpus_data)
logger.info(f"Start processing, total {total_docs} documents to process.")

all_docs = []
for idx, obj in enumerate(tqdm(corpus_data, desc="Loading docs", total=total_docs)):
    enriched_content = f"Title: {obj['title']}\nDocument ID: {obj['doc_id']}\n\n{obj['context']}"
    all_docs.append(enriched_content)
 
# insert_docs = all_docs[:10]
# graph_func.insert(insert_docs)
graph_func.insert(all_docs)

print(graph_func.query("Which position did Pat Duncan hold in Feb 1996?", param=QueryParam(mode="dynamic")))
