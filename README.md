# DyG-RAG: Dynamic Graph Retrieval-Augmented Generation with Event-Centric Reasoning

## Quick Start

### 1. Environment Setup

```bash
cd DyG-RAG
conda create -n dygrag python=3.10
conda activate dygrag
# Install dependencies
pip install -r requirements.txt
```

### 2. OpenAI Configuration

Set your OpenAI API key:

```bash
# Option 1: Environment variable
export OPENAI_API_KEY="your_api_key_here"
```

### 3. Download Models

```bash
# Download required models
python models/download.py
```

**Downloaded Models:**

The script downloads two essential models:

1. **Cross-Encoder Model**:

   - Model name: `cross-encoder/ms-marco-TinyBERT-L-2-v2`
   - Local path: `./models/cross-encoder_ms-marco-TinyBERT-L-2-v2/`
   - Purpose: Semantic reranking and relevance scoring
   - Size: ~67MB
2. **NER Model**:

   - Model name: `dslim/bert-base-NER`
   - Local path: `./models/dslim_bert_base_ner/`
   - Purpose: Named entity recognition
   - Size: ~1.2GB

### 4. Local Model Configuration

If you want to use local models instead of OpenAI, you can set up environment variables for the local embedding and LLM configuration (using BGE and Qwen as examples):

```bash
# Local BGE embedding model path (example: BGE-M3)
export LOCAL_BGE_PATH="/path/to/your/bge-m3"

# VLLM API service URL (example: local VLLM server)
export VLLM_BASE_URL="http://127.0.0.1:8000/v1"

# Model name for LLM (example: Qwen model)
export QWEN_BEST="qwen-14b"
```

### 5. Run Examples

```bash
# Basic usage with OpenAI
cd examples
python openai_all.py

# Using local models
python local_BGE_local_LLM.py
```

## Project Structure

```
DyG-RAG/
├── requirements.txt    # Dependencies
├── datasets/           # three types of temporal reasoning
├── examples/          # Usage examples
├── graphrag/         # src codes of DyG-RAG
├── demo/             # TimeQA dataset examples
└── models/           # Downloaded models
```

## TODO

1. 🤖 Support for More Model Choices
2. 🗄️ Support for Diverse Vector Databases
3. 📊 Support for Diverse Graph Databases

## Acknowledgments

This project is built upon the excellent work of [nano-graphrag](https://github.com/gusye1234/nano-graphrag) by Gustavo Ye. We extend our sincere gratitude to the original author for providing a simple, easy-to-hack GraphRAG implementation that serves as the foundation for our DyG-RAG system.
