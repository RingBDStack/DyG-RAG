#!/usr/bin/env python3
"""
Download script for DyG-RAG auxiliary models
Downloads Cross-Encoder rerank model and NER model to local directories
"""

import os
import sys
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_script_directory():
    """Get the directory where this script is located"""
    return Path(__file__).parent.absolute()

def test_cross_encoder_model(model_path):
    """Simple functionality test for Cross-Encoder model"""
    logger.info("Testing Cross-Encoder model functionality...")
    try:
        from sentence_transformers import CrossEncoder
        
        # Load the model
        model = CrossEncoder(model_path)
        
        # Simple test with one query-document pair
        test_pairs = [
            ["What is Python?", "Python is a programming language."],
            ["Machine learning", "AI and machine learning are related fields."]
        ]
        
        # Get scores
        scores = model.predict(test_pairs)
        
        # Basic validation
        if len(scores) == len(test_pairs):
            logger.info(f"✓ Cross-Encoder test passed: {len(scores)} scores generated")
            logger.info(f"✓ Sample scores: {[f'{score:.3f}' for score in scores]}")
        else:
            logger.warning("⚠ Cross-Encoder test warning: Unexpected result")
            
    except Exception as e:
        logger.error(f"Cross-Encoder test failed: {e}")
        logger.warning("Model downloaded but functionality test failed")

def download_cross_encoder_model():
    """Download Cross-Encoder model with retry mechanism and functionality verification"""
    try:
        from sentence_transformers import CrossEncoder
        import time
        import shutil
        import tempfile
        import os
        
        model_name = "cross-encoder/ms-marco-TinyBERT-L-2-v2"
        script_dir = get_script_directory()
        local_path = script_dir / model_name.replace("/", "_")
        temp_dir = None
        
        logger.info(f"Downloading Cross-Encoder model: {model_name}")
        logger.info(f"Local path: {local_path}")
        
        # Backup existing model if it exists
        if local_path.exists():
            backup_path = str(local_path) + f"_backup_{int(time.time())}"
            shutil.move(str(local_path), backup_path)
            logger.info(f"Backed up existing model to: {backup_path}")
        
        # Download with retry mechanism
        max_retries = 3
        success = False
        model = None
        
        for attempt in range(max_retries):
            logger.info(f"Download attempt {attempt + 1}/{max_retries}")
            
            try:
                # Create temporary directory for this attempt
                temp_dir = tempfile.mkdtemp(prefix="cross_encoder_download_")
                
                # Download model to temporary location first
                logger.info("Downloading Cross-Encoder model...")
                model = CrossEncoder(model_name, cache_folder=temp_dir)
                
                success = True
                logger.info("Download completed successfully")
                break
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                
                # Clean up failed attempt
                if temp_dir and os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
                    temp_dir = None
                
                if "IncompleteRead" in str(e) or "Connection broken" in str(e):
                    logger.info("Network connection interrupted, retrying...")
                    time.sleep(5)  # Wait 5 seconds before retry
                elif "timeout" in str(e).lower():
                    logger.info("Download timeout, retrying...")
                    time.sleep(10)  # Wait longer for timeout
                elif "HTTPError" in str(e) or "ConnectionError" in str(e):
                    logger.info("HTTP/Connection error, retrying...")
                    time.sleep(3)
                else:
                    logger.info(f"Other error, retrying...")
                    time.sleep(3)
                
                if attempt == max_retries - 1:
                    logger.error(f"All {max_retries} attempts failed")
                    raise
        
        if not success or model is None:
            raise Exception("Failed to download model after all retries")
        
        # Create final local directory
        local_path.mkdir(parents=True, exist_ok=True)
        
        # Save model to final location
        logger.info("Saving model to final location...")
        model.save(str(local_path))
        
        # Clean up temporary directory
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        
        # Verify functionality
        test_cross_encoder_model(str(local_path))
        
        logger.info(f"Cross-Encoder model downloaded successfully to {local_path}")
        return True
        
    except ImportError:
        logger.error("sentence-transformers not installed. Please install it first:")
        logger.error("pip install sentence-transformers")
        return False
    except Exception as e:
        logger.error(f"Failed to download Cross-Encoder model: {e}")
        
        # Clean up on failure
        if 'temp_dir' in locals() and temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        
        return False

def download_ner_model():
    """Download NER model with retry mechanism and functionality verification"""
    try:
        from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
        import torch
        import time
        import shutil
        
        model_name = "dslim/bert-base-NER"
        script_dir = get_script_directory()
        local_path = script_dir / "dslim_bert_base_ner"
        temp_cache_dir = script_dir / "temp_ner_cache"
        
        logger.info(f"Downloading NER model: {model_name}")
        logger.info(f"Local path: {local_path}")
        
        # Clean up existing directories
        if local_path.exists():
            backup_path = str(local_path) + f"_backup_{int(time.time())}"
            shutil.move(str(local_path), backup_path)
            logger.info(f"Backed up existing model to: {backup_path}")
        
        if temp_cache_dir.exists():
            shutil.rmtree(temp_cache_dir)
        
        # Download with retry mechanism
        max_retries = 3
        success = False
        
        for attempt in range(max_retries):
            logger.info(f"Download attempt {attempt + 1}/{max_retries}")
            
            try:
                # Create temporary cache directory
                temp_cache_dir.mkdir(parents=True, exist_ok=True)
                
                # Download tokenizer first
                logger.info("Downloading tokenizer...")
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name, 
                    cache_dir=str(temp_cache_dir),
                    resume_download=True
                )
                
                # Wait a bit to avoid connection issues
                time.sleep(2)
                
                # Download model
                logger.info("Downloading model...")
                model = AutoModelForTokenClassification.from_pretrained(
                    model_name, 
                    cache_dir=str(temp_cache_dir),
                    resume_download=True
                )
                
                success = True
                logger.info("Download completed successfully")
                break
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                
                # Clean up failed attempt
                if temp_cache_dir.exists():
                    shutil.rmtree(temp_cache_dir)
                
                if "IncompleteRead" in str(e) or "Connection broken" in str(e):
                    logger.info("Network connection interrupted, retrying...")
                    time.sleep(5)  # Wait 5 seconds before retry
                elif "timeout" in str(e).lower():
                    logger.info("Download timeout, retrying...")
                    time.sleep(10)  # Wait longer for timeout
                else:
                    time.sleep(3)
                
                if attempt == max_retries - 1:
                    logger.error(f"All {max_retries} attempts failed")
                    raise
        
        if not success:
            raise Exception("Failed to download model after all retries")
        
        # Create final local directory
        local_path.mkdir(parents=True, exist_ok=True)
        
        # Copy files from cache to final location
        logger.info("Organizing downloaded files...")
        
        # Find the actual model files in cache
        cache_models_dir = temp_cache_dir / "models--dslim--bert-base-NER"
        if cache_models_dir.exists():
            snapshots_dir = cache_models_dir / "snapshots"
            if snapshots_dir.exists():
                # Find the latest snapshot
                snapshot_dirs = [d for d in snapshots_dir.iterdir() if d.is_dir()]
                if snapshot_dirs:
                    latest_snapshot = snapshot_dirs[0]  # Usually there's only one
                    
                    # Copy all files from snapshot to local path
                    for file_path in latest_snapshot.iterdir():
                        if file_path.is_file() or file_path.is_symlink():
                            target_path = local_path / file_path.name
                            if file_path.is_symlink():
                                # Resolve symlink and copy actual file
                                shutil.copy2(file_path.resolve(), target_path)
                            else:
                                shutil.copy2(file_path, target_path)
        
        # Also save using transformers' save_pretrained method
        logger.info("Saving model in standard format...")
        tokenizer.save_pretrained(str(local_path))
        model.save_pretrained(str(local_path))
        
        # Clean up temporary cache
        if temp_cache_dir.exists():
            shutil.rmtree(temp_cache_dir)
        
        # Verify functionality
        logger.info("Verifying model functionality...")
        try:
            # Test the downloaded model
            test_tokenizer = AutoTokenizer.from_pretrained(str(local_path))
            test_model = AutoModelForTokenClassification.from_pretrained(str(local_path))
            test_model.eval()
            
            # Create pipeline for testing
            device = 0 if torch.cuda.is_available() else -1
            test_pipeline = pipeline(
                "ner",
                model=test_model,
                tokenizer=test_tokenizer,
                device=device,
                aggregation_strategy="simple"
            )
            
            # Test with a simple sentence
            test_sentence = "John works at Apple Inc. in New York."
            result = test_pipeline(test_sentence)
            
            entities_count = len(result) if result else 0
            if entities_count > 0:
                logger.info(f"✓ Functionality test passed: {entities_count} entities detected")
                logger.info(f"✓ Example entities: {[r['word'] for r in result[:3]]}")
            else:
                logger.warning("⚠ Functionality test warning: No entities detected")
                logger.warning("  Model downloaded but may have issues")
            
        except Exception as test_error:
            logger.error(f"Functionality test failed: {test_error}")
            logger.warning("Model downloaded but functionality cannot be verified")
        
        logger.info(f"NER model downloaded successfully to {local_path}")
        return True
        
    except ImportError:
        logger.error("transformers not installed. Please install it first:")
        logger.error("pip install transformers torch")
        return False
    except Exception as e:
        logger.error(f"Failed to download NER model: {e}")
        
        # Clean up on failure
        if 'temp_cache_dir' in locals() and temp_cache_dir.exists():
            shutil.rmtree(temp_cache_dir)
        
        return False

def verify_model_download(model_type: str, local_path: str) -> bool:
    """Verify that the model was downloaded correctly"""
    path = Path(local_path)
    
    if not path.exists():
        logger.error(f"Model directory does not exist: {local_path}")
        return False
    
    if model_type == "cross_encoder":
        # Check for Cross-Encoder specific files
        required_files = ["config.json", "pytorch_model.bin"]
        # Also check for safetensors format
        safetensors_files = list(path.glob("*.safetensors"))
        if safetensors_files:
            required_files = ["config.json"]  # If safetensors exist, we don't need pytorch_model.bin
    
    elif model_type == "ner":
        # Check for NER model files
        required_files = ["config.json", "pytorch_model.bin", "tokenizer.json", "tokenizer_config.json"]
        # Also check for safetensors format
        safetensors_files = list(path.glob("*.safetensors"))
        if safetensors_files:
            required_files = ["config.json", "tokenizer.json", "tokenizer_config.json"]
    
    else:
        logger.error(f"Unknown model type: {model_type}")
        return False
    
    missing_files = []
    for file in required_files:
        if not (path / file).exists():
            missing_files.append(file)
    
    if missing_files:
        logger.error(f"Missing required files for {model_type} model: {missing_files}")
        return False
    
    logger.info(f"{model_type.upper()} model verification passed")
    return True

def main():
    logger.info("=" * 50)
    logger.info("DOWNLOADING DYG-RAG AUXILIARY MODELS")
    logger.info("=" * 50)
    
    script_dir = get_script_directory()
    success_count = 0
    total_count = 2
    
    # Download Cross-Encoder model
    logger.info("=" * 50)
    logger.info("DOWNLOADING CROSS-ENCODER MODEL")
    logger.info("=" * 50)
    
    if download_cross_encoder_model():
        success_count += 1
        
        # Verify Cross-Encoder model
        cross_encoder_local_path = script_dir / "cross-encoder_ms-marco-TinyBERT-L-2-v2"
        if verify_model_download("cross_encoder", str(cross_encoder_local_path)):
            logger.info("Cross-Encoder model verification: PASSED")
        else:
            logger.warning("Cross-Encoder model verification: FAILED")
    
    # Download NER model
    logger.info("=" * 50)
    logger.info("DOWNLOADING NER MODEL")
    logger.info("=" * 50)
    
    if download_ner_model():
        success_count += 1
        
        # Verify NER model
        ner_local_path = script_dir / "dslim_bert_base_ner"
        if verify_model_download("ner", str(ner_local_path)):
            logger.info("NER model verification: PASSED")
        else:
            logger.warning("NER model verification: FAILED")
    
    # Summary
    logger.info("=" * 50)
    logger.info("DOWNLOAD SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Successfully downloaded: {success_count}/{total_count} models")
    
    if success_count == total_count:
        logger.info("All models downloaded successfully!")
        logger.info("\nNext steps:")
        logger.info(f"1. Cross-Encoder model saved to: {script_dir / 'cross-encoder_ms-marco-TinyBERT-L-2-v2'}")
        logger.info(f"2. NER model saved to: {script_dir / 'dslim_bert_base_ner'}")
        logger.info("3. DyG-RAG will automatically use these local models")
        return 0
    else:
        logger.error("Some models failed to download. Please check the error messages above.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 