import re
import json
import asyncio
import tiktoken
import datetime
import time
from dateutil import parser as date_parser
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum

from typing import Union, Optional, List, Dict
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import numpy as np

from ._splitter import SeparatorSplitter
from ._utils import (
    logger,
    clean_str,
    compute_mdhash_id,
    compute_args_hash,
    decode_tokens_by_tiktoken,
    encode_string_by_tiktoken,
    is_float_regex,
    list_of_list_to_csv,
    pack_user_ass_to_openai_messages,
    split_string_by_multi_markers,
    truncate_list_by_token_size,
)
from .base import (
    BaseGraphStorage,
    BaseVectorStorage,
    TextChunkSchema,
)
from .prompt import GRAPH_FIELD_SEP, PROMPTS
import bisect  # TODO: might not need this everywhere, check usage later

@dataclass
class EventRelationshipConfig:
    entity_factor: float = 0.2
    entity_ratio: float = 0.6
    time_ratio: float = 0.4
    max_links: int = 3
    time_factor: float = 1.0
    decay_rate: float = 0.01
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'EventRelationshipConfig':
        """Factory method to create config from legacy dict format"""
        return cls(
            entity_factor=config_dict.get("ent_factor", 0.2),
            entity_ratio=config_dict.get("ent_ratio", 0.6),
            time_ratio=config_dict.get("time_ratio", 0.4),
            max_links=config_dict.get("max_links", 3),
            time_factor=config_dict.get("time_factor", 1.0),
            decay_rate=config_dict.get("decay_rate", 0.01)
        )


@dataclass 
class ExtractionConfig:
    """Configuration for event extraction pipeline"""
    model_path: str = "./models"
    ner_model_name: str = "dslim_bert_base_ner"
    ner_device: str = "cuda:0"
    ner_batch_size: int = 32
    event_extract_max_gleaning: int = 3
    enable_timestamp_encoding: bool = False
    if_wri_ents: bool = False
    
    # Relationship computation settings
    event_relationship_batch_size: int = 100
    event_relationship_max_workers: Optional[int] = None
    
    @property
    def ner_model_full_path(self) -> str:
        return str(Path(self.model_path) / self.ner_model_name)


# === Strategy Pattern for Time Weight Calculation ===

class TimeWeightStrategy(ABC):
    """Abstract base class for time weight calculation strategies"""
    
    @abstractmethod
    def calculate_weight(self, days_difference: Optional[int]) -> float:
        """Calculate weight based on time difference"""
        pass


class ExponentialDecayTimeWeight(TimeWeightStrategy):
    """Exponential decay time weight - closer events get higher weight"""
    
    def __init__(self, max_weight: float = 1.0, decay_factor: float = 0.01):
        self.max_weight = max_weight
        self.decay_factor = decay_factor
    
    def calculate_weight(self, days_difference: Optional[int]) -> float:
        if days_difference is None:
            return 0.0
        
        abs_diff = abs(days_difference)
        weight = self.max_weight * math.exp(-self.decay_factor * abs_diff)
        return weight


# Global strategy instances - using dependency injection pattern for weight calculation
_time_weight_calculator = ExponentialDecayTimeWeight()


class NERExtractorFactory:
    
    @staticmethod
    def create_batch_extractor(config: ExtractionConfig) -> 'BatchNERExtractor':
        """Create a batch NER extractor from configuration"""
        return BatchNERExtractor(
            model_path=config.ner_model_full_path,
            device=config.ner_device,
            batch_size=config.ner_batch_size
        )

def monitor_performance(func):
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            elapsed = time.time() - start_time
            logger.info(f"Performance: {func.__name__} took {elapsed:.4f}s")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"Error in {func.__name__}: {e}, took {elapsed:.4f}s")
            raise
    return wrapper

def normalize_timestamp(timestamp_str: str) -> str:
    if not timestamp_str or timestamp_str.lower() == "static":
        return "static"
        
    # Easy case - already looks like ISO format
    iso_pattern = r"^\d{4}(-\d{2}(-\d{2})?)?$"
    if re.match(iso_pattern, timestamp_str):
        return timestamp_str
    
    try:
        dt = date_parser.parse(timestamp_str, fuzzy=True)
        
        # FIXME: this month detection logic is kinda hacky
        months = ["january", "february", "march", "april", "may", "june", 
                 "july", "august", "september", "october", "november", "december"]
        
        if "day" in timestamp_str.lower() or any(m in timestamp_str.lower() for m in months):
            return dt.strftime("%Y-%m-%d")
        elif "month" in timestamp_str.lower() or any(m in timestamp_str for m in ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]):
            return dt.strftime("%Y-%m")
        else:
            return dt.strftime("%Y")
            
    except:
        # Last ditch effort - just grab a year if we can find one
        year_pattern = r"(?:in\s+)?(\d{4})"
        year_match = re.search(year_pattern, timestamp_str)
        if year_match:
            return year_match.group(1)
        
        return "static"

def calculate_time_distance(timestamp1: str, timestamp2: str) -> Optional[int]:
    """Calculate days between two timestamps. Returns None if either is static."""
    if timestamp1 == "static" or timestamp2 == "static":
        return None
    
    def standardize(ts: str) -> datetime.datetime:
        # Handle different timestamp formats we might encounter
        if len(ts) == 4:  # Just year
            return datetime.datetime(int(ts), 1, 1)
        elif len(ts) == 7:  # Year-month
            year, month = ts.split('-')
            return datetime.datetime(int(year), int(month), 1)
        elif len(ts) == 10:  # Full date
            year, month, day = ts.split('-')
            return datetime.datetime(int(year), int(month), int(day))
        else:
            return date_parser.parse(ts)
    
    try:
        dt1 = standardize(timestamp1)
        dt2 = standardize(timestamp2)
        delta = dt2 - dt1
        return delta.days
    except:
        # TODO: maybe log what failed to parse?
        return None

def calculate_time_weight(days_difference: Optional[int], max_weight: float = 1.0, 
                          decay_factor: float = 0.01) -> float:
    # Create strategy with custom parameters if provided
    if max_weight != 1.0 or decay_factor != 0.01:
        strategy = ExponentialDecayTimeWeight(max_weight, decay_factor)
        return strategy.calculate_weight(days_difference)
    
    # Use global default strategy
    return _time_weight_calculator.calculate_weight(days_difference)

def compute_event_relationships_batch(event_batch_data: tuple) -> List[tuple]:
    current_events, all_events_data, config_params = event_batch_data
    
    # Handle both new config objects and legacy dict format
    if isinstance(config_params, EventRelationshipConfig):
        config = config_params
    else:
        # Legacy support - convert dict to config object
        config = EventRelationshipConfig.from_dict(config_params)
    
    relationships = []
    
    for current_event_id, current_event_data in current_events.items():
        current_timestamp = current_event_data.get("timestamp", "static")
        current_entities = set(current_event_data.get("entities_involved", []))
        
        if current_timestamp == "static" or not current_entities:
            continue
        
        entity_to_events = defaultdict(set)
        valid_events = {}
        
        for other_id, other_data in all_events_data.items():
            if other_id == current_event_id:
                continue
                
            other_timestamp = other_data.get("timestamp", "static")
            if other_timestamp == "static":
                continue
                
            other_entities = other_data.get("entities_involved", [])
            if not other_entities:
                continue
            
            valid_events[other_id] = (other_data, other_timestamp, other_entities)
            
            for entity in other_entities:
                entity_to_events[entity].add(other_id)
        
        candidate_event_ids = set()
        for entity in current_entities:
            candidate_event_ids.update(entity_to_events.get(entity, set()))
        
        candidate_relationships = []
        for other_id in candidate_event_ids:
            other_data, other_timestamp, other_entities = valid_events[other_id]
            
            common_entities = current_entities.intersection(other_entities)
            if not common_entities:
                continue
            
            time_distance = calculate_time_distance(current_timestamp, other_timestamp)
            if time_distance is None:
                continue
            
            abs_time_distance = abs(time_distance)
            
            entity_weight = min(1.0, config.entity_factor * len(common_entities))
            
            time_weight = calculate_time_weight(
                abs_time_distance,
                max_weight=config.time_factor,
                decay_factor=config.decay_rate
            )
            
            combined_score = config.entity_ratio * entity_weight + config.time_ratio * time_weight
            
            candidate_relationships.append((
                other_id,
                list(common_entities),
                abs_time_distance,
                entity_weight,
                time_weight,
                combined_score
            ))
        
        candidate_relationships.sort(key=lambda x: x[5], reverse=True)  # x[5] is combined_score
        selected_relationships = candidate_relationships[:config.max_links]
        
        for other_id, common_entities, abs_time_distance, entity_weight, time_weight, combined_score in selected_relationships:
            edge_data = {
                "relation_type": "event_temporal_proximity",
                "weight": combined_score,
                "time_distance": abs_time_distance,
                "shared_entities": ",".join(common_entities),
                "description": f"Events share {len(common_entities)} entities and are {abs_time_distance} days apart: {', '.join(common_entities)}",
                "source_id": current_event_data.get("source_id", ""),
                "is_undirected": True
            }
            
            relationships.append((current_event_id, other_id, edge_data))
            relationships.append((other_id, current_event_id, edge_data))
    
    return relationships



def chunking_by_token_size(
    tokens_list: list[list[int]],
    doc_keys,
    tiktoken_model,
    overlap_token_size=64,
    max_token_size=1200,
):
    
    results = []
    for index, tokens in enumerate(tokens_list):
        chunk_token = []
        lengths = []
        
        # Sliding window approach with overlap
        for start in range(0, len(tokens), max_token_size - overlap_token_size):
            chunk_token.append(tokens[start : start + max_token_size])
            lengths.append(min(max_token_size, len(tokens) - start))

        # FIXME: This nested list structure is getting confusing
        # tokens -> list[list[list[int]]] for corpus(doc(chunk))
        chunk_token = tiktoken_model.decode_batch(chunk_token)
        
        for i, chunk in enumerate(chunk_token):
            results.append({
                "tokens": lengths[i],
                "content": chunk.strip(),
                "chunk_order_index": i,
                "full_doc_id": doc_keys[index],
            })

    return results

def chunking_by_seperators(  # Yeah, I know it's "separators" but keeping it for consistency
    tokens_list: list[list[int]],
    doc_keys,
    tiktoken_model,
    overlap_token_size=128,
    max_token_size=1024,
):
    
    splitter = SeparatorSplitter(
        separators=[
            tiktoken_model.encode(s) for s in PROMPTS["default_text_separator"]
        ],
        chunk_size=max_token_size,
        chunk_overlap=overlap_token_size,
    )
    
    results = []
    for index, tokens in enumerate(tokens_list):
        chunk_token = splitter.split_tokens(tokens)
        lengths = [len(c) for c in chunk_token]

        # Same nested structure issue as above
        chunk_token = tiktoken_model.decode_batch(chunk_token)
        
        for i, chunk in enumerate(chunk_token):
            results.append({
                "tokens": lengths[i],
                "content": chunk.strip(),
                "chunk_order_index": i,
                "full_doc_id": doc_keys[index],
            })

    return results

def get_chunks(new_docs, chunk_func=chunking_by_token_size, **chunk_func_params):
    """
    Convert documents into chunks for processing.
    Tries to extract reasonable titles from the first line.
    """
    inserting_chunks = {}

    new_docs_list = list(new_docs.items())
    docs = [new_doc[1]["content"] for new_doc in new_docs_list]
    doc_keys = [new_doc[0] for new_doc in new_docs_list]
    
    # Extract document titles - first line usually works
    doc_titles = []
    for doc in docs:
        title = doc.split('\n')[0].strip()
        
        # If first line is too long, try first sentence
        if len(title) > 100:
            sentences = doc.split('.')
            title = sentences[0].strip()[:100] + '...' if len(sentences[0]) > 100 else sentences[0].strip()
            
        if not title:
            title = "Untitled Document"  # Fallback
            
        doc_titles.append(title)

    # Use OpenAI's tokenizer - seems to work well enough
    ENCODER = tiktoken.get_encoding("cl100k_base")
    tokens = ENCODER.encode_batch(docs, num_threads=16)  # TODO: make threads configurable
    
    chunks = chunk_func(
        tokens, doc_keys=doc_keys, tiktoken_model=ENCODER, **chunk_func_params
    )

    # Add titles back to chunks and create hash IDs
    for i, chunk in enumerate(chunks):
        doc_index = chunk["full_doc_id"]
        original_doc_index = doc_keys.index(doc_index)
        chunk["doc_title"] = doc_titles[original_doc_index]
        
        inserting_chunks.update(
            {compute_mdhash_id(chunk["content"], prefix="chunk-"): chunk}
        )

    return inserting_chunks

@monitor_performance
async def extract_events(
    chunks: dict[str, TextChunkSchema],
    dyg_inst: BaseGraphStorage,  # Dynamic graph storage
    events_vdb: BaseVectorStorage,
    global_config: dict,
    using_amazon_bedrock: bool=False,
) -> Union[tuple[BaseGraphStorage, dict], tuple[None, dict]]:

    extraction_start_time = time.time()
    
    # Convert legacy config to modern config objects
    config = ExtractionConfig(
        model_path=global_config.get("model_path", "./models"),
        ner_model_name=global_config.get("ner_model_name", "dslim_bert_base_ner"),
        ner_device=global_config.get("ner_device", "cuda:0"),
        ner_batch_size=global_config.get("ner_batch_size", 32),
        event_extract_max_gleaning=global_config.get("event_extract_max_gleaning", 3),
        enable_timestamp_encoding=global_config.get("enable_timestamp_encoding", False),
        if_wri_ents=global_config.get("if_wri_ents", False),
        event_relationship_batch_size=global_config.get("event_relationship_batch_size", 100),
        event_relationship_max_workers=global_config.get("event_relationship_max_workers", None)
    )
    
    phase_times = {
        "event_extraction": 0,
        "ner_extraction": 0,
        "event_merging": 0,
        "relationship_computation": 0,
        "events_vdb_update": 0,
    }
    
    # Debug file setup if requested
    if config.if_wri_ents:
        try:
            with open('debug.txt', 'w', encoding='utf-8') as f:
                f.write(f"=== DEBUGGING LOG STARTED AT {datetime.datetime.now()} ===\n")
                f.write(f"Processing {len(chunks)} chunks\n\n")
        except Exception as e:
            logger.error(f"Failed to initialize debug file: {e}")
    
    use_llm_func: callable = global_config["best_model_func"]
    
    # Use factory pattern for NER extractor creation
    try:
        ner_extractor = NERExtractorFactory.create_batch_extractor(config)
        logger.info(f"NER extractor initialized from: {config.ner_model_full_path}")
    except Exception as e:
        logger.error(f"Failed to initialize NER extractor: {e}")
        return None, {"failed": True, "error": "NER initialization failed"}

    ordered_chunks = list(chunks.items())

    event_extract_prompt = PROMPTS["dynamic_event_units"]
    event_extract_continue_prompt = PROMPTS["event_continue_extraction"]
    event_extract_if_loop_prompt = PROMPTS["event_if_loop_extraction"]

    already_processed = 0
    already_events = 0
    failed_chunks = 0

    async def _process_events_only(chunk_key_dp: tuple[str, TextChunkSchema]):
        nonlocal already_processed, already_events
        
        try:
            chunk_key = chunk_key_dp[0]
            chunk_dp = chunk_key_dp[1]
            doc_title = chunk_dp.get("doc_title", "")
            content = f"Title: {doc_title}\n\n{chunk_dp['content']}" if doc_title else chunk_dp["content"]
            
            maybe_events = defaultdict(list)
            
            event_hint_prompt = event_extract_prompt.replace("{input_text}", content)
            
            current_event_result = await use_llm_func(event_hint_prompt)
            if isinstance(current_event_result, list):
                current_event_result = current_event_result[0]["text"]
             
            if not current_event_result or not str(current_event_result).strip():
                logger.error(f"Empty response from LLM for chunk {chunk_key}")
                logger.error(f"Raw response: '{current_event_result}'")
                return {}
            
            event_history = pack_user_ass_to_openai_messages(event_hint_prompt, current_event_result, using_amazon_bedrock)
            combined_event_data = {"events": []} 

            try:
                parsed_data = json.loads(current_event_result)
                # logger.info(f"Successfully parsed JSON for chunk {chunk_key}")
                if isinstance(parsed_data, dict) and "events" in parsed_data:
                    combined_event_data["events"].extend(parsed_data["events"])
                    # logger.info(f"Found {len(parsed_data['events'])} events in initial response")
                else:
                    logger.warning(f"Parsed JSON does not contain 'events' key for chunk {chunk_key}")
                    logger.warning(f"Parsed data keys: {list(parsed_data.keys()) if isinstance(parsed_data, dict) else 'not a dict'}")
            except json.JSONDecodeError as e:
                logger.error(f"Initial event JSON parsing error for chunk {chunk_key}: {e}")
                logger.error(f"Failed to parse: '{current_event_result}'")
                logger.error(f"Error details: line {e.lineno}, column {e.colno}, pos {e.pos}")
                
                try:
                    start_idx = current_event_result.find('{')
                    end_idx = current_event_result.rfind('}') + 1
                    # logger.info(f"Attempting JSON recovery: start_idx={start_idx}, end_idx={end_idx}")
                    
                    if start_idx >= 0 and end_idx > start_idx:
                        json_str = current_event_result[start_idx:end_idx]
                        # logger.info(f"Extracted JSON substring: '{json_str[:200]}...'")
                        parsed_data = json.loads(json_str)
                        if isinstance(parsed_data, dict) and "events" in parsed_data:
                            combined_event_data["events"].extend(parsed_data["events"])
                            # logger.info(f"Recovery successful: found {len(parsed_data['events'])} events")
                        else:
                            logger.warning(f"Recovery failed: no 'events' key in recovered JSON")
                    else:
                        logger.error(f"Could not find JSON brackets in response: start_idx={start_idx}, end_idx={end_idx}")
                except Exception as inner_e:
                    logger.error(f"Failed to recover JSON for chunk {chunk_key}: {inner_e}")
                    logger.error(f"Recovery attempt failed on: '{json_str[:200] if 'json_str' in locals() else 'N/A'}...'")

            # Gleaning process
            for now_glean_index in range(config.event_extract_max_gleaning):
                # logger.info(f"Starting gleaning iteration {now_glean_index + 1}/{config.event_extract_max_gleaning} for chunk {chunk_key}")
                
                glean_event_result = await use_llm_func(event_extract_continue_prompt, history_messages=event_history)
                if isinstance(glean_event_result, list):
                    glean_event_result = glean_event_result[0]["text"]
                
                event_history += pack_user_ass_to_openai_messages(event_extract_continue_prompt, glean_event_result, using_amazon_bedrock)
                
                try:
                    gleaned_data = json.loads(glean_event_result)
                    if isinstance(gleaned_data, dict) and "events" in gleaned_data:
                        combined_event_data["events"].extend(gleaned_data["events"])
                        # logger.info(f"Gleaning iteration {now_glean_index + 1}: found {len(gleaned_data['events'])} additional events")
                    else:
                        logger.warning(f"Gleaning iteration {now_glean_index + 1}: no 'events' key in response")
                except json.JSONDecodeError as e:
                    logger.error(f"Gleaning event JSON parsing error for chunk {chunk_key}, iteration {now_glean_index + 1}: {e}")
                    logger.error(f"Failed to parse gleaning response: '{glean_event_result}'")

                if now_glean_index == config.event_extract_max_gleaning - 1:
                    break

                if_loop_event_result = await use_llm_func(event_extract_if_loop_prompt, history_messages=event_history)
                if_loop_event_result = if_loop_event_result.strip().strip('"').strip("'").lower()
                # logger.info(f"Continue gleaning decision for chunk {chunk_key}: '{if_loop_event_result}'")
                if if_loop_event_result != "yes":
                    # logger.info(f"Stopping gleaning for chunk {chunk_key} after {now_glean_index + 1} iterations")
                    break
            
            # Process event data
            logger.info(f"Processing {len(combined_event_data.get('events', []))} total events for chunk {chunk_key}")
            
            for event in combined_event_data.get("events", []):
                try:
                    if not isinstance(event, dict):
                        logger.warning(f"Skipping non-dict event in chunk {chunk_key}: {type(event)}")
                        continue
                        
                    sentence = event.get('sentence', '')
                    if not sentence or not isinstance(sentence, str):
                        logger.warning(f"Skipping event with invalid sentence in chunk {chunk_key}: '{sentence}'")
                        continue

                    context = event.get('context', '')
                    if context and not isinstance(context, str):
                        context = ''
                    
                    raw_time = event.get('time', 'static')
                    try:
                        normalized_time = normalize_timestamp(raw_time)
                    except Exception:
                        normalized_time = 'static'

                    event_id = compute_mdhash_id(f"{sentence}-{normalized_time}", prefix="event-")
 
                    event_obj = {
                        "event_id": event_id,
                        "timestamp": normalized_time,
                        "sentence": sentence,
                        "context": context,
                        "source_id": chunk_key,
                        "entities_involved": []  # Temporarily empty, will be filled later by NER
                    }
                    
                    if event_obj["sentence"]:  # Only add if sentence is not empty
                        maybe_events[event_id].append(event_obj)
                        already_events += 1
                        logger.debug(f"Added event {event_id} for chunk {chunk_key}: {sentence[:100]}...")
                        
                except Exception as event_err:
                    logger.error(f"Error processing individual event in chunk {chunk_key}: {event_err}")
                    logger.error(f"Problematic event data: {event}")
            
            logger.info(f"Successfully processed {len(maybe_events)} unique events for chunk {chunk_key}")
            
            already_processed += 1
            
            now_ticks = PROMPTS["process_tickers"][already_processed % len(PROMPTS["process_tickers"])]
            print(f"{now_ticks} Event extraction: {already_processed}({already_processed*100//len(ordered_chunks)}%) chunks, "
                  f"{already_events} events\r", end="", flush=True)
            
            return dict(maybe_events)
            
        except Exception as e:
            already_processed += 1
            logger.error(f"Failed to extract events from chunk {chunk_key_dp[0]}: {e}")
            return {}

    event_extraction_start = time.time()
    try:
        event_results = await asyncio.gather(
            *[_process_events_only(c) for c in ordered_chunks],
            return_exceptions=True
        )
        logger.info(f"\nEvent extraction completed, processing {len(event_results)} results")
        
        # Merge all event results
        all_maybe_events = defaultdict(list)
        for result in event_results:
            if isinstance(result, Exception):
                logger.error(f"Event extraction task failed: {result}")
                continue
                
            for k, v in result.items():
                all_maybe_events[k].extend(v)
                
        logger.info(f"Event extraction complete: {len(all_maybe_events)} unique events")
        
    except Exception as e:
        logger.error(f"Error during event extraction phase: {e}")
        return None, {"failed": True, "phase": "event_extraction"}
    
    phase_times["event_extraction"] = time.time() - event_extraction_start
    
    ner_extraction_start = time.time()
    logger.info("=== NER ENTITY EXTRACTION PHASE ===")
    try:
        all_maybe_events = ner_extractor.extract_entities_from_events(all_maybe_events)
        logger.info("NER entity extraction completed")
        
        # Build entity node data from entities extracted by NER
        all_entities = set()
        for event_id, event_list in all_maybe_events.items():
            for event_obj in event_list:
                entities = event_obj.get("entities_involved", [])
                all_entities.update(entities)
        
        logger.info(f"NER extraction complete: {len(all_entities)} unique entities extracted")
        
    except Exception as e:
        logger.error(f"Error during NER extraction: {e}")
        return None, {"failed": True, "phase": "ner_extraction"}
    
    phase_times["ner_extraction"] = time.time() - ner_extraction_start
    
    event_merging_start = time.time()
    maybe_events = all_maybe_events
    all_events_data = []
    for k, v in maybe_events.items():
        event_data = await _merge_events_then_upsert(k, v, dyg_inst, global_config)
        all_events_data.append(event_data)
    
    phase_times["event_merging"] = time.time() - event_merging_start
    
    relationship_computation_start = time.time()
    if len(maybe_events) > 1:
        # logger.info(f"Starting multiprocess event relationship processing for {len(maybe_events)} events")
        
        try:
            await batch_process_event_relationships_multiprocess(
                dyg_inst,
                global_config,
                batch_size=config.event_relationship_batch_size,
                max_workers=config.event_relationship_max_workers
            )
            # logger.info("Multiprocess event relationship processing completed successfully")
        except Exception as e:
            logger.error(f"Error in multiprocess event relationship processing: {e}")
            logger.warning("Falling back to single-threaded processing if needed")
    else:
        logger.info("Not enough events for multiprocess relationship processing")
    
    phase_times["relationship_computation"] = time.time() - relationship_computation_start
    
    if not len(all_events_data):
        logger.warning("No events found, maybe your LLM is not working")
        return None, {}
        
    events_vdb_update_start = time.time()
    if events_vdb is not None and len(all_events_data) > 0:
        events_for_vdb = {}
        for dp in all_events_data:
            event_content_for_vdb = dp["sentence"]
            if dp["timestamp"] != "static":
                event_content_for_vdb += f" (Time: {dp['timestamp']})"
            
            events_for_vdb[dp["event_id"]] = {
                "content": event_content_for_vdb,
                "event_id": dp["event_id"],
                "timestamp": dp["timestamp"],
                "sentence": dp["sentence"],
                "context": dp.get("context", ""),
                "source_id": dp.get("source_id", "")
            }
        
        try:
            if config.enable_timestamp_encoding:
                logger.info(f"Using timestamp-enhanced vector storage for events")
                for event_id, event_data in events_for_vdb.items():
                    if event_data["timestamp"] == "":
                        event_data["timestamp"] = "static"
            
            await events_vdb.upsert(events_for_vdb)
            logger.info(f"Updated events vector database with {len(events_for_vdb)} events")
        except Exception as e:
            if "out of memory" in str(e).lower():
                logger.error(f"CUDA OOM during events_vdb.upsert: {e}", exc_info=True)
            else:
                logger.error(f"Error during events_vdb.upsert: {e}", exc_info=True)
            logger.warning("Failed to update events vector database, but continuing")
    
    phase_times["events_vdb_update"] = time.time() - events_vdb_update_start
    
    total_chunks = already_processed
    success_rate = 100.0 if failed_chunks == 0 else ((already_processed - failed_chunks) / already_processed * 100)
    
    logger.info(f"Processing completion statistics:")
    logger.info(f"Total processed chunks: {already_processed}")
    logger.info(f"Failed chunks: {failed_chunks}")
    logger.info(f"Success rate: {success_rate:.2f}%")
    logger.info(f"Extracted events: {already_events} (before deduplication)")
    logger.info(f"NER extracted entities: {len(all_entities)}")
    logger.info(f"Final unique events: {len(maybe_events)}")
    
    total_extraction_time = time.time() - extraction_start_time
    
    stats = {
        "total_chunks": already_processed,
        "failed_chunks": failed_chunks, 
        "success_rate": success_rate,
        "raw_events": already_events,
        "ner_extracted_entities": len(all_entities),
        "unique_events": len(maybe_events),
        "extraction_mode": "event_first_ner",
        "phase_times": phase_times,
        "total_extraction_time": total_extraction_time
    }
    
    logger.info("=== DyG Construction Phase Time Statistics ===")
    logger.info(f"Event Extraction (LLM): {phase_times['event_extraction']:.2f}s")
    logger.info(f"NER Entity Extraction: {phase_times['ner_extraction']:.2f}s")
    logger.info(f"Event Node Merging: {phase_times['event_merging']:.2f}s")
    logger.info(f"Relationship Computation: {phase_times['relationship_computation']:.2f}s")
    logger.info(f"Vector Database Update: {phase_times['events_vdb_update']:.2f}s")
    logger.info(f"Total Time: {total_extraction_time:.2f}s")
    
    if config.if_wri_ents:
        try:
            import datetime
            with open('debug.txt', 'a', encoding='utf-8') as f:
                f.write(f"\n=== EXTRACTED EVENTS DEBUG INFO ({datetime.datetime.now()}) ===\n")
                f.write(f"Total extracted events: {len(all_events_data)}\n\n")
                
                for i, event_data in enumerate(all_events_data, 1):
                    f.write(f"Event #{i}: {event_data.get('event_id', 'unknown_id')}\n")
                    f.write(f"  Timestamp: {event_data.get('timestamp', 'static')}\n")
                    f.write(f"  Sentence: {event_data.get('sentence', '')}\n")
                    f.write(f"  Context: {event_data.get('context', '')}\n")
                    f.write(f"  Entities Involved: {event_data.get('entities_involved', [])}\n")
                    f.write(f"  Source ID: {event_data.get('source_id', '')}\n")
                    f.write("-" * 80 + "\n")
                
                f.write(f"\n=== END OF EVENTS DEBUG INFO ===\n\n")
            
            logger.info(f"Debug information written to debug.txt for {len(all_events_data)} events")
        except Exception as e:
            logger.error(f"Failed to write debug information: {e}")
    
    return dyg_inst, stats

async def _merge_events_then_upsert(
    event_id: str,
    events_data: list[dict],
    dyg_inst: BaseGraphStorage,
    global_config: dict,
):
    already_timestamps = []
    already_sentences = []
    already_contexts = []
    already_source_ids = []
    already_entities_involved = []

    already_event = await dyg_inst.get_node(event_id)
    if already_event is not None:
        already_timestamps.append(already_event.get("timestamp", ""))
        already_sentences.append(already_event.get("sentence", ""))
        already_contexts.append(already_event.get("context", ""))
        already_source_ids.extend(
            split_string_by_multi_markers(already_event.get("source_id", ""), [GRAPH_FIELD_SEP])
        )
        existing_entities = already_event.get("entities_involved", [])
        if isinstance(existing_entities, list):
            already_entities_involved.extend(existing_entities)
        elif isinstance(existing_entities, str):
            already_entities_involved.extend(existing_entities.split(",") if existing_entities else [])

    timestamps = [dp.get("timestamp", "") for dp in events_data] + already_timestamps
    timestamp = sorted(Counter(timestamps).items(), key=lambda x: x[1], reverse=True)[0][0] if timestamps else ""
    
    sentences = [dp.get("sentence", "") for dp in events_data] + already_sentences
    sentence = max(sentences, key=len) if sentences else ""
    
    contexts = [dp.get("context", "") for dp in events_data] + already_contexts
    context = max(contexts, key=len) if contexts else ""
    
    all_entities_involved = []
    for dp in events_data:
        entities = dp.get("entities_involved", [])
        if isinstance(entities, list):
            all_entities_involved.extend(entities)
        elif isinstance(entities, str) and entities:
            all_entities_involved.extend(entities.split(","))
    
    all_entities_involved.extend(already_entities_involved)
    entities_involved = list(set([e.strip() for e in all_entities_involved if e and e.strip()]))
        
    source_id = GRAPH_FIELD_SEP.join(
        set([dp.get("source_id", "") for dp in events_data] + already_source_ids)
    )
    
    event_data = dict(
        timestamp=timestamp,
        sentence=sentence,
        context=context,
        source_id=source_id,
        entities_involved=entities_involved,
        participants="",
        location="",
    )
    
    await dyg_inst.upsert_node(
        event_id,
        node_data=event_data,
    )
    
    event_data["event_id"] = event_id
    return event_data

@monitor_performance
async def _merge_event_relations_then_upsert(
    event_id: str,
    events_data: list[dict],
    dyg_inst: BaseGraphStorage,
    global_config: dict,
):
    event_data = events_data[0]
    
    try:
        existing_event = await dyg_inst.get_node(event_id)
        if existing_event is None:
            await dyg_inst.upsert_node(event_id, node_data=event_data)
            logger.info(f"Created missing event node: {event_id}")
    except Exception as e:
        logger.error(f"Error checking event node {event_id}: {e}")
        return False
    
    return True

@monitor_performance
async def batch_process_event_relationships_multiprocess(
    dyg_inst: BaseGraphStorage,
    global_config: dict,
    batch_size: int = 100,
    max_workers: int = None
):
    all_events = await dyg_inst.get_all_nodes()
    
    valid_events = {}
    for event_id, event_data in all_events.items():
        timestamp = event_data.get("timestamp", "static")
        entities = event_data.get("entities_involved", [])
        if timestamp != "static" and entities:
            valid_events[event_id] = event_data
    
    if not valid_events:
        logger.info("No valid events found for relationship processing")
        return
    
    logger.info(f"Processing {len(valid_events)} valid events for relationships")
    
    # Prepare configuration parameters
    config_params = {
        "ent_factor": global_config.get("ent_factor", 0.2),
        "ent_ratio": global_config.get("ent_ratio", 0.6),
        "time_ratio": global_config.get("time_ratio", 0.4),
        "max_links": global_config.get("max_links", 3),
        "time_factor": global_config.get("time_factor", 1.0),
        "decay_rate": global_config.get("decay_rate", 0.01)
    }
    
    # Batch events for processing
    event_ids = list(valid_events.keys())
    batches = []
    
    for i in range(0, len(event_ids), batch_size):
        batch_events = {eid: valid_events[eid] for eid in event_ids[i:i+batch_size]}
        batches.append((batch_events, valid_events, config_params))
    
    logger.info(f"Created {len(batches)} batches for multiprocess processing")
    
    if max_workers is None:
        max_workers = min(mp.cpu_count(), len(batches))
    
    all_relationships = []
    
    loop = asyncio.get_event_loop()
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        logger.info(f"Starting multiprocess computation with {max_workers} workers")
        
        futures = [
            loop.run_in_executor(executor, compute_event_relationships_batch, batch_data)
            for batch_data in batches
        ]
        
        batch_results = await asyncio.gather(*futures, return_exceptions=True)
        
        for i, result in enumerate(batch_results):
            if isinstance(result, Exception):
                logger.error(f"Batch {i} failed: {result}")
            else:
                all_relationships.extend(result)
    
    logger.info(f"Computed {len(all_relationships)} relationships, now updating graph")
    
    edge_updates = []
    for src_id, tgt_id, edge_data in all_relationships:
        edge_updates.append((src_id, tgt_id, edge_data))
    
    write_batch_size = 1000
    total_updates = len(edge_updates)
    
    for i in range(0, total_updates, write_batch_size):
        batch_updates = edge_updates[i:i+write_batch_size]
        
        update_tasks = [
            dyg_inst.upsert_edge(src_id, tgt_id, edge_data=edge_data)
            for src_id, tgt_id, edge_data in batch_updates
        ]
        
        await asyncio.gather(*update_tasks, return_exceptions=True)
        
        progress = min(i + write_batch_size, total_updates)
        logger.info(f"Updated {progress}/{total_updates} edges ({progress*100//total_updates}%)")
    
    logger.info(f"Successfully processed all {total_updates} event relationships")


class BatchNERExtractor:
    """Batch NER entity extractor, using BERT model for efficient entity recognition"""
    
    def __init__(self, model_path: str, device: str = "cuda:0", batch_size: int = 32):
        """
        Initialize NER extractor
        
        Args:
            model_path: Path to the NER model (required, no default)
            device: Computing device
            batch_size: Batch size for processing
        """
        self.model_path = model_path
        self.device = device
        self.batch_size = batch_size
        self.tokenizer = None
        self.model = None
        self.ner_pipeline = None
        
        # Standard NER label mapping - BERT uses BIO tagging
        self.label_mapping = {
            "B-PER": "PERSON", "I-PER": "PERSON",
            "B-ORG": "ORGANIZATION", "I-ORG": "ORGANIZATION", 
            "B-LOC": "LOCATION", "I-LOC": "LOCATION",
            "B-MISC": "MISCELLANEOUS", "I-MISC": "MISCELLANEOUS"
        }
        
        self._load_model()
    
    def _load_model(self):
        """Load the NER model and set up the pipeline."""
        try:
            logger.info(f"Loading NER model from {self.model_path} on {self.device}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForTokenClassification.from_pretrained(self.model_path)
            
            # Move model to specified device
            self.model.to(self.device)
            self.model.eval()
            
            # Fix device mapping for pipeline
            if self.device.startswith("cuda"):
                # Extract device number from "cuda:0", "cuda:1", etc.
                device_num = int(self.device.split(":")[-1]) if ":" in self.device else 0
                pipeline_device = device_num
            else:
                pipeline_device = -1
            
            logger.info(f"Initializing NER pipeline with device mapping: {self.device} -> {pipeline_device}")
            
            self.ner_pipeline = pipeline(
                "ner",
                model=self.model,
                tokenizer=self.tokenizer,
                device=pipeline_device,
                aggregation_strategy="simple",
                batch_size=self.batch_size
            )
            
            logger.info("NER model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load NER model: {e}", exc_info=True)
            raise
    
    def extract_entities_batch(self, sentences: List[str]) -> List[List[str]]:
        if not sentences:
            return []
        
        try:
            valid_sentences = []
            sentence_indices = []
            for i, sentence in enumerate(sentences):
                if sentence and isinstance(sentence, str) and sentence.strip():
                    valid_sentences.append(sentence.strip())
                    sentence_indices.append(i)
            
            if not valid_sentences:
                return [[] for _ in sentences]
            
            logger.info(f"Processing {len(valid_sentences)} sentences with NER model")
            
            ner_results = self.ner_pipeline(valid_sentences)
            
            all_entities = [[] for _ in sentences]
            
            for idx, (sentence_idx, sentence_entities) in enumerate(zip(sentence_indices, ner_results)):
                logger.debug(f"Processing sentence {idx}: found {len(sentence_entities)} raw entities")
                entities = self._process_ner_result(sentence_entities)
                all_entities[sentence_idx] = entities
            
            total_extracted = sum(len(entities) for entities in all_entities)
            logger.info(f"Extracted {total_extracted} entities from {len(valid_sentences)} sentences")
            
            return all_entities
            
        except Exception as e:
            logger.error(f"Error during batch NER extraction: {e}", exc_info=True)
            return [[] for _ in sentences]
    
    def _process_ner_result(self, ner_result: List[Dict]) -> List[str]:
        entities = []
        
        try:
            for entity_info in ner_result:
                entity_text = entity_info.get('word', '').strip()
                entity_label = entity_info.get('entity_group', '')
                confidence = entity_info.get('score', 0.0)
                
                # Higher confidence threshold for better quality
                if confidence < 0.8 or len(entity_text) < 2:
                    continue
                
                entity_text = entity_text.replace('##', '').strip()
                if not entity_text:
                    continue
                
                entity_name = entity_text.upper()
                
                # Avoid duplicates
                if entity_name not in entities:
                    entities.append(entity_name)
                    logger.debug(f"Added entity: '{entity_name}'")
                else:
                    logger.debug(f"Duplicate entity skipped: '{entity_name}'")
            
            # logger.info(f"NER processing result: {len(entities)} entities extracted from {len(ner_result)} candidates")
            
        except Exception as e:
            logger.error(f"Error processing NER result: {e}")
        
        return entities
    
    def extract_entities_from_events(self, events_data: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:
        if not events_data:
            return events_data
        
        sentences = []
        event_mapping = []  # (event_id, event_index)
        
        for event_id, event_list in events_data.items():
            for idx, event_obj in enumerate(event_list):
                sentence = event_obj.get("sentence", "")
                if sentence:
                    sentences.append(sentence)
                    event_mapping.append((event_id, idx))
        
        if not sentences:
            logger.warning("No valid sentences found for NER extraction")
            return events_data
        
        logger.info(f"Extracting entities from {len(sentences)} event sentences")
        
        batch_entities = self.extract_entities_batch(sentences)
        
        # Map entities back to their events
        for (event_id, event_idx), entities in zip(event_mapping, batch_entities):
            if event_id in events_data and event_idx < len(events_data[event_id]):
                events_data[event_id][event_idx]["entities_involved"] = entities
        
        total_entities = sum(len(entities) for entities in batch_entities)
        logger.info(f"NER extraction completed: {total_entities} entities extracted")
        
        return events_data
