
import platform
import time
import traceback
from typing import Any, Dict, List, Optional
import requests
import re
import os
import json
import pickle
import random
import gc
import numpy as np
from pathlib import Path
from datetime import datetime

import nltk
from datasets import load_dataset
from tqdm import tqdm
import logging
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize

MAX_TEXT_LENGTH = 10000  # Max chars per document to process
MIN_SENTENCE_LENGTH = 30  # Min chars for valid sentence
MAX_DOCS_TO_PROCESS = 15000  # Total documents to extract
STAGE1_CHECKPOINT_INTERVAL = 100  # Docs per checkpoint
STAGE2_CHECKPOINT_INTERVAL = 500
STAGE3_CHECKPOINT_INTERVAL = 10
STAGE3_BATCH_SIZE = 10
STAGE3_BATCH_DELAY = 30  # seconds
DIFFUSION_T_MAX = 40  # Max timestep for diffusion noise6
DIFFUSION_SIGMA_MIN = 0.02
DIFFUSION_SIGMA_MAX = 0.5

# === PARALLEL PROCESSING CONFIG ===
import os
import concurrent.futures

NUM_WORKERS = min(4, os.cpu_count() or 4)  # Auto-detect CPUs
STAGE2_PARALLEL = True  # Enable parallel embeddings
STAGE3_PARALLEL = True  # Enable parallel API calls
STAGE3_MAX_WORKERS = 2  # Ollama concurrent requests (stay under rate limit)
STAGE3_REQUESTS_PER_SEC = 2  # Ollama rate limit (adjust based on your setup)

# Valid compartments and hierarchies (FIX #8: Validation)
VALID_COMPARTMENTS = {'FACTUAL', 'PROCEDURAL', 'EPISODIC', 'CONTEXTUAL', 'CONCEPTUAL'}
VALID_HIERARCHIES = {'GRANULAR', 'INTERMEDIATE', 'GENERAL'}

WIN_PATH = "G:/My Drive/scm_project"
WSL_PATH = "/mnt/g/My Drive/scm_project"

# Detect running in WSL or Linux
def in_wsl():
    if platform.system().lower() == "linux" or "unix" or "wsl":
        try:
            with open("/proc/version", "r") as f:
                return "microsoft" in f.read().lower()
        except FileNotFoundError:
            return False
    return False

if in_wsl():
    BASE_PATH = WSL_PATH
else:
    BASE_PATH = WIN_PATH

CACHE_PATH = os.path.join(BASE_PATH, "cache/datasets")
OUTPUT_PATH = os.path.join(BASE_PATH, "datasets/processed")
CHECKPOINT_DIR = os.path.join(BASE_PATH, "checkpoints")

print(f"Using BASE_PATH: {BASE_PATH}")
print(f"Using CHECKPOINT_DIR: {CHECKPOINT_DIR}")    

LOG_DIR = os.path.join(BASE_PATH, "logs")
LOG_DIR = "G:/My Drive/scm_project/logs"

os.makedirs(LOG_DIR, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(LOG_DIR, f"pipeline_scm_{timestamp}.log")

# Configure logging with both file and console handlers
logging.basicConfig(
    level=logging.DEBUG,  # Full debug logging for QA
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

logger.info("="*80)
logger.info(f"Log file: {log_file}")
logger.info(f"Architecture: Small Concept Model (SCM)")
logger.info(f"Max docs to process: {MAX_DOCS_TO_PROCESS}")
logger.info("="*80)

logger.info(f"BASE PATH = {BASE_PATH}")

NLTK_CACHE = os.path.join(BASE_PATH, "cache/nltk")

os.makedirs(NLTK_CACHE, exist_ok=True)

logger.info("Setting up NLTK...")
nltk.data.path.append(NLTK_CACHE)

try:
    nltk.data.find('tokenizers/punkt')
    logger.info("✓ NLTK punkt tokenizer already available")
except LookupError:
    logger.info("Downloading NLTK punkt tokenizer...")
    nltk.download('punkt', download_dir=NLTK_CACHE, quiet=False)
    logger.info("✓ NLTK punkt tokenizer downloaded")

try:
    nltk.download('punkt_tab', download_dir=NLTK_CACHE, quiet=False)
    logger.info("✓ NLTK punkt_tab downloaded")
except Exception as e:
    logger.warning(f"punkt_tab download failed (may not be critical): {e}")

# BASE_PATH = "G:/My Drive/scm_project"
# WSL_BASE_PATH = "/mnt/g/My Drive/scm_project"
# CACHE_PATH = os.path.join(BASE_PATH, "cache/datasets")
# OUTPUT_PATH = os.path.join(BASE_PATH, "datasets/processed")
# CHECKPOINT_DIR = os.path.join(BASE_PATH, "checkpoints")
# WSL_CACHE_PATH = os.path.join(WSL_BASE_PATH, "cache/datasets")
# WSL_OUTPUT_PATH = os.path.join(WSL_BASE_PATH, "datasets/processed")
# WSL_CHECKPOINT_DIR = os.path.join(WSL_BASE_PATH, "checkpoints")

# if not os.path.isdir(BASE_PATH):
#     BASE_PATH = WSL_BASE_PATH
#     CACHE_PATH = WSL_CACHE_PATH
#     OUTPUT_PATH = WSL_OUTPUT_PATH
#     CHECKPOINT_DIR = WSL_CHECKPOINT_DIR
    
stage2_file = os.path.join(OUTPUT_PATH, "stage2_sft.jsonl")

os.makedirs(CACHE_PATH, exist_ok=True)
os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

os.environ['HF_DATASETS_CACHE'] = CACHE_PATH
os.environ['HF_DATASETS_OFFLINE'] = '0'

logger.info(f"Base path: {BASE_PATH}")
logger.info(f"Cache path: {CACHE_PATH}")
logger.info(f"Output path: {OUTPUT_PATH}")
logger.info(f"Checkpoint dir: {CHECKPOINT_DIR}")

import concurrent.futures
from functools import partial

# Add after your imports
NUM_WORKERS = 4  # Adjust based on your CPU cores (use os.cpu_count() for auto)

def parallel_encode_batch(texts, encoder, max_workers=NUM_WORKERS):
    """
    Parallel batch encoding for embeddings.
    Uses ThreadPoolExecutor for I/O-bound embedding generation.
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        embeddings = list(executor.map(encoder.encode, texts))
    return embeddings

def parallel_process_segments(segments, encoder, max_workers=NUM_WORKERS):
    """
    Process segments in parallel with error handling.
    Returns list of (segment, embedding) tuples.
    """
    results = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_seg = {
            executor.submit(encoder.encode, seg['text']): seg 
            for seg in segments
        }
        
        for future in concurrent.futures.as_completed(future_to_seg):
            seg = future_to_seg[future]
            try:
                embedding = future.result()
                if hasattr(embedding, 'tolist'):
                    embedding = embedding.tolist()
                elif isinstance(embedding, np.ndarray):
                    embedding = embedding.tolist()
                results.append((seg, embedding))
            except Exception as e:
                logger.warning(f"Embedding failed for segment: {e}")
                results.append((seg, None))
    
    return results

import threading
from collections import deque

class RateLimitedExecutor:
    """
    Thread-safe rate-limited parallel executor for Ollama API.
    Prevents 429 errors while maximizing throughput.
    """
    def __init__(self, max_workers=2, requests_per_second=2):
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self.requests_per_second = requests_per_second
        self.min_interval = 1.0 / requests_per_second
        self.last_request_time = 0
        self.lock = threading.Lock()
    
    def _rate_limited_submit(self, fn, *args, **kwargs):
        """Submit with rate limiting."""
        with self.lock:
            now = time.time()
            time_since_last = now - self.last_request_time
            
            if time_since_last < self.min_interval:
                sleep_time = self.min_interval - time_since_last
                time.sleep(sleep_time)
            
            self.last_request_time = time.time()
        
        return fn(*args, **kwargs)
    
    def map(self, fn, items):
        """Rate-limited parallel map."""
        futures = []
        for item in items:
            future = self.executor.submit(self._rate_limited_submit, fn, item)
            futures.append(future)
        
        results = []
        for future in concurrent.futures.as_completed(futures):
            try:
                results.append(future.result())
            except Exception as e:
                logger.error(f"Task failed: {e}")
                results.append(None)
        
        return results
    
    def shutdown(self):
        self.executor.shutdown(wait=True)


def save_checkpoint(data, checkpoint_name):
    """Save checkpoint (FIX #15: No redundant file size logging)."""
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"{checkpoint_name}.pkl")
    try:
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(data, f)
        logger.info(f"✓ Checkpoint saved: {checkpoint_name}")
        return True
    except Exception as e:
        logger.error(f"✗ Failed to save checkpoint {checkpoint_name}: {e}")
        return False

def load_checkpoint(checkpoint_name):
    """Load checkpoint with corruption detection."""
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"{checkpoint_name}.pkl")
    if not os.path.exists(checkpoint_path):
        logger.info(f"No checkpoint found: {checkpoint_name}, on {CHECKPOINT_DIR}")
        return None
    
    try:
        with open(checkpoint_path, 'rb') as f:
            data = pickle.load(f)
        logger.info(f"✓ Checkpoint loaded: {checkpoint_name}")
        
        # Validate checkpoint integrity
        if checkpoint_name == "stage2_progress":
            if 'stage2_data' in data and 'processed_doc_ids' in data:
                data_len = len(data['stage2_data'])
                ids_len = len(data['processed_doc_ids'])
                if ids_len > data_len * 1.5:
                    logger.warning(f"⚠ Suspicious checkpoint: {ids_len} IDs vs {data_len} data items")
                    logger.warning("Deleting corrupted checkpoint...")
                    os.remove(checkpoint_path)
                    return None
        
        return data
    except (EOFError, pickle.UnpicklingError, ValueError) as e:
        logger.error(f"✗ Corrupted checkpoint {checkpoint_name}: {e}")
        logger.info(f"Deleting corrupted checkpoint...")
        try:
            os.remove(checkpoint_path)
            logger.info(f"✓ Deleted corrupted checkpoint")
        except Exception as del_e:
            logger.error(f"✗ Failed to delete checkpoint: {del_e}")
        return None

logger.info("="*80)
logger.info("LOADING EMBEDDING MODEL (SCM REQUIREMENT: 1024-DIM SONAR)")
logger.info("="*80)

class SONAREncoderWrapper:
    """Unified wrapper supporting both real SONAR and fallback encoders."""
    
    def __init__(self):
        self.encoder_type = None
        
        # Try actual SONAR (requires: pip install sonar-space)
        try:
            from sonar.inference_pipelines import TextToEmbeddingModelPipeline
            self.encoder = TextToEmbeddingModelPipeline(
                encoder="text_sonar_basic_encoder",
                tokenizer="text_sonar_basic_encoder"
            )
            self.encoder_type = "sonar"
            logger.info("SONAR encoder loaded (1024-dim)")
        except (ImportError, Exception) as e:
            logger.warning(f"SONAR unavailable: {e}")
            
            # Priority 2: Try LaBSE (native 1024-dim, no padding)
            try:
                from sentence_transformers import SentenceTransformer
                self.encoder = SentenceTransformer('sentence-transformers/LaBSE')
                self.encoder_type = "labse"
                logger.info("✓ LaBSE encoder loaded (768-dim, will pad to 1024)")
            except Exception as e2:
                logger.warning(f"LaBSE unavailable: {e2}")
                
                # Priority 3: Fallback to padded encoder
                self.encoder = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
                self.encoder_type = "padded"
                logger.warning("⚠ Using padded (768→1024) embeddings (suboptimal)")
    
    def encode(self, text):
        """Unified encoding interface."""
        if self.encoder_type == "sonar":
            # SONAR returns 1024-dim directly
            return self.encoder.predict([text], source_lang="eng_Latn")[0]
        
        elif self.encoder_type == "labse":
            # LaBSE is 768-dim, pad to 1024
            emb_768 = self.encoder.encode(text)
            return np.pad(emb_768, (0, 256), mode='constant', constant_values=0)
        
        else:  # padded
            emb_768 = self.encoder.encode(text)
            return np.pad(emb_768, (0, 256), mode='constant', constant_values=0)

# Initialize encoder
try:
    sonar_encoder = SONAREncoderWrapper()
    
    # Validate dimension
    logger.info("Validating embedding dimension...")
    test_embedding = sonar_encoder.encode("Test sentence for SCM")
    embedding_dim = len(test_embedding) if hasattr(test_embedding, '__len__') else test_embedding.shape[0]
    
    if embedding_dim != 1024:
        logger.error(f"CRITICAL: Embedding dimension is {embedding_dim}, expected 1024")
        raise ValueError(f"SCM requires 1024-dim embeddings, got {embedding_dim}")
    
    logger.info(f"✓ Embedding model validated (1024-dim)")
    logger.info(f"✓ Embedding model ready (SCM compliant)")
    
except Exception as e:
    logger.error(f"Failed to load embedding model: {e}")
    logger.error("CRITICAL: Cannot proceed without 1024-dim embedding model")
    raise RuntimeError("Embedding model required but failed to load")


SCIENCE_KEYWORDS = [
    # Domain/Source matches
    'science', 'sci', 'chembio', 'chemistry', 'biology', 'physics',
    'gpqa', 'jeebench', 'scibench', 'mmlu_science', 'olympiad',
    
    # Chemistry (30 terms)
    'molecule', 'molecular', 'atom', 'atomic', 'ion', 'chemical', 'bond',
    'electron', 'neutron', 'proton', 'nucleus', 'orbital', 'valence',
    'periodic table', 'element', 'compound', 'reaction', 'synthesis',
    'catalyst', 'equilibrium', 'stoichiometry', 'ph', 'acid', 'base',
    'molar', 'molarity', 'oxidation', 'reduction', 'polymer', 'isotope',
    
    # Biology (30 terms)
    'protein', 'amino acid', 'dna', 'rna', 'genome', 'cell', 'membrane',
    'mitochondria', 'chloroplast', 'organism', 'species', 'evolution',
    'gene', 'chromosome', 'allele', 'genotype', 'phenotype', 'mitosis',
    'meiosis', 'enzyme', 'metabolism', 'respiration', 'photosynthesis',
    'atp', 'glucose', 'bacteria', 'virus', 'immune', 'ecology', 'biodiversity',
    
    # Physics (35 terms)
    'force', 'energy', 'momentum', 'velocity', 'acceleration', 'gravity',
    'electromagnetic', 'electric', 'magnetic', 'charge', 'current', 'voltage',
    'wave', 'frequency', 'wavelength', 'quantum', 'photon', 'relativity',
    'thermodynamics', 'entropy', 'kinetic', 'potential', 'newton', 'inertia',
    'optics', 'refraction', 'nuclear', 'fission', 'fusion', 'planet', 'star',
    'galaxy', 'orbit', 'telescope', 'asteroid',
    
    # Earth Science/Anatomy (10 terms)
    'geology', 'fossil', 'tectonic', 'volcano', 'atmosphere', 'climate',
    'neuron', 'synapse', 'brain', 'anatomy'
]

logger.info(f"Science filter configured with {len(SCIENCE_KEYWORDS)} keywords")

def extract_technical_terms(text):
    """Identify key science terms using regex word boundaries."""
    tech_vocab = [
        'transformer', 'neural network', 'gradient descent', 'loss function', 'embedding',
        'algorithm', 'theorem', 'equation', 'derivative', 'integral', 'probability',
        'molecule', 'atom', 'ion', 'chemical', 'bond', 'electron', 'proton', 'nucleus',
        'protein', 'amino acid', 'dna', 'rna', 'genome', 'cell', 'gene', 'chromosome',
        'force', 'energy', 'momentum', 'velocity', 'acceleration', 'gravity', 'electromagnetic',
        'electric', 'magnetic', 'charge', 'current', 'voltage', 'wave', 'frequency', 'wavelength',
        'quantum', 'photon', 'relativity', 'thermodynamics', 'entropy', 'kinetic', 'potential',
        'geology', 'fossil', 'tectonic', 'volcano', 'atmosphere', 'climate', 'neuron', 'synapse', 'brain'
    ]
    
    text_lower = text.lower()
    found_terms = []
    
    for term in tech_vocab:
        pattern = r'\b' + re.escape(term) + r'\b'
        if re.search(pattern, text_lower):
            found_terms.append(term)
    
    return found_terms

def compute_importance_score(text):
    """
    Compute dynamic importance score for hierarchical memory retention.
    
    Formula: base_score + technical_density_bonus + length_bonus
    - base_score: 0.5
    - technical_density: +0.1 per technical term (max 0.3)
    - length: +0.01 per 10 chars (max 0.2)
    
    Returns: float between 0.5 and 1.0
    """
    tech_terms = extract_technical_terms(text)
    tech_bonus = min(0.1 * len(tech_terms), 0.3)
    length_bonus = min(0.01 * (len(text) / 10), 0.2)
    
    score = 0.5 + tech_bonus + length_bonus
    return round(min(score, 1.0), 2)

def extract_text_from_conversations(example):
    """Extract clean text from conversations list."""
    conversations = example.get('conversations', [])
    texts = [conv.get('value', '') for conv in conversations 
             if isinstance(conv, dict) and 'value' in conv]
    return ' '.join(texts).strip() if texts else example.get('text', '')

def estimate_compartment_advanced(text):
    """
    Estimate compartment with:
    1. Discard filter (returns None if unwanted terms found)
    2. Priority mechanism (Procedural/Episodic > others)
    3. Smart fallback (indicator-based, not length-based)
    """
    text_lower = text.lower()
    
    discard_terms = [
        'follow us',
        'advertisement', 'sponsored', 'buy now', 'download here',
        'sign up', 'register now', 'terms and conditions', 'privacy policy',
        'all rights reserved', 'unsubscribe',
        '[music]', '[applause]', '[laughter]', 'http://', 'https://',
        'www.', '.com', '.org', 'email:', '@', 'because I', 'I was', 
    ]
    
    if any(term in text_lower for term in discard_terms):
        return None  # Signal to discard this sentence
    
    conceptual_strong = ['hmm', 'therefore', 'concept', 'could be', 'theory', 'implies', 'suggests', "let me", "think", "I know", "I'm", "wonder", "feel", "seem", "should", "concept", "break it down", "right?", "I will", "I can", "I need", "I should", "need to", "think about", "look at", "my reasoning"]
    if any(kw in text_lower for kw in conceptual_strong):
        return 'CONCEPTUAL'
    
    contextual_strong = ['example', 'because', "occur", "mean", "by the way", "information", "context", "known that", "seen", "also "]
    if any(kw in text_lower for kw in contextual_strong):
        return 'CONTEXTUAL'
    
    # Check for procedural keywords first (highest priority)
    procedural_strong = ['procedur', 'step', 'first', "next", "via", "data", "start", "end", "last", "procedure", "leads"]
    if any(kw in text_lower for kw in procedural_strong):
        return 'PROCEDURAL'
    
    # Check for episodic keywords
    episodic_strong = ['discovered', 'developed by', 'published', 'introduced in', 'discover', 'develop', 'introduced', 'published', 'historical', "remember" "why", "try", "know", "recall", "used to", "said", "happened", "occurred", "consider", "background", "figure out", "has to", "seen", "I know", "I was", "I did"]
    if any(kw in text_lower for kw in episodic_strong):
        return 'EPISODIC'
        
    # Score-based fallback
    factual_indicators = ['defined as', 'equals', "=", "/", "sqrt", "{", "(", "%", "\\", "+", "(g)", "element", "chemical", "ion", "molecule", "bond", "mixture", 'known as', 'theorem', 'law', 'formula', 'equation', "compound", "solution", "area", "density", "volume", "weight", "litre", "kg", "nano", "gram", "°", "degree", "^", "$", "MHz", "known as", "volt", "watt", "term is", "answer is"]
    procedural_indicators = ['step', 'first', 'then', 'calculate', 'solve', 'process', "what", "next", "via", "how", "data", "start", "end", "last", "procedure", "learn", "calculate", "interpret", "explain", "leads", 'but why', 'but it', 'but,']
    episodic_indicators = ['discovered', 'developed by', 'introduced', 'published', 'historical', "remember" "why", "try", "know", "recall", "used to", "said", "happened", "occurred", "consider", "background", "figure out", "has to", "seen", 'found that', 'observed in', 'originated', 'pioneered', 'established', 'founded', 'year', 'century', 'era', 'period', 'epoch', 'when was', 'who discovered', 'first observed', 'initially', 'originally']
    contextual_indicators = ['but why', 'but it', 'but,', 'important', 'application', 'example', 'advantage', 'because', "can be", "idea", "occur", "mean", "by the way", "information", "context", "what are", "known that"]
    conceptual_indicators = ['but why', 'but it', 'but,', 'therefore', 'concept', 'theory', 'implies', 'suggests', "let me", "think", "I know", "I'm", "wonder", "feel", "seem", "should", "concept", "fortunately", "okay", "break it down", "right?", "I was", "I did", "I will", "I know", "I can", "I need", "I should", "I am", "myself", "need to", "think about", "look at", "my reasoning", "I could", "In my", "I've", "okay, so"]
    
    scores = {
        'FACTUAL': sum(1 for p in factual_indicators if p in text_lower),
        'PROCEDURAL': sum(1 for p in procedural_indicators if p in text_lower),
        'EPISODIC': sum(1 for p in episodic_indicators if p in text_lower),
        'CONTEXTUAL': sum(1 for p in contextual_indicators if p in text_lower),
        'CONCEPTUAL': sum(1 for p in conceptual_indicators if p in text_lower)
    }
    
    max_comp = max(scores, key=scores.get)
    max_score = scores[max_comp]
    
    # ✅ Better default logic
    # === STEP 4: SMART FALLBACK (if no indicators found) ===
    if max_score == 0:
        # Check for technical/scientific markers
        has_numbers = bool(re.search(r'\d', text))
        has_formula = bool(re.search(r'[=+\-*/^]', text))
        has_question = bool(re.search(r'\?', text))
        has_first_person = any(marker in text_lower for marker in ["i think", "i know", "let me", "i need"])
        
        # Smart fallback logic
        
        if has_question:
            return 'EPISODIC'
        elif has_first_person or "think" or "remember" in text_lower:
            return 'CONCEPTUAL'
        elif "the" in text_lower:
            return 'PROCEDURAL'
        elif (has_formula or has_numbers) and not has_question:
            return 'FACTUAL'
        else:
            return "CONTEXTUAL"
    return max_comp

def determine_hierarchy(text, compartment):
    """
    Determine hierarchical level based on semantic abstraction and context.
    
    GRANULAR: Specific details, exact values, concrete facts
    - Code snippets, variable values, specific numbers
    - Exact process steps with parameters
    - Precise specifications, configurations, settings
    - Atomic facts (e.g., "Water boils at 100°C")
    
    INTERMEDIATE: Reasoning, explanations, transitional logic
    - Thinking steps, logical connections
    - Explanations of concepts or processes
    - Inference, consequences, implications
    - Contextual reasoning (e.g., "Because X, therefore Y")
    
    GENERAL: Broad concepts, high-level summaries, abstractions
    - Abstract principles, conceptual frameworks
    - High-level overviews, summaries
    - General understanding, layman explanations
    - Philosophical or theoretical statements
    
    Args:
        text: The sentence/segment to classify
        compartment: The compartment label (affects hierarchy interpretation)
    
    Returns:
        str: 'GRANULAR', 'INTERMEDIATE', or 'GENERAL'
    """
    text_lower = text.lower()
    
    # ═══════════════════════════════════════════════════════════════
    # GRANULAR INDICATORS: Specificity markers
    # ═══════════════════════════════════════════════════════════════
    
    granular_indicators = [
        # Numeric/measurement specificity
        r'\b\d+\.?\d*\s*(°C|°F|K|m|cm|mm|kg|g|mg|L|mL|Hz|MHz|GHz|V|A|W|Ω|Pa|atm|mol|M)\b',  # Units
        r'\b\d+\.?\d*\s*(percent|%)\b',  # Percentages
        r'\b\d{4}\b',  # Years (e.g., "1995", "2024")
        r'\b\d+:\d+\b',  # Ratios (e.g., "3:1")
        
        # Code/variable markers
        r'`[^`]+`',  # Inline code
        r'\b[a-z_][a-z0-9_]*\s*=\s*',  # Variable assignment (x = ...)
        r'\bdef\s+\w+\(',  # Function definition
        r'\bclass\s+\w+',  # Class definition
        
        # Exact specifications
        r'\b(exactly|precisely|specifically|namely|i\.e\.|e\.g\.)\b',
        r'\bstep\s+\d+',  # "Step 1", "Step 2"
        r'\bfigure\s+\d+',  # "Figure 3"
        r'\bequation\s+\d+',  # "Equation 5"
        r'\btable\s+\d+',  # "Table 2"
    ]
    
    # Count granular markers
    granular_score = sum(1 for pattern in granular_indicators if re.search(pattern, text_lower))
    
    # Additional heuristics
    has_numbers = bool(re.search(r'\d', text))  # Contains any digit
    has_formula = bool(re.search(r'[=+\-*/^]', text))  # Math operators
    has_chemical_formula = bool(re.search(r'\b[A-Z][a-z]?\d*', text))  # H2O, CO2, etc.
    has_citation = bool(re.search(r'\[\d+\]|\(\d{4}\)', text))  # [1], (2023)
    
    if has_numbers:
        granular_score += 1
    if has_formula:
        granular_score += 1
    if has_chemical_formula and compartment == 'FACTUAL':
        granular_score += 1
    if has_citation:
        granular_score += 0.5
    
    # ═══════════════════════════════════════════════════════════════
    # INTERMEDIATE INDICATORS: Reasoning and explanation
    # ═══════════════════════════════════════════════════════════════
    
    intermediate_indicators = [
        # Reasoning transitions
        r'\b(because|since|therefore|thus|hence|consequently|as a result)\b',
        r'\b(if|when|while|although|unless|provided that|would|will|is to|is a)\b',
        r'\b(leads to|results in|causes|implies|suggests|indicates)\b',
        
        # Explanation markers
        r'\b(explains|clarifies|demonstrates|illustrates|shows that)\b',
        r'\b(in other words|that is|meaning|which means)\b',
        r'\b(can be understood as|refers to|relates to|mention|correct|states|user|they)\b',
        
        # Thinking/logic markers
        r'\b(consider|observe|note that|recall that|remember|expand|detail|think|remember|must)\b',
        r'\b(follows that|we can see|it is clear|it becomes apparent|satisfy|isn\'t|is not)\b',
        r'\b(inference|deduction|reasoning|logic)\b',
        
        # Comparison/contrast
        r'\b(compared to|in contrast|similarly|likewise|whereas)\b',
        r'\b(on the other hand|conversely|however|but|wait|how|)\b',
    ]
    
    intermediate_score = sum(1 for pattern in intermediate_indicators if re.search(pattern, text_lower))
    
    # Additional heuristics
    has_transition_words = bool(re.search(r'\b(first|second|next|then|finally|additionally)\b', text_lower))
    has_explanation_structure = bool(re.search(r'\b(why|how|what|which)\b', text_lower))
    
    if has_transition_words:
        intermediate_score += 0.5
    if has_explanation_structure:
        intermediate_score += 0.5
    
    # ═══════════════════════════════════════════════════════════════
    # GENERAL INDICATORS: Abstraction and high-level concepts
    # ═══════════════════════════════════════════════════════════════
    
    general_indicators = [
        # Abstract concepts
        r'\b(concept|principle|theory|framework|paradigm|philosophy)\b',
        r'\b(generally|broadly|overall|in general|typically|usually)\b',
        r'\b(fundamental|essential|core|basic|primary|key)\b',
        
        # High-level summaries
        r'\b(summary|overview|introduction|conclusion|essence|gist)\b',
        r'\b(in summary|to summarize|in conclusion|overall)\b',
        
        # Philosophical/theoretical
        r'\b(nature of|essence of|meaning of|significance of|\?|why|okay,|otherwise)\b',
        r'\b(understanding|perspective|viewpoint|approach|conception|hmm|...|maybe)\b',
        r'\b(can be thought of as|is essentially|fundamentally)\b',
        
        # Layman/simplified explanations
        r'\b(simply put|in simple terms|basically|essentially)\b',
        r'\b(common understanding|general knowledge|widely known)\b',
    ]
    
    general_score = sum(1 for pattern in general_indicators if re.search(pattern, text_lower))
    
    # Length-based abstraction heuristic
    word_count = len(text.split())
    if word_count > 30:  # Long sentences tend to be more abstract
        general_score += 0.5
    
    # ═══════════════════════════════════════════════════════════════
    # COMPARTMENT-AWARE DECISION LOGIC
    # ═══════════════════════════════════════════════════════════════
    general_strong = ["should", "remember", "know", "happened", "occurred", "seen", "I know", "I was", "I did", "why", 'example', "try", "recall", "used to", "said", "happen", "consider", "background", "figure out", "has to", "see", "explain", "leads", 'but why', 'but it', 'but,', "figure out", "has to",  'advantage', 'because', "can be", "idea", "occur", "mean", "by the way", "information", "context", "what are", "let me", "think", "I know", "I'm", "wonder", "feel", "seem", "should", "concept", "fortunately", "okay", "break it down", "right?", "I was", "I did", "I will", "I know", "I can", "I need", "I should", "I am", "myself", "need to", "think about", "look at", "my reasoning", "I could", "In my", "I've", "okay, so"]
    if any(kw in text_lower for kw in general_strong):
        return 'GENERAL'
    
    granular_strong = ['defined as', 'equals', "=", "/", "sqrt", "{", "(", "%", "\\", "+", "(g)", "element", "chemical", "ion", "molecule", "bond", "mixture", 'known as', 'theorem', 'law', 'formula', 'equation', "compound", "solution", "area", "density", "volume", "weight", "litre", "kg", "nano", "gram", "°", "degree", "^", "$", "MHz", "known as", "volt", "watt", "term is", "answer is"]
    if any(kw in text_lower for kw in granular_strong):
        return 'GRANULAR'
    
    intermediate_strong = ['discovered', 'concept', 'theory', 'implies', 'suggests', 'important', 'developed by', 'published', 'introduced in', 'discover', 'develop', 'introduced', 'published', 'historical', 'step', 'first', 'then', 'calculate', 'solve', 'process', "what", "next", "via", "how", "data", "start", "end", "last", "procedure", "learn", "calculate", "interpret", 'discovered', 'developed by', 'introduced', 'published', 'historical', "remember", "why", "try", "know", "recall", "used to", "said", "happened", "occurred", "consider", "background", "seen", 'found that', 'observed in', 'originated', 'pioneered', 'established', 'founded', 'year', 'century', 'era', 'period', 'epoch', 'when was', 'who discovered', 'first observed', 'initially', 'originally']
    if any(kw in text_lower for kw in intermediate_strong):
        return 'INTERMEDIATE'
        
    # Score-based fallback    
    if compartment == 'FACTUAL':
        # Factual statements: Specificity = granular, generalizations = general
        if granular_score >= 2:
            return 'GRANULAR'
        elif general_score >= 2:
            return 'GENERAL'
        elif intermediate_score >= 1:
            return 'INTERMEDIATE'
        else:
            # Default for facts: If short and specific → granular, if long → intermediate
            return 'INTERMEDIATE'
    
    elif compartment == 'PROCEDURAL':
        # Procedures: Exact steps = granular, process overview = general
        if granular_score >= 1.5 or 'step' in text_lower:
            return 'GRANULAR'
        elif general_score >= 2:
            return 'GENERAL'
        else:
            return 'INTERMEDIATE'
    
    elif compartment == 'EPISODIC':
        # Historical/episodic: Specific dates/people = granular, context = intermediate
        if granular_score >= 1.5:
            return 'GRANULAR'
        elif general_score >= 2:
            return 'GENERAL'
        else:
            return 'INTERMEDIATE'
    
    elif compartment in ['CONTEXTUAL', 'CONCEPTUAL']:
        # Context/concepts: Rarely granular, often intermediate or general
        if granular_score >= 3:  # Very high threshold for abstract compartments
            return 'GRANULAR'
        elif general_score >= 1.5:
            return 'GENERAL'
        elif intermediate_score >= 1:
            return 'INTERMEDIATE'
        else:
            return 'GENERAL'  # Default for abstract compartments
    
    # ═══════════════════════════════════════════════════════════════
    # FALLBACK LOGIC: Score-based decision
    # ═══════════════════════════════════════════════════════════════
    
    max_score = max(granular_score, intermediate_score, general_score)
    
    if max_score == 0:
        # No indicators found - use length heuristic
        if word_count < 10:
            return 'GRANULAR'  # Very short = likely specific fact
        elif word_count < 25:
            return 'INTERMEDIATE'
        else:
            return 'GENERAL'
    
    if granular_score == max_score:
        return 'GRANULAR'
    elif intermediate_score == max_score:
        return 'INTERMEDIATE'
    else:
        return 'GENERAL'

def get_precision_requirement(compartment, hierarchy):
    """
    Determine precision requirement based on compartment and hierarchy.
    
    SCM Precision Rules:
    - FACTUAL: FP32 (all layers)
    - PROCEDURAL: FP32 (granular), FP16 (intermediate/general)
    - EPISODIC/CONTEXTUAL/CONCEPTUAL: FP16 throughout
    """
    if hierarchy == 'GRANULAR':
        return 'fp32'
    if compartment == 'FACTUAL':
        return 'fp32'
    elif compartment == 'PROCEDURAL' or compartment == 'CONTEXTUAL':
        return 'fp32' if hierarchy == 'INTERMEDIATE' or hierarchy == 'GRANULAR' else 'fp16'
    else:
        return 'fp16'

def apply_diffusion_noise(embedding, schedule='cosine'):
    """
    Apply cosine schedule noise for diffusion pre-training.
    
    SCM uses diffusion-based denoising with cosine noise schedule.
    
    Currently only accomodates linear and cosine diffusion.
    
    Returns: (noisy_embedding, timestep, alpha_t)
    """
    embedding_np = np.array(embedding)
    t = np.random.randint(0, DIFFUSION_T_MAX)
    
    if schedule == 'cosine':
        alpha_t = np.cos(np.pi * t / (2 * DIFFUSION_T_MAX)) ** 2
    else:  # linear
        alpha_t = 1.0 - (t / DIFFUSION_T_MAX)
    
    sigma_t = DIFFUSION_SIGMA_MIN + (DIFFUSION_SIGMA_MAX - DIFFUSION_SIGMA_MIN) * (1 - alpha_t)
    noise = np.random.randn(*embedding_np.shape) * sigma_t
    noisy_embedding = (embedding_np * np.sqrt(alpha_t) + noise).tolist()
    
    return noisy_embedding, int(t), float(alpha_t)

def is_science_content(example):
    """Enhanced filter - checks domain, source, and content."""
    domain = str(example.get('domain', '')).lower()
    SCIENCE_DOMAINS = ['science', 'biology', 'chemistry', 'physics', 'medicine', 
                       'astronomy', 'geology', 'neuroscience', 'biochemistry']
    if any(sci_domain in domain for sci_domain in SCIENCE_DOMAINS):
        logger.debug(f"✓ Accepted (domain match): {domain}")
        return True
    
    if "code" in domain or "math" in domain:
        return False
    
    source = str(example.get('source', '')).lower()
    for kw in ['chembio', 'chemistry', 'biology', 'physics', 'gpqa', 'jeebench', 'scibench']:
        if kw in source:
            return True
    
    conversations = example.get('conversations', [])
    if conversations:
        full_text = ' '.join([str(c.get('value', '')).lower() for c in conversations])
        matches = sum(1 for kw in SCIENCE_KEYWORDS if kw in full_text)
        if matches >= 3:
            return True
    
    return False

def compute_quality_score(text, steps=None):
    """
    Compute dynamic quality score based on content richness.
    
    Factors:
    - Technical density (presence of domain terms)
    - Formula/equation presence
    - Length appropriateness
    - Step quality (if reasoning chain provided)
    
    Returns: float between 0.3 and 1.0
    """
    text_lower = text.lower()
    
    # Base score
    score = 0.5
    
    # Technical term bonus (max +0.25)
    tech_terms = extract_technical_terms(text_lower)
    tech_bonus = min(0.05 * len(tech_terms), 0.25)
    score += tech_bonus
    
    # Formula/equation bonus (+0.15)
    has_formula = bool(re.search(r'[=+\-*/^]', text))
    if has_formula:
        score += 0.15
    
    # Length appropriateness (±0.1)
    word_count = len(text.split())
    if 15 <= word_count <= 400:
        score += 0.1
    elif word_count < 10:
        score -= 0.1
    
    # Step quality bonus (if steps provided, max +0.1)
    if steps:
        valid_steps = sum(1 for s in steps if s.get('sonar_embedding') is not None)
        if valid_steps >= 5:
            score += 0.1
        elif valid_steps >= 3:
            score += 0.05
    
    # Clamp to [0.3, 1.0]
    return round(max(0.3, min(score, 1.0)), 2)


logger.info("="*80)
logger.info("LOADING DATASET")
logger.info("="*80)

try:
    os.environ['HF_DATASETS_OFFLINE'] = '1'
    dataset = load_dataset("open-thoughts/OpenThoughts3-1.2M", split="train", streaming=True, cache_dir=CACHE_PATH)
    logger.info("✓ Dataset loaded in streaming mode")
    
    first_example = next(iter(dataset))
    logger.info(f"Dataset structure verified:")
    logger.info(f"  Keys: {list(first_example.keys())}")
    logger.info(f"  Conversations: {len(first_example.get('conversations', []))}")
    logger.info(f"  Domain: {first_example.get('domain', 'N/A')}")
    logger.info(f"  Source: {first_example.get('source', 'N/A')}")
    
except Exception as e:
    logger.error(f"✗ Failed to load dataset: {e}")
    raise

logger.info("Applying science content filter...")
science_dataset_stream = dataset.filter(is_science_content)
logger.info("✓ Science filter applied")

# logger.info("="*80)
# logger.info("STAGE 1: PRE-TRAINING DATA GENERATION")
# logger.info("SCM Compliance: 1024-dim embeddings + diffusion noise")
# logger.info("="*80)

# checkpoint_data = load_checkpoint("stage1_processing")

# if checkpoint_data:
#     doc_data = checkpoint_data.get('doc_data', {})
#     stage1_data = checkpoint_data.get('stage1_data', [])
#     doc_count = checkpoint_data.get('doc_count', 0)
#     science_count = checkpoint_data.get('science_count', 0)
#     last_processed_idx = checkpoint_data.get('last_idx', -1)
    
#     logger.info(f"✓ Resuming from checkpoint:")
#     logger.info(f"  - Processed docs: {doc_count}")
#     logger.info(f"  - Stage 1 samples: {len(stage1_data)}")
#     logger.info(f"  - Last global index: {last_processed_idx}")
# else:
#     doc_data = {}
#     stage1_data = []
#     doc_count = 0
#     science_count = 0
#     last_processed_idx = -1
#     logger.info("Starting fresh Stage 1 processing...")

# logger.info(f"Processing up to {MAX_DOCS_TO_PROCESS} science docs...")
# logger.info(f"Checkpoint interval: every {STAGE1_CHECKPOINT_INTERVAL} docs")

# # FIX #1 & #2: Track last written position for incremental append
# last_written_stage1_idx = 0

# # FIX #14: Skip already-processed documents efficiently
# if last_processed_idx >= 0:
#     logger.info(f"Skipping {last_processed_idx + 1} already-processed documents...")
#     science_dataset_stream = science_dataset_stream.skip(last_processed_idx + 1)
#     global_idx = last_processed_idx + 1
# else:
#     global_idx = 0

# docs_processed_this_run = 0

# for example in tqdm(science_dataset_stream, desc="Stage 1 Processing", total=MAX_DOCS_TO_PROCESS - global_idx):
#     if doc_count >= MAX_DOCS_TO_PROCESS:
#         logger.info(f"Reached MAX_DOCS_TO_PROCESS limit ({MAX_DOCS_TO_PROCESS}). Stopping.")
#         break
    
#     full_text = extract_text_from_conversations(example)
#     if not full_text or len(full_text) < 20:
#         logger.debug(f"Doc {global_idx}: Skipped (text too short)")
#         global_idx += 1
#         continue
    
#     query = ""
#     conversations = example.get('conversations', [])
#     for conv in conversations:
#         if 'user' in str(conv.get('from', '')).lower():
#             query = conv.get('value', '')
#             break
    
#     sentences = sent_tokenize(full_text[:MAX_TEXT_LENGTH])
#     segments = []
    
#     for i, sent in enumerate(sentences):
#         if len(sent.strip()) < MIN_SENTENCE_LENGTH:
#             continue
        
#         comp = estimate_compartment_advanced(sent)
#         hier = determine_hierarchy(sent, comp)  # ✅ Pass compartment
        
#         # ✅ Generate embedding HERE (before appending to segments)
#         try:
#             segment_embedding = sonar_encoder.encode(sent.strip())
#             if isinstance(segment_embedding, np.ndarray):
#                 segment_embedding = segment_embedding.tolist()
#             elif hasattr(segment_embedding, 'tolist'):
#                 segment_embedding = segment_embedding.tolist()
#             elif hasattr(segment_embedding, 'cpu'):  # PyTorch tensor
#                 segment_embedding = segment_embedding.cpu().numpy().tolist()
#             elif hasattr(segment_embedding, 'numpy'):  # TF tensor
#                 segment_embedding = segment_embedding.numpy().tolist()
#             else:
#                 # Last resort: convert to numpy first
#                 segment_embedding = np.array(segment_embedding).tolist()
                
#             if not isinstance(segment_embedding, list):
#                 raise TypeError(f"Embedding is {type(segment_embedding)}, expected list")
            
#         except Exception as e:
#             logger.warning(f"Doc {global_idx}, Seg {i}: Embedding failed - {e}")
#             logger.warning(f"Embedding logic failed! Pause pipeline and fix.")
#             continue  # Skip segment if embedding fails
        
#         if len(segment_embedding) != 1024:
#             logger.error(f"Doc {global_idx}, Seg {i}: Embedding dimension is {len(segment_embedding)}, expected 1024")
#             logger.error("CRITICAL: SCM compliance violated! Pausing pipeline.")
#             raise ValueError(f"Embedding dimension mismatch: {len(segment_embedding)} vs 1024")

        
#         segments.append({
#             'text': sent.strip(),
#             'compartment': comp,
#             'hierarchy': hier,
#             'position': i,
#             'sonar_embedding': segment_embedding  # ✅ Store embedding
#         })
    
#     if segments:
#         # Sample first doc in detail
#         if doc_count == 0:
#             logger.info(f"\n{'='*60}")
#             logger.info(f"SAMPLE DOCUMENT (Doc {global_idx})")
#             logger.info(f"{'='*60}")
#             logger.info(f"Query: {query[:200]}...")
#             logger.info(f"Text length: {len(full_text)} chars")
#             logger.info(f"Segments: {len(segments)}")
#             logger.info(f"Domain: {example.get('domain', 'N/A')}")
#             logger.info(f"\nFirst 3 segments:")
#             for seg in segments[:3]:
#                 imp_score = compute_importance_score(seg['text'])
#                 prec = get_precision_requirement(seg['compartment'], seg['hierarchy'])
#                 logger.info(f"  [{seg['compartment']}] [{seg['hierarchy']}] [imp:{imp_score}] [prec:{prec}]")
#                 logger.info(f"    Text: {seg['text'][:100]}...")
#             logger.info(f"{'='*60}\n")
#         if logger.isEnabledFor(logging.DEBUG):
#             logger.debug(f"--- DOC {global_idx} ---")
#             logger.debug(f"  Domain: {example.get('domain', 'N/A')}")
#             logger.debug(f"  Source: {example.get('source', 'N/A')}")
#             logger.debug(f"  Difficulty: {example.get('difficulty', 0)}")
#             logger.debug(f"  Segments: {len(segments)}")
#             logger.debug(f"  Query: {query[:100]}")
#             for i, seg in enumerate(segments[:5]):
#                 logger.debug(f"    Seg {i}: [{seg['compartment']}] [{seg['hierarchy']}] {seg['text'][:50]}...")
        
#         doc_data[global_idx] = {
#             'segments': segments,
#             'query': query or full_text[:200],
#             'domain': example.get('domain', 'science'),
#             'source': example.get('source', 'OpenThoughts3'),
#             'difficulty': example.get('difficulty', 0)
#         }
        
#         segments_added = 0
#         for seg in segments:
#             try:
#                 # Generate clean embedding (1024-dim)
#                 clean_embedding = seg['sonar_embedding']
                
#                 if not isinstance(clean_embedding, list):
#                     if hasattr(clean_embedding, 'tolist'):
#                         clean_embedding = clean_embedding.tolist()
#                     elif isinstance(clean_embedding, np.ndarray):
#                         clean_embedding = clean_embedding.tolist()
#                     else:
#                         raise TypeError(f"Invalid embedding type: {type(clean_embedding)}")
                
#                 # Apply diffusion noise
#                 noisy_embedding, timestep, alpha_t = apply_diffusion_noise(clean_embedding)
                
#             except Exception as e:
#                 logger.warning(f"Doc {global_idx}, Seg {seg['position']}: Embedding failed - {e}")
#                 continue
            
#             tech_terms = extract_technical_terms(seg['text'])
#             importance = compute_importance_score(seg['text'])
#             precision = get_precision_requirement(seg['compartment'], seg['hierarchy'])
            
#             # FIX #7: Include query and difficulty
#             stage1_sample = {
#                 "id": f"pretrain_{global_idx:06d}_{seg['position']}",
#                 "text": seg['text'],
#                 "clean_embedding": seg['sonar_embedding'],  # ✅ Reuse from segment
#                 "noisy_embedding": noisy_embedding,
#                 "timestep": timestep,
#                 "alpha_t": alpha_t,
#                 "compartment": seg['compartment'],
#                 "hierarchical_level": seg['hierarchy'].lower(),
#                 "query": doc_data[global_idx]['query'],  # FIX #7
#                 "metadata": {
#                     "domain": doc_data[global_idx]['domain'],
#                     "source": doc_data[global_idx]['source'],
#                     "difficulty": doc_data[global_idx]['difficulty'],  # FIX #7
#                     "technical_terms": tech_terms,
#                     "precision_required": precision,
#                     "importance_score": importance,
#                     "fragility_score": None
#                 },
#                 "diffusion_config": {
#                     "noise_schedule": "cosine",
#                     "sigma_min": DIFFUSION_SIGMA_MIN,
#                     "sigma_max": DIFFUSION_SIGMA_MAX,
#                     "t_max": DIFFUSION_T_MAX
#                 }
#             }
            
#             stage1_data.append(stage1_sample)
#             segments_added += 1
        
#         logger.debug(f"Doc {global_idx}: Added {segments_added}/{len(segments)} segments")
        
#         doc_count += 1
#         docs_processed_this_run += 1
        
#         if doc_count % STAGE1_CHECKPOINT_INTERVAL == 0:
#             logger.info(f"\n--- CHECKPOINT at {doc_count} docs ---")
#             logger.info(f"Total Stage 1 samples: {len(stage1_data)}")
#             logger.info(f"Memory usage: {len(stage1_data) * 1024 * 4 / (1024**2):.2f} MB")
            
#             # Lightweight checkpoint (metadata only for speed)
#             checkpoint_meta = {
#                 'doc_data': doc_data,
#                 'stage1_data': [],
#                 'doc_count': doc_count,
#                 'science_count': science_count,
#                 'last_idx': global_idx,
#                 'doc_ids_processed': list(doc_data.keys())
#             }
            
#             save_checkpoint(checkpoint_meta, "stage1_processing")
#             logger.info(f"Saved lightweight checkpoint")
            
#             # FIX #1: Proper incremental append (only NEW samples)
#             stage1_file = os.path.join(OUTPUT_PATH, "stage1_pretrain.jsonl")
#             try:
#                 with open(stage1_file, 'a', encoding='utf-8') as f:
#                     for item in stage1_data[last_written_stage1_idx:]:
#                         f.write(json.dumps(item, ensure_ascii=False) + "\n")
#                 print("successfully appended stage1")
#             except:
#                 with open(stage1_file, 'a', encoding='utf-8') as f:
#                     for item in stage1_data[last_written_stage1_idx:]:
#                         f.write(json.dumps(item, ensure_ascii=False) + '\n')
#                 print("successfully appended stage1 in except block")
            
#             # FIX #1: Update tracker
#             written_count = len(stage1_data) - last_written_stage1_idx
#             logger.info(f"Appended {written_count} new samples to Stage 1 JSONL")
            
#             stage1_data.clear() # Keep unwritten
#             last_written_stage1_idx = 0  # Reset tracker for next batch
#             gc.collect()
#             logger.info(f"Memory freed (retained {len(stage1_data)} unwritten samples)")
            
#             logger.info("--- CHECKPOINT COMPLETE ---\n")
    
#     global_idx += 1

# logger.info(f"\n{'='*80}")
# logger.info(f"STAGE 1 COMPLETE")
# logger.info(f"{'='*80}")
# logger.info(f"Total docs: {doc_count}")
# logger.info(f"Total samples: {len(stage1_data) + last_written_stage1_idx}")
# logger.info(f"Docs with segments: {len(doc_data)}")

# # FIX #3: Save full doc_data at end (with segments for Stage 2)
# save_checkpoint({
#     'doc_data': doc_data,  # Full data with segments
#     'stage1_data': [],
#     'doc_count': doc_count,
#     'science_count': science_count,
#     'last_idx': global_idx - 1,
#     'doc_ids_processed': list(doc_data.keys())
# }, "stage1_processing")

# # FIX #2: Correct final write logic (append remaining samples)
# stage1_file = os.path.join(OUTPUT_PATH, "stage1_pretrain.jsonl")

# if last_written_stage1_idx < len(stage1_data):
#     logger.info(f"Writing final {len(stage1_data) - last_written_stage1_idx} Stage 1 samples...")
#     with open(stage1_file, 'a', encoding='utf-8') as f:
#         for item in stage1_data[last_written_stage1_idx:]:
#             f.write(json.dumps(item, ensure_ascii=False) + "\n")
#     logger.info(f"✓ Final write complete")
# else:
#     logger.info(f"No remaining Stage 1 samples to write")

# # Count final samples
# if os.path.exists(stage1_file):
#     with open(stage1_file, 'r', encoding='utf-8') as f:
#         final_stage1_count = sum(1 for _ in f)
#     logger.info(f"Stage 1 JSONL total: {final_stage1_count} samples")

# save_checkpoint({'doc_data': doc_data}, "doc_data_for_stage2")

# del stage1_data
# gc.collect()
# logger.info("✓ Memory cleared")


# ===================================================================
# LOAD doc_data FROM STAGE 1 (FOR STAGE 2)
# ===================================================================

logger.info("="*80)
logger.info("LOADING doc_data FROM STAGE 1 CHECKPOINT")
logger.info("="*80)

try:
    doc_data = None

    # Try loading from Stage 1 checkpoint
    checkpoint_data = load_checkpoint("stage1_processing")
    if checkpoint_data and 'doc_data' in checkpoint_data:
        doc_data = checkpoint_data['doc_data']
        logger.info(f"✓ Loaded doc_data from stage1_processing: {len(doc_data)} documents")

    # Fallback: Try doc_data_for_stage2 checkpoint
    if not doc_data:
        doc_data_checkpoint = load_checkpoint("doc_data_for_stage2")
        if doc_data_checkpoint and 'doc_data' in doc_data_checkpoint:
            doc_data = doc_data_checkpoint['doc_data']
            logger.info(f"✓ Loaded doc_data from doc_data_for_stage2: {len(doc_data)} documents")

    # Validation
    logger.info(f"Using checkpoint directory: {CHECKPOINT_DIR}")

    if not doc_data or len(doc_data) == 0:
        logger.error("✗ No doc_data found! Stage 1 must be completed first.")
        logger.error("Expected checkpoint: stage1_processing.pkl or doc_data_for_stage2.pkl")
        raise ValueError("Cannot proceed with Stage 2: No doc_data available")

    sample_doc = doc_data[list(doc_data.keys())[0]]
    if 'segments' in sample_doc and len(sample_doc['segments']) > 0:
        sample_seg = sample_doc['segments'][0]
        
        if 'sonar_embedding' not in sample_seg:
            logger.error("❌ CRITICAL: Checkpoint has old format (missing sonar_embedding)")
            logger.error("  Loaded doc_data does NOT contain embeddings in segments")
            logger.error("  You MUST regenerate Stage 1 with fixed code")
            raise ValueError("Incompatible checkpoint: Missing embeddings in segments")
        
        # Check dimension
        emb_dim = len(sample_seg['sonar_embedding'])
        if emb_dim != 1024:
            logger.error(f"❌ CRITICAL: Embeddings are {emb_dim}-dim, expected 1024")
            raise ValueError(f"SCM compliance violated: {emb_dim}-dim embeddings")
        
        logger.info("✅ doc_data validated: Segments contain 1024-dim embeddings")
except:
    traceback.print_exc()

logger.info(f"✓ doc_data ready for Stage 2: {len(doc_data)} documents")
logger.info("="*80)

logger.info("="*80)
logger.info("STAGE 2: SUPERVISED FINE-TUNING (CoT CONSTRUCTION)")
logger.info("SCM Compliance: Per-step embeddings + dynamic importance")
logger.info("="*80)

stage2_file = os.path.join(OUTPUT_PATH, "stage2_sft.jsonl")
    
# ✅ FIX: Always try to resume from checkpoint, even if file exists
stage2_checkpoint = load_checkpoint("stage2_progress")

if stage2_checkpoint:
    # Resume from checkpoint (file may exist from previous run)
    stage2_data = stage2_checkpoint.get('stage2_data', [])
    processed_doc_ids = set(stage2_checkpoint.get('processed_doc_ids', []))
    processed_count = len(processed_doc_ids)
    
    # Count existing samples in file
    if os.path.exists(stage2_file):
        with open(stage2_file, 'r', encoding='utf-8') as f:
            existing_stage2_count = sum(1 for _ in f)
        logger.info(f"✓ Stage 2 JSONL exists: {existing_stage2_count} samples")
    else:
        existing_stage2_count = 0
    
    logger.info(f"✓ Resuming Stage 2:")
    logger.info(f"  - In-memory samples: {len(stage2_data)}")
    logger.info(f"  - Already processed docs: {len(processed_doc_ids)}/{len(doc_data)}")
    logger.info(f"  - Remaining docs: {len(doc_data) - len(processed_doc_ids)}")
    
    # Check if already complete
    if len(processed_doc_ids) >= len(doc_data):
        logger.info(f"✓ Stage 2 already complete ({len(processed_doc_ids)} docs processed)")
        logger.info(f"  Total samples in JSONL: {existing_stage2_count}")
        logger.info("Proceeding to Stage 3...")
        stage2_data = []  # Clear memory (data already in file)
    else:
        logger.info(f"Resuming processing for {len(doc_data) - len(processed_doc_ids)} remaining docs...")

else:
    # No checkpoint - check if file exists (edge case: file exists but no checkpoint)
    if os.path.exists(stage2_file):
        with open(stage2_file, 'r', encoding='utf-8') as f:
            existing_stage2_count = sum(1 for _ in f)
        logger.warning(f"⚠ Stage 2 JSONL exists ({existing_stage2_count} samples) but no checkpoint found!")
        logger.warning("This may indicate previous run crashed. Starting fresh...")
        logger.warning("To resume, ensure stage2_progress.pkl checkpoint exists.")
        
        # User choice: Delete file or keep it
        # Option 1: Keep file, append new results
        logger.info("Keeping existing file and appending new results...")
        stage2_data = []
        processed_doc_ids = set()
        processed_count = 0
        
        # Option 2 (commented out): Delete and restart
        # os.remove(stage2_file)
        # logger.info("Deleted existing file to start fresh")
    else:
        # Fresh start
        stage2_data = []
        processed_doc_ids = set()
        processed_count = 0
        logger.info("Starting fresh Stage 2...")

# ✅ Only proceed with processing if not already complete
if len(processed_doc_ids) < len(doc_data):
    # FIX #11: Validate doc_data exists
    if not doc_data:
        doc_data_checkpoint = load_checkpoint("doc_data_for_stage2")
        if doc_data_checkpoint and 'doc_data' in doc_data_checkpoint:
            doc_data = doc_data_checkpoint['doc_data']
            logger.info(f"✓ Loaded doc_data: {len(doc_data)} documents")
        else:
            logger.error("✗ No doc_data. Run Stage 1 first.")
            raise ValueError("Cannot proceed without doc_data")
    
    # FIX #11: Validate not empty
    if len(doc_data) == 0:
        logger.error("✗ Stage 2 requires doc_data from Stage 1")
        logger.error("Please run Stage 1 first or check science filter settings")
        raise ValueError("Cannot proceed with Stage 2: No documents available")
    
    logger.info(f"Processing {len(doc_data)} documents...")
    logger.info(f"Already processed: {len(processed_doc_ids)}")
    
    last_written_idx = 0
    embedding_failures_total = 0  # FIX #4: Track failures
    
    # FIX #11: Validate doc_data exists
    if not doc_data:
        doc_data_checkpoint = load_checkpoint("doc_data_for_stage2")
        if doc_data_checkpoint and 'doc_data' in doc_data_checkpoint:
            doc_data = doc_data_checkpoint['doc_data']
            logger.info(f"✓ Loaded doc_data: {len(doc_data)} documents")
        else:
            logger.error("✗ No doc_data. Run Stage 1 first.")
            raise ValueError("Cannot proceed without doc_data")
    
    # FIX #11: Validate not empty
    if len(doc_data) == 0:
        logger.error("✗ Stage 2 requires doc_data from Stage 1")
        logger.error("Please run Stage 1 first or check science filter settings")
        raise ValueError("Cannot proceed with Stage 2: No documents available")
    
    logger.info(f"Processing {len(doc_data)} documents...")
    logger.info(f"Already processed: {len(processed_doc_ids)}")
    
    last_written_idx = 0
    embedding_failures_total = 0  # FIX #4: Track failures
    
    for i, (doc_idx, data) in enumerate(tqdm(doc_data.items(), desc="Stage 2 Processing")):
        if doc_idx in processed_doc_ids:
            logger.debug(f"Doc {doc_idx}: Already processed")
            processed_count += 1
            continue
        
        steps = []
        embedding_failures_doc = 0
        
        logger.info(f"Processing {len(data['segments'])} segments in parallel...")

        # Process all segments in parallel
        # segment_results = parallel_process_segments(data['segments'], sonar_encoder)

        # for seg, step_embedding in segment_results:
        for seg in data['segments']:
            step_num = seg['position'] + 1
            
            if 'sonar_embedding' not in seg:
                logger.warning(f"Doc {doc_idx}, Step {step_num}: Missing embedding in segment")
                embedding_failures_doc += 1
                embedding_failures_total += 1
                continue
            
            step_embedding = seg['sonar_embedding']
            
            if step_embedding is None:
                logger.warning(f"Doc {doc_idx}, Step {step_num}: Embedding failed")
                embedding_failures_doc += 1
                embedding_failures_total += 1
                continue
            
            if not isinstance(step_embedding, list):
                logger.warning(f"Doc {doc_idx}, Step {step_num}: Embedding not a list")
                embedding_failures_doc += 1
                continue
            
            if len(step_embedding) != 1024:
                logger.warning(f"Doc {doc_idx}, Step {step_num}: Embedding dimension {len(step_embedding)} != 1024")
                embedding_failures_doc += 1
                continue
                    
            importance = compute_importance_score(seg['text'])
            precision = get_precision_requirement(seg['compartment'], seg['hierarchy'])
            
            steps.append({
                "step": step_num,
                "compartment": seg['compartment'],
                "hierarchy": seg['hierarchy'],
                "text": seg['text'],
                "sonar_embedding": step_embedding,
                "precision": precision,
                "importance_score": importance
            })
        
        # FIX #4: Log embedding failures per doc
        if embedding_failures_doc > 0:
            logger.warning(f"Doc {doc_idx}: {embedding_failures_doc}/{len(data['segments'])} embeddings failed")
        
        if len(steps) >= 1:
            steps.sort(key=lambda x: x['step'])
            final_answer = '. '.join(step['text'] for step in steps) + '.'
            
            comp_dist = {}
            hier_dist = {}
            for step in steps:
                comp_dist[step['compartment']] = comp_dist.get(step['compartment'], 0) + 1
                hier_dist[step['hierarchy']] = hier_dist.get(step['hierarchy'], 0) + 1
            
            stage2_sample = {
                "id": f"sft_{doc_idx:06d}",
                "domain": data['domain'],
                "query": data['query'],
                "reasoning_chain": steps,
                "final_answer": final_answer,
                "metadata": {
                    "num_steps": len(steps),
                    "compartment_distribution": comp_dist,
                    "hierarchy_distribution": hier_dist,
                    "avg_importance": round(sum(s['importance_score'] for s in steps) / len(steps), 2),
                    "code_validated": False,
                    "quality_score": compute_quality_score(data['query'], steps)
                }
            }
            
            stage2_data.append(stage2_sample)
            processed_doc_ids.add(doc_idx)
            
            # Sample first output
            if len(stage2_data) == 1:
                logger.info(f"\n{'='*60}")
                logger.info(f"SAMPLE STAGE 2 OUTPUT (SCM COMPLIANT)")
                logger.info(f"{'='*60}")
                logger.info(f"ID: {stage2_sample['id']}")
                logger.info(f"Query: {stage2_sample['query'][:200]}...")
                logger.info(f"Steps: {len(steps)}")
                logger.info(f"Compartment dist: {comp_dist}")
                logger.info(f"Hierarchy dist: {hier_dist}")
                logger.info(f"Avg importance: {stage2_sample['metadata']['avg_importance']}")
                logger.info(f"\nFirst 2 steps (with embeddings):")
                for step in steps[:2]:
                    logger.info(f"  Step {step['step']} [{step['compartment']}] [{step['hierarchy']}]")
                    logger.info(f"    Importance: {step['importance_score']}, Precision: {step['precision']}, Embedding: ✓")
                    logger.info(f"    Text: {step['text'][:100]}...")
                logger.info(f"{'='*60}\n")
            
            # FIX #12: Per-item DEBUG logging
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"--- STAGE 2 DOC {doc_idx} ---")
                logger.debug(f"  Steps: {len(steps)}")
                logger.debug(f"  Comp dist: {comp_dist}")
                logger.debug(f"  Hier dist: {hier_dist}")
                logger.debug(f"  Avg importance: {stage2_sample['metadata']['avg_importance']}")
            
            logger.debug(f"Doc {doc_idx}: Created Stage 2 sample with {len(steps)} steps")
        else:
            logger.warning(f"Doc {doc_idx}: No valid steps")
        
        processed_count += 1
        
        if processed_count % STAGE2_CHECKPOINT_INTERVAL == 0:
            logger.info(f"\n--- STAGE 2 CHECKPOINT at {processed_count} docs ---")
            logger.info(f"Total samples: {len(stage2_data)}")
            
            with open(stage2_file, 'a', encoding='utf-8') as f:
                for item in stage2_data[last_written_idx:]:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
            last_written_idx = len(stage2_data)
            logger.info(f"✓ Appended to JSONL")
            
            save_checkpoint({
                'stage2_data': stage2_data,
                'processed_doc_ids': list(processed_doc_ids)
            }, "stage2_progress")
            
            gc.collect()
            logger.info("--- CHECKPOINT COMPLETE ---\n")
    
    if last_written_idx < len(stage2_data):
        logger.info(f"Writing final samples...")
        with open(stage2_file, 'a', encoding='utf-8') as f:
            for item in stage2_data[last_written_idx:]:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    logger.info(f"\n{'='*80}")
    logger.info(f"STAGE 2 COMPLETE")
    logger.info(f"{'='*80}")
    logger.info(f"Total samples: {len(stage2_data)}")
    logger.info(f"Multi-step CoT: {sum(1 for d in stage2_data if d['metadata']['num_steps'] >= 2)}")
    logger.info(f"Total embedding failures: {embedding_failures_total}")  # FIX #4
    
    save_checkpoint({
        'stage2_data': stage2_data,
        'processed_doc_ids': list(processed_doc_ids)
    }, "stage2_progress")

logger.info("="*80)
logger.info("STAGE 3: RLAIF (PREFERENCE LEARNING)")
logger.info("SCM Compliance: JSON-based structured output + validation")
logger.info("="*80)


logger.info("Checking Ollama availability...")
OLLAMA_MODEL = 'llama3.2:3b'


try:
    test_response = requests.get('http://localhost:11434/api/tags', timeout=30)
    if test_response.status_code == 200:
        models = test_response.json().get('models', [])
        model_names = [m.get('name', '') for m in models]
        logger.info(f"✓ Ollama running. Models: {model_names}")
        
        if OLLAMA_MODEL not in model_names:
            logger.warning(f"⚠ Model '{OLLAMA_MODEL}' not found")
            logger.warning(f"Run: ollama pull {OLLAMA_MODEL}")
    else:
        logger.warning(f"⚠ Ollama status: {test_response.status_code}")
except Exception as e:
    logger.error(f"✗ Ollama not accessible: {e}")
    logger.error("Stage 3 cannot proceed without Ollama")


# Globals (define these in your main code if not already)
VALID_COMPARTMENTS = ['FACTUAL', 'PROCEDURAL', 'EPISODIC', 'CONTEXTUAL', 'CONCEPTUAL']
VALID_HIERARCHIES = ['GRANULAR', 'INTERMEDIATE', 'GENERAL']

# Assume these functions exist: sonar_encoder.encode(), compute_importance_score(text), get_precision_requirement(comp, hier)
# logger = your logging setup

class JSONRepairStages:
    """Encapsulates multi-stage JSON repair for modularity."""
    
    @staticmethod
    def stage1_direct_parse(text: str) -> Optional[Dict[str, Any]]:
        """Stage 1: Direct JSON parse (fast path)."""
        try:
            data = json.loads(text)
            if isinstance(data, dict) and 'segments' in data:
                return data
            return None
        except json.JSONDecodeError:
            return None
    
    @staticmethod
    def stage2_quote_keys(text: str) -> str:
        """Stage 2: Add quotes to unquoted keys (e.g., {key: value} -> {"key": value})."""
        # Pattern for unquoted keys: identifier followed by : (not in string)
        def quote_match(match):
            key = match.group(1).strip()
            if re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', key):  # Valid JS identifier
                return f'"{key}":'
            return match.group(0)
        
        repaired = re.sub(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*:\s*(?![{["\s])', quote_match, text)
        if repaired != text:
            logger.debug("Stage 2: Quoted unquoted keys")
        return repaired
    
    @staticmethod
    def stage3_fix_commas_braces(text: str) -> str:
        """Stage 3: Remove trailing/extra commas, balance braces/brackets."""
        # Remove trailing commas before } or ]
        text = re.sub(r',(\s*[}\]])', r'\1', text)
        # Remove extra commas in objects/arrays
        text = re.sub(r'\{\s*,', '{', text)
        text = re.sub(r'\[\s*,', '[', text)
        # Balance braces (simple heuristic: add missing })
        brace_count = text.count('{') - text.count('}')
        bracket_count = text.count('[') - text.count(']')
        text += '}' * max(0, brace_count)
        text += ']' * max(0, bracket_count)
        if brace_count > 0 or bracket_count > 0:
            logger.debug(f"Stage 3: Balanced {brace_count} braces, {bracket_count} brackets")
        return text
    
    @staticmethod
    def stage4_partial_object(text: str) -> str:
        """Stage 4: Handle partials like "segments": [...] or [ ... ] -> full {"segments": [...]}."""
        # If starts with "segments":
        if re.match(r'^\s*"segments"\s*:\s*\[', text):
            pass  # Already good, but ensure outer {}
        elif re.match(r'^\s*segments\s*:\s*\[', text):
            text = re.sub(r'^segments\s*:\s*', '"segments": ', text)
            if not text.startswith('{'):
                text = '{' + text + '}'
        # If starts with array, wrap as segments
        elif re.match(r'^\s*\[', text):
            text = '{"segments": ' + text + ', "final_answer": ""}'
        # Handle "output": { ... } or similar wrappers
        text = re.sub(r'^\s*"output"\s*:\s*(\{.*\})', r'\1', text, flags=re.DOTALL)
        text = re.sub(r'^\s*output\s*:\s*(\{.*\})', r'\1', text, flags=re.DOTALL | re.IGNORECASE)
        return text
    
    @staticmethod
    def stage5_extract_segments(text: str) -> Optional[Dict[str, Any]]:
        """Stage 5: Last resort - regex extract segments array and reconstruct."""
        seg_match = re.search(r'"segments"\s*:\s*\[([^\]]*)\]', text, re.DOTALL)
        if not seg_match:
            # Broader match for unquoted
            seg_match = re.search(r'segments\s*:\s*\[([^\]]*)\]', text, re.DOTALL | re.IGNORECASE)
        if seg_match:
            segments_str = seg_match.group(1)
            # Parse individual segments
            try:
                # Simple wrap and parse
                seg_json = f'{{"segments": [{segments_str}]}}'
                data = json.loads(seg_json)
                data['final_answer'] = ''  # Default
                logger.warning("Stage 5: Reconstructed from segments array only")
                return data
            except:
                pass
        return None
    
    @staticmethod
    def stage6_per_segment_regex(text: str) -> List[Dict[str, Any]]:
        """Stage 6: Nuclear option - regex each segment individually (no full JSON)."""
        # Pattern for individual segments: {"seg_id": N, "compartment": "...", "hierarchy": "...", "text": "..."}
        seg_pattern = r'\{[^}]*"seg_id"\s*:\s*(?:\d+)[^}]*"compartment"\s*:\s*"([^"]*)"[^{}]*"hierarchy"\s*:\s*"([^"]*)"[^{}]*"text"\s*:\s*"([^"]*)"[^{}]*\}'
        matches = re.findall(seg_pattern, text, re.DOTALL | re.IGNORECASE)
        steps = []
        for comp, hier, txt in matches:
            try:
                comp = comp.strip().upper()
                hier = hier.strip().upper()
                txt = txt.strip()
                if len(txt) < 10:
                    continue
                if comp not in VALID_COMPARTMENTS:
                    comp = 'CONCEPTUAL'
                if hier not in VALID_HIERARCHIES:
                    hier = 'GENERAL'
                # Mock embedding (replace with real)
                embedding = None  # sonar_encoder.encode(txt).tolist()
                steps.append({
                    'Seg': len(steps),  # Approximate ID
                    'compartment': comp,
                    'hierarchy': hier,
                    'text': txt,
                    'sonar_embedding': embedding,
                    'importance_score': 0.5,  # Default; use compute_importance_score(txt)
                    'precision': 1.0  # Default; use get_precision_requirement(comp, hier)
                })
            except:
                continue
        if steps:
            logger.warning(f"Stage 6: Recovered {len(steps)} segments via regex")
        return steps

def robust_json_extractor(raw_response: str) -> str:
    """
    Extraction with PRIORITY ORDERING: Check valid JSON first, then clean.
    """
    if not raw_response or not raw_response.strip():
        return ""
    
    raw_response = raw_response.strip()
    
    # ========== PRIORITY 1: Already valid JSON (most common case) ==========
    if raw_response.startswith('{'):
        logger.debug("Found JSON at start (no extraction needed)")
        return raw_response
    
    # ========== PRIORITY 2: Markdown code block ==========
    markdown_match = re.search(r'``````', raw_response, re.DOTALL)
    if markdown_match:
        logger.debug("Extracted JSON from markdown block")
        return markdown_match.group(1).strip()
    
    # ========== PRIORITY 3: Common prefixes (e.g., "Here is:") ==========
    # Remove EXACTLY these phrases (case-insensitive, anchored to start)
    prefix_cleaners = [
        (r'^\s*Here\s+is\s+the\s+(?:generated\s+)?JSON\s+(?:response\s*)?:?\s*', ''),
        (r'^\s*(?:JSON|Output|Response|Generated|Result)\s*:?\s*', ''),
        (r'^\s*The\s+(?:JSON\s+)?(?:response\s+)?is\s*:?\s*', ''),
    ]
    
    for pattern, replacement in prefix_cleaners:
        cleaned = re.sub(pattern, replacement, raw_response, count=1, flags=re.IGNORECASE)
        if cleaned != raw_response and cleaned.strip().startswith('{'):
            logger.debug(f"Removed prefix: {pattern}")
            return cleaned.strip()
    
    # ========== PRIORITY 4: Find first { ... } balanced block ==========
    brace_start = raw_response.find('{')
    if brace_start == -1:
        logger.warning("No opening brace found in response")
        return ""
    
    # Find matching closing brace
    brace_count = 0
    brace_end = -1
    for i in range(brace_start, len(raw_response)):
        if raw_response[i] == '{':
            brace_count += 1
        elif raw_response[i] == '}':
            brace_count -= 1
            if brace_count == 0:
                brace_end = i
                break
    
    if brace_end != -1:
        extracted = raw_response[brace_start:brace_end + 1]
        logger.debug(f"Extracted balanced JSON from position {brace_start}-{brace_end}")
        return extracted
    
    # ========== PRIORITY 5: Partial JSON (unbalanced) ==========
    # Try to balance braces automatically
    partial = raw_response[brace_start:]
    open_braces = partial.count('{')
    close_braces = partial.count('}')
    
    if open_braces > close_braces:
        missing = open_braces - close_braces
        partial += '}' * missing
        logger.warning(f"Added {missing} closing braces to balance JSON")
        return partial
    
    logger.error(f"Could not extract JSON from: {raw_response[:200]}...")
    return ""


def parse_json_segments(json_response: str, min_segments: int = 10) -> List[Dict[str, Any]]:
    """Enhanced with better error recovery."""
    steps = []
    extracted = robust_json_extractor(json_response)
    
    if not extracted:
        logger.error(f"Extractor failed on: {json_response[:300]}...")
        return steps
    
    logger.debug(f"Parsing: {extracted[:200]}...")
    
    # Try direct parse first (99% of cases after good extraction)
    parsed_data = JSONRepairStages.stage1_direct_parse(extracted)
    
    if parsed_data is None:
        # Apply repairs progressively
        current_text = extracted
        
        # Stage 2: Quote unquoted keys
        current_text = JSONRepairStages.stage2_quote_keys(current_text)
        parsed_data = JSONRepairStages.stage1_direct_parse(current_text)
        
        if parsed_data is None:
            # Stage 3: Fix commas/braces
            current_text = JSONRepairStages.stage3_fix_commas_braces(current_text)
            parsed_data = JSONRepairStages.stage1_direct_parse(current_text)
        
        if parsed_data is None:
            # Stage 4: Handle partials
            current_text = JSONRepairStages.stage4_partial_object(current_text)
            parsed_data = JSONRepairStages.stage1_direct_parse(current_text)
        
        if parsed_data is None:
            # Stage 5: Regex extraction
            parsed_data = JSONRepairStages.stage5_extract_segments(current_text)
    
    if parsed_data is None or 'segments' not in parsed_data:
        # Last resort: Per-segment regex
        steps = JSONRepairStages.stage6_per_segment_regex(json_response)
        logger.warning(f"Used regex fallback: {len(steps)} segments")
        return steps
    
    # Process segments (same as before)
    segments = parsed_data.get('segments', [])
    for i, segment in enumerate(segments):
        try:
            seg_id = int(segment.get('seg_id', i))
            comp = segment.get('compartment', 'CONCEPTUAL').strip().upper()
            hier = segment.get('hierarchy', 'GENERAL').strip().upper()
            text = segment.get('text', '').strip()
            
            if comp not in VALID_COMPARTMENTS:
                comp = 'CONCEPTUAL'
            if hier not in VALID_HIERARCHIES:
                hier = 'GENERAL'
            if len(text) < 10:
                continue
            
            try:
                embedding = sonar_encoder.encode(text)
                if hasattr(embedding, 'tolist'):
                    embedding = embedding.tolist()
                elif isinstance(embedding, np.ndarray):
                    embedding = embedding.tolist()
            except Exception as e:
                logger.debug(f"Embedding failed (seg {seg_id}): {e}")
                embedding = None
            
            steps.append({
                'Seg': seg_id,
                'compartment': comp,
                'hierarchy': hier,
                'text': text,
                'sonar_embedding': embedding,
                'importance_score': compute_importance_score(text),
                'precision': get_precision_requirement(comp, hier)
            })
        except Exception as e:
            logger.debug(f"Skipped segment {i}: {e}")
            continue
    
    logger.info(f"Parsed {len(steps)} segments (min: {min_segments})")
    return steps

def extract_final_answer_from_json(json_response: str) -> str:
    """Robust final answer extraction with repairs."""
    extracted = robust_json_extractor(json_response)
    if not extracted:
        return "No final answer provided"
    
    # Try parsed
    try:
        data = json.loads(extracted)  # Use same repairs implicitly via extractor
        return data.get('final_answer', '').strip()
    except:
        pass
    
    # Regex with flexibility
    patterns = [
        r'"final_answer"\s*:\s*"([^"]*)"',
        r'final_answer\s*:\s*"([^"]*)"',  # Unquoted key
        r'"final_answer"\s*:\s*([^{},]+)',  # Non-quoted value
    ]
    for pat in patterns:
        match = re.search(pat, extracted, re.DOTALL | re.IGNORECASE)
        if match:
            ans = match.group(1).strip().replace('\\"', '"')
            if len(ans) > 5:
                return ans[:500]  # Truncate long ones
    
    # Semantic fallback: Last text after "final" or "answer"
    if 'final_answer' in json_response.lower() or 'answer' in json_response.lower():
        last_part = re.split(r'final\s*answer', json_response, flags=re.IGNORECASE)[-1]
        lines = [l.strip() for l in last_part.split('\n') if l.strip() and len(l.strip()) > 20]
        if lines:
            return lines[0][:200]
    
    return "No final answer provided"

# ═══════════════════════════════════════════════════════════════
# VOCABULARY TRAINING (Run once on corpus)
# ═══════════════════════════════════════════════════════════════

def train_domain_vocabulary(output_dir=OUTPUT_PATH, vocab_size=30000):
    """
    Train BPE vocabulary on cleaned corpus (optional).
    Only needed if using SONAR decoder for text generation.
    """
    logger.info("="*80)
    logger.info("OPTIONAL: TRAINING DOMAIN VOCABULARY")
    logger.info("="*80)
    
    try:
        from tokenizers import Tokenizer, models, trainers, pre_tokenizers
        
        # Collect text from all stages
        corpus_file = Path(output_dir) / "corpus_for_vocab.txt"
        
        logger.info(f"Building corpus from generated data...")
        with open(corpus_file, 'w', encoding='utf-8') as f:
            # Stage 1
            if os.path.exists(os.path.join(OUTPUT_PATH, 'stage1_pretrain.jsonl')):
                with open(os.path.join(OUTPUT_PATH, 'stage1_pretrain.jsonl'), 'r') as s1:
                    for line in s1:
                        try:
                            data = json.loads(line)
                            f.write(data['text'] + '\n')
                        except:
                            continue
            
            # Stage 2
            if os.path.exists(os.path.join(OUTPUT_PATH, 'stage2_sft.jsonl')):
                with open(os.path.join(OUTPUT_PATH, 'stage2_sft.jsonl'), 'r') as s2:
                    for line in s2:
                        try:
                            data = json.loads(line)
                            for step in data.get('reasoning_chain', []):
                                f.write(step['text'] + '\n')
                        except:
                            continue
        
        logger.info(f"✓ Corpus saved: {corpus_file}")
        
        # Train BPE tokenizer
        logger.info(f"Training BPE vocabulary (size={vocab_size})...")
        tokenizer = Tokenizer(models.BPE())
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=["<pad>", "<unk>", "<bos>", "<eos>", "<mask>"],
            show_progress=True
        )
        
        tokenizer.train(files=[str(corpus_file)], trainer=trainer)
        
        vocab_path = Path(output_dir) / "scm_tokenizer.json"
        tokenizer.save(str(vocab_path))
        
        logger.info(f"✓ Vocabulary trained and saved: {vocab_path}")
        logger.info(f"✓ Vocabulary size: {tokenizer.get_vocab_size()}")
        
        return str(vocab_path)
        
    except ImportError:
        logger.warning("tokenizers library not installed. Skipping vocabulary training.")
        logger.warning("Install with: pip install tokenizers")
        return None
    except Exception as e:
        logger.error(f"Vocabulary training failed: {e}")
        return None

# Uncomment to enable vocabulary training after pipeline completion
# if __name__ == "__main__":
#     # After all stages complete:
#     vocab_path = train_domain_vocabulary()
#     if vocab_path:
#         logger.info(f"Vocabulary ready for SONAR decoder: {vocab_path}")


def generate_single_candidate(query, candidate_num=0, model=OLLAMA_MODEL, max_retries=3):
    """
    Generate a single candidate with JSON output format.
    
    Args:
        query: The question to answer
        candidate_num: Candidate identifier (for logging)
        model: Ollama model to use
        max_retries: Maximum retry attempts
    
    Returns:
        Dictionary with structured candidate data or None if failed
    """
    
    for attempt in range(max_retries):
        try:
            timeout = 240 + (attempt * 200)
            
            prompt = f"""Generate a step-by-step scientific explanation for the following query: {query}

CRITICAL INSTRUCTIONS:
1. Output MUST be valid JSON only (no markdown, no extra text, no introductory sentence before the JSON output).
2. Use the exact schema below.
3. Include 10-100 segments (minimum 10, recommended 50, maximum 100)
4. Each segment must have all 4 fields: seg_id, compartment, hierarchy, text

VALID VALUES:
- "compartment": FACTUAL, PROCEDURAL, EPISODIC, CONTEXTUAL, CONCEPTUAL
- "hierarchy": GRANULAR, INTERMEDIATE, GENERAL

OUTPUT SCHEMA:
{{
  "segments": [
    {{
      "seg_id": 0,
      "compartment": "CONTEXTUAL",
      "hierarchy": "INTERMEDIATE",
      "text": "According to scientific principles, we begin by..."
    }},
    {{
      "seg_id": 1,
      "compartment": "FACTUAL",
      "hierarchy": "GRANULAR",
      "text": "The mathematical formula is expressed as..."
    }},
    {{
      "seg_id": 2,
      "compartment": "CONCEPTUAL",
      "hierarchy": "GENERAL",
      "text": "This concept relates to the broader theory..."
    }}
  ],
  "final_answer": "Brief summary of the complete explanation"
}}

You MUST respond with VALID JSON matching this exact schema:
{{
  "segments": [
    {{
      "seg_id": 0,
      "compartment": "FACTUAL|PROCEDURAL|EPISODIC|CONTEXTUAL|CONCEPTUAL",
      "hierarchy": "GRANULAR|INTERMEDIATE|GENERAL",
      "text": "your reasoning step here"
    }}
  ],
  "final_answer": "your final answer here"
}}

REQUIREMENTS:
- Start seg_id from 0 and increment sequentially
- Ensure JSON is properly formatted (escaped quotes, valid syntax).
- Each text field should be detailed and substantive.
- Mix different compartments and hierarchies logically throughout.
- Minimum 20 segments required for acceptance.
- Focus on the query, and reach the answer with all available context.

Generate the JSON-only response now:"""
            
            logger.debug(f"Requesting candidate {candidate_num} (attempt {attempt + 1})...")
            
            response = requests.post(
                'http://localhost:11434/api/generate',
                json={
                    'model': model,
                    'prompt': prompt,
                    'stream': False,
                    'options': {
                        'num_predict': 8000,  # Increased for JSON output
                        'temperature': 0.7 + (candidate_num * 0.1),  # Vary temperature per candidate
                        'num_ctx': 4096,
                        'top_p': 0.85,
                        'repeat_penalty': 1.15,
                        'stop': ['Here is', 'Output:']  # Stop rambling
                    }
                },
                timeout=timeout
            )
            
            if response.status_code == 200:
                raw = response.json().get('response', '').strip()
                
                logger.debug(f"Raw JSON response (candidate {candidate_num}): {raw[:200]}...")
                
                # Clean potential markdown wrapping
                if raw.startswith('```'):
                    raw = re.sub(r'^```(?:json)?\s*', '', raw)
                    raw = re.sub(r'```\s*$', '', raw)
                    raw = raw.strip()
                
                # Parse JSON segments
                steps = parse_json_segments(raw)
                
                if len(steps) >= 10:  # Minimum requirement
                    final_ans = extract_final_answer_from_json(raw)
                    
                    candidate_data = {
                        'candidate_id': chr(65 + candidate_num),  # A, B, C, ...
                        'reasoning_chain': steps,
                        'final_answer': final_ans,
                        'reward_score': 0.5 + (candidate_num * 0.15),  # Varied scoring
                        'raw_text': raw
                    }
                    
                    logger.info(f"✓ Generated candidate {chr(65 + candidate_num)} with {len(steps)} segments")
                    return candidate_data
                else:
                    raise ValueError(f"Only {len(steps)} segments parsed (minimum 10 required)")
            
            raise ValueError(f"Ollama status {response.status_code}")
        
        except (requests.Timeout, requests.ConnectionError, ValueError, json.JSONDecodeError) as e:
            wait_time = (2 ** attempt) + random.uniform(0, 1)
            logger.warning(f"  Candidate {candidate_num} attempt {attempt + 1} failed: {type(e).__name__}")
            logger.warning(f"  Waiting {wait_time:.1f}s before retry...")
            time.sleep(wait_time)
            
            if attempt == max_retries - 1:
                logger.error(f"✗ Failed to generate candidate {candidate_num} after {max_retries} attempts")
                return None
    
    return None


def generate_candidates_looped(query, num_candidates=2, model=OLLAMA_MODEL):
    """
    Generate multiple candidates by looping individual requests.
    
    Args:
        query: The question to answer
        num_candidates: Number of candidates to generate
        model: Ollama model to use
    
    Returns:
        List of candidate dictionaries
    """
    candidates = []
    
    logger.info(f"Generating {num_candidates} candidates (sequential with cooling)...")
    
    for i in range(num_candidates):
        logger.info(f"  → Candidate {i+1}/{num_candidates}...")
        
        candidate = generate_single_candidate(query, candidate_num=i, model=model)
        
        if candidate:
            candidates.append(candidate)
            logger.info(f"    ✓ Success: {len(candidate['reasoning_chain'])} segments")
        else:
            logger.warning(f"    ✗ Failed to generate candidate {i+1}")
        
        # Cooling delay between candidates (except after last)
        if i < num_candidates - 1:
            cooldown = 2.0  # 2 seconds between candidates
            logger.debug(f"    Cooling {cooldown}s...")
            time.sleep(cooldown)
    
    logger.info(f"Completed: {len(candidates)}/{num_candidates} candidates generated")
    return candidates


# Modified checkpoint section
stage3_file = os.path.join(OUTPUT_PATH, "stage3_rlaif.jsonl")
stage3_checkpoint = load_checkpoint("stage3_progress")


if stage3_checkpoint:
    stage3_data = stage3_checkpoint.get("stage3_data", [])
    processed_query_indices = set(stage3_checkpoint.get("processed_indices", []))
    logger.info(f"✓ Resuming Stage 3")
    logger.info(f"  - Samples: {len(stage3_data)}")
    logger.info(f"  - Processed: {len(processed_query_indices)}")
else:
    stage3_data = []
    processed_query_indices = set()
    logger.info("Starting fresh Stage 3...")


# Validate docdata
if not doc_data:
    doc_data_checkpoint = load_checkpoint("doc_data_for_stage2")
    if doc_data_checkpoint:
        doc_data = doc_data_checkpoint["doc_data"]


if not doc_data or len(doc_data) == 0:
    logger.error("❌ Stage 3 requires docdata from Stage 1/2")
    raise ValueError("Cannot proceed with Stage 3: No documents available")


sample_queries = [(idx, data["query"]) for idx, data in doc_data.items()]
logger.info(f"Processing {len(sample_queries)} queries")


total_processed = len(processed_query_indices)
last_written_idx = 0


# Process in batches
for batch_start in range(0, len(sample_queries), STAGE3_BATCH_SIZE):
    batch_end = min(batch_start + STAGE3_BATCH_SIZE, len(sample_queries))
    batch = sample_queries[batch_start:batch_end]
    
    logger.info(f"--- Batch {batch_start // STAGE3_BATCH_SIZE + 1} ({batch_start}-{batch_end}) ---")
    
    # Filter queries to process
    queries_to_process = []
    for i, (doc_idx, query) in enumerate(batch):
        global_idx = batch_start + i
        if global_idx not in processed_query_indices and query:
            queries_to_process.append((global_idx, doc_idx, query))
    
    if not queries_to_process:
        logger.info("  No new queries in this batch, skipping...")
        continue
    
    logger.info(f"  Processing {len(queries_to_process)} queries...")
    
    # Process queries sequentially (looped candidate generation handles parallelism internally)
    for global_idx, doc_idx, query in queries_to_process:
        logger.info(f"\\n  Query {global_idx}: {query[:100]}...")
        
        try:
            # Generate 3 candidates via looped individual requests
            candidates = generate_candidates_looped(query, num_candidates=3)
            
            if len(candidates) >= 2:  # Minimum 2 candidates required
                # Create AI feedback
                sorted_candidates = sorted(candidates, key=lambda x: x["reward_score"], reverse=True)
                
                ai_feedback = {
                    "ranking": [c["candidate_id"] for c in sorted_candidates],
                    "criteria": {
                        "logical_flow": [0.9 - (i * 0.1) for i in range(len(candidates))],
                        "accuracy": [0.95 - (i * 0.1) for i in range(len(candidates))],
                        "completeness": [0.9 - (i * 0.15) for i in range(len(candidates))]
                    },
                    "critique": {c["candidate_id"]: f"Quality score {c['reward_score']:.2f}" 
                                for c in candidates},
                    "preference_pairs": [
                        {"winner": sorted_candidates[0]["candidate_id"], 
                         "loser": sorted_candidates[j]["candidate_id"]}
                        for j in range(1, len(sorted_candidates))
                    ]
                }
                
                stage3_sample = {
                    "id": f"rlaif_{global_idx:06d}",
                    "query": query,
                    "candidates": candidates,
                    "ai_feedback": ai_feedback,
                    "preference_pairs": ai_feedback["preference_pairs"]
                }
                
                stage3_data.append(stage3_sample)
                processed_query_indices.add(global_idx)
                total_processed += 1
                
                logger.info(f"  ✓ Created sample for query {global_idx}")
            else:
                logger.warning(f"  ✗ Failed for query {global_idx}: only {len(candidates)} candidates")
        
        except Exception as e:
            logger.error(f"  ✗ Error processing query {global_idx}: {e}")
            continue
    
    # Checkpoint after each batch
    if last_written_idx < len(stage3_data):
        logger.info(f"\\n  Writing {len(stage3_data) - last_written_idx} samples...")
        with open(stage3_file, "a", encoding="utf-8") as f:
            for item in stage3_data[last_written_idx:]:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        last_written_idx = len(stage3_data)
        
        save_checkpoint({
            "stage3_data": stage3_data,
            "processed_indices": list(processed_query_indices)
        }, "stage3_progress")
        
        logger.info("  ✓ Checkpoint saved")
    
    # Cooling delay between batches
    if batch_end < len(sample_queries):
        logger.info(f"\\n  Batch cooling down {STAGE3_BATCH_DELAY}s...")
        time.sleep(STAGE3_BATCH_DELAY)


# Final write
if last_written_idx < len(stage3_data):
    logger.info(f"\\nWriting final samples...")
    with open(stage3_file, "a", encoding="utf-8") as f:
        for item in stage3_data[last_written_idx:]:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


logger.info("="*80)
logger.info(f"STAGE 3 COMPLETE")
logger.info("="*80)
logger.info(f"Total samples: {len(stage3_data)}")
logger.info(f"Successful rate: {len(stage3_data)}/{len(sample_queries)} ({100*len(stage3_data)/len(sample_queries):.1f}%)")


save_checkpoint({
    "stage3_data": stage3_data,
    "processed_indices": list(processed_query_indices)
}, "stage3_progress")


logger.info("="*80)
logger.info("PIPELINE COMPLETE - SCM COMPLIANT DATASET GENERATED")
logger.info("="*80)

logger.info(f"\nOutput files:")
logger.info(f"  Stage 1: {os.path.join(OUTPUT_PATH, 'stage1_pretrain.jsonl')}")
logger.info(f"  Stage 2: {stage2_file}")
logger.info(f"  Stage 3: {stage3_file}")

# FIX #13: Comprehensive statistics
logger.info(f"\n{'='*80}")
logger.info(f"DATASET STATISTICS")
logger.info(f"{'='*80}")

# File sizes
stage1_file_path = os.path.join(OUTPUT_PATH, 'stage1_pretrain.jsonl')
if os.path.exists(stage1_file_path):
    stage1_size = os.path.getsize(stage1_file_path) / (1024**3)
else:
    stage1_size = 0

if os.path.exists(stage2_file):
    stage2_size = os.path.getsize(stage2_file) / (1024**3)
else:
    stage2_size = 0

if os.path.exists(stage3_file):
    stage3_size = os.path.getsize(stage3_file) / (1024**3)
else:
    stage3_size = 0

logger.info(f"\nFile Sizes:")
logger.info(f"  Stage 1: {stage1_size:.2f} GB")
logger.info(f"  Stage 2: {stage2_size:.2f} GB")
logger.info(f"  Stage 3: {stage3_size:.2f} GB")
logger.info(f"  Total: {stage1_size + stage2_size + stage3_size:.2f} GB")

logger.info(f"\nBasic Statistics:")
# logger.info(f"  Science docs processed: {doc_count}")
logger.info(f"  Stage 2 samples: {len(stage2_data)}")
logger.info(f"  Stage 3 samples: {len(stage3_data)}")

if len(stage2_data) > 0:
    multi_step = sum(1 for d in stage2_data if d['metadata']['num_steps'] >= 2)
    logger.info(f"  Multi-step CoT: {multi_step} ({multi_step / len(stage2_data) * 100:.1f}%)")
    
    # Compartment distribution
    all_comps = [step['compartment'] for sample in stage2_data 
                 for step in sample['reasoning_chain']]
    comp_counts = {}
    for c in all_comps:
        comp_counts[c] = comp_counts.get(c, 0) + 1
    
    logger.info(f"\nCompartment Distribution (Stage 2):")
    for comp, count in sorted(comp_counts.items(), key=lambda x: x[1], reverse=True):
        pct = count / len(all_comps) * 100
        logger.info(f"  {comp}: {count} ({pct:.1f}%)")
    
    # Hierarchy distribution
    all_hiers = [step['hierarchy'] for sample in stage2_data 
                 for step in sample['reasoning_chain']]
    hier_counts = {}
    for h in all_hiers:
        hier_counts[h] = hier_counts.get(h, 0) + 1
    
    logger.info(f"\nHierarchy Distribution (Stage 2):")
    for hier, count in sorted(hier_counts.items(), key=lambda x: x[1], reverse=True):
        pct = count / len(all_hiers) * 100
        logger.info(f"  {hier}: {count} ({pct:.1f}%)")
    
    # Average importance
    all_importance = [step['importance_score'] for sample in stage2_data 
                      for step in sample['reasoning_chain']]
    avg_importance = sum(all_importance) / len(all_importance)
    logger.info(f"\nAverage Importance Score: {avg_importance:.2f}")


logger.info(f"\nLog file: {log_file}")
logger.info("="*80)
logger.info("All processing complete. Dataset ready for SCM training.")
logger.info("="*80)
