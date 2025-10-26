# C:\Users\Khwaish\Google Drive Streaming

# !nohup ollama serve &
import time
import requests

import nltk
nltk.download('punkt_tab')

nltk.download('punkt', download_dir='C:/Users/Khwaish/Google Drive Streaming/My Drive/scm_project/cache/nltk')
nltk.data.path.append('C:/Users/Khwaish/Google Drive Streaming/My Drive/scm_project/cache/nltk')

# ===================================================================
# Process OpenThoughts3-1.2M Dataset (Fixed Loading)
# ===================================================================
import re  # Added for regex in technical terms
import os
import json
import nltk
from datasets import load_dataset, Dataset
from tqdm import tqdm
import logging

from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize  # Added: Missing import for sentence tokenization

# Configure HF cache to use your Drive
os.environ['HF_DATASETS_CACHE'] = 'C:/Users/Khwaish/Google Drive Streaming/My Drive/scm_project/cache/datasets'
os.environ['HF_DATASETS_OFFLINE'] = '0'  # Allow verification but use cache

# Output path
OUTPUT_PATH = "C:/Users/Khwaish/Google Drive Streaming/My Drive/scm_project/datasets/processed"
os.makedirs(OUTPUT_PATH, exist_ok=True)

# NLTK setup
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

# Enable logging for dataset loading
logging.basicConfig(level=logging.INFO)

# ===================================================================
# CHECKPOINT SYSTEM
# ===================================================================
import pickle
from pathlib import Path

CHECKPOINT_DIR = "C:/Users/Khwaish/Google Drive Streaming/My Drive/scm_project/checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def save_checkpoint(data, checkpoint_name):
    """Save checkpoint to Google Drive."""
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"{checkpoint_name}.pkl")
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"✓ Checkpoint saved: {checkpoint_path}")

def load_checkpoint(checkpoint_name):
    """Load checkpoint from Google Drive."""
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"{checkpoint_name}.pkl")
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, 'rb') as f:
                data = pickle.load(f)
            print(f"✓ Checkpoint loaded: {checkpoint_path}")
            return data
        except (EOFError, pickle.UnpicklingError, ValueError) as e:
            print(f"⚠ Corrupted checkpoint {checkpoint_path}: {e}. Deleting and starting fresh.")
            os.remove(checkpoint_path)  # Auto-delete bad file
            return None
    return None


def checkpoint_exists(checkpoint_name):
    """Check if checkpoint exists."""
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"{checkpoint_name}.pkl")
    return os.path.exists(checkpoint_path)

# Load SONAR proxy (multilingual for science robustness)
try:
    sonar_encoder = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
except Exception as e:
    print(f"Error loading SentenceTransformer: {e}")
    print("Please ensure sentence-transformers is installed correctly.")
    sonar_encoder = None # Set to None if loading fails



def extract_technical_terms(text):
    """Identify key science terms."""
    tech_vocab = [
        'transformer', 'neural network', 'gradient descent', 'loss function', 'embedding',
        'algorithm', 'theorem', 'equation', 'derivative', 'integral', 'probability',
        'molecule', 'atom', 'ion', 'chemical', 'bond', 'electron', 'proton', 'nucleus',
        'protein', 'amino acid', 'dna', 'rna', 'genome', 'cell', 'gene', 'chromosome',
        'force', 'energy', 'momentum', 'velocity', 'acceleration', 'gravity', 'electromagnetic',
        'electric', 'magnetic', 'charge', 'current', 'voltage', 'wave', 'frequency', 'wavelength',
        'quantum', 'photon', 'relativity', 'thermodynamics', 'entropy', 'kinetic', 'potential',
        'geology', 'fossil', 'tectonic', 'volcano', 'atmosphere', 'climate', 'neuron', 'synapse', 'brain', 'sci'
    ]
    text_lower = text.lower()
    # Use regex to find whole words or phrases
    found_terms = []
    for term in tech_vocab:
        # Create a regex pattern that matches the whole word or phrase
        pattern = r'\b' + re.escape(term) + r'\b'
        if re.search(pattern, text_lower):
            found_terms.append(term)
    return found_terms

def extract_and_segment(example, idx):
    """Extract from conversations, segment into sentences, assign labels (FIXED)."""
    full_text = extract_text_from_conversations(example)
    if not full_text or len(full_text) < 20:
        return []

    # Segment into sentences
    sentences = sent_tokenize(full_text[:5000])  # Limit for efficiency

    segments = []
    query = ""  # Store original query if available
    conversations = example.get('conversations', [])
    for conv in conversations:
        if 'user' in str(conv.get('from', '')).lower():
            query = conv.get('value', '')
            break

    for i, sent in enumerate(sentences):
        if len(sent.strip()) < 15:
            continue

        compartment = estimate_compartment_advanced(sent)
        hierarchy = determine_hierarchy(sent)

        segments.append({
            'id': f"{idx}_seg{i}",
            'text': sent.strip(),
            'compartment': compartment,
            'hierarchy': hierarchy,
            'position': i,
            'query': query if i == 0 else "",  # Include query in first segment
            'domain': example.get('domain', 'science'),
            'source': example.get('source', 'OpenThoughts3'),
            'difficulty': example.get('difficulty', 0)
        })

    return segments

def extract_text_from_conversations(example):
    """Extract clean text from conversations list."""
    conversations = example.get('conversations', [])
    texts = [conv.get('value', '') for conv in conversations if isinstance(conv, dict) and 'value' in conv]
    return ' '.join(texts).strip() if texts else example.get('text', '')  # Fallback to 'text' field if no conversations

def estimate_compartment_advanced(text):
    """Rule-based compartment labeling with science-focused patterns."""
    text_lower = text.lower()

    factual_indicators = ['is', 'are', 'was', 'were', 'defined as', 'equals', 'known as', 'theorem', 'law', 'formula', 'equation']
    procedural_indicators = ['step', 'first', 'then', 'algorithm', 'calculate', 'solve', 'implement', 'process']
    episodic_indicators = ['discovered', 'developed by', 'introduced', 'published', 'historical']
    contextual_indicators = ['important', 'application', 'example', 'advantage']
    conceptual_indicators = ['therefore', 'thus', 'concept', 'theory', 'implies']

    scores = {
        'FACTUAL': sum(1 for p in factual_indicators if p in text_lower),
        'PROCEDURAL': sum(1 for p in procedural_indicators if p in text_lower),
        'EPISODIC': sum(1 for p in episodic_indicators if p in text_lower),
        'CONTEXTUAL': sum(1 for p in contextual_indicators if p in text_lower),
        'CONCEPTUAL': sum(1 for p in conceptual_indicators if p in text_lower)
    }

    max_comp = max(scores, key=scores.get)
    return max_comp if scores[max_comp] > 0 else 'CONCEPTUAL'

def determine_hierarchy(text):
    """Hierarchy based on length and technical density."""
    length = len(text)
    tech_terms = ['neural', 'gradient', 'algorithm', 'equation', 'derivative', 'probability', 'matrix']
    tech_density = sum(1 for term in tech_terms if term in text.lower()) / max(len(text.split()), 1)

    if length < 50 or tech_density > 0.1:
        return 'GRANULAR'
    elif length < 150 or tech_density > 0.05:
        return 'INTERMEDIATE'
    return 'GENERAL'


def process_to_stage1(example, idx):
    """Generate Stage 1 samples from one example."""
    full_text = extract_text_from_conversations(example)
    if not full_text or len(full_text) < 20:
        return []

    sentences = sent_tokenize(full_text[:5000])  # Limit for efficiency
    samples = []

    for i, sent in enumerate(sentences):
        if len(sent) < 15:
            continue

        compartment = estimate_compartment_advanced(sent)
        hierarchy = determine_hierarchy(sent)
        tech_terms = extract_technical_terms(sent)

        # Encode embedding (batch if possible, but single for streaming)
        embedding = None
        if sonar_encoder:
            try:
                embedding = sonar_encoder.encode(sent).tolist()
            except Exception:
                # Skip if encoding fails
                continue
        else:
            # Skip if encoder failed to load
            continue

        precision = "fp32" if compartment in ['FACTUAL', 'PROCEDURAL'] else "fp16"

        sample = {
            "id": f"pretrain_{idx:06d}_{i}",
            "text": sent.strip(),
            "sonar_embedding": embedding,  # 768-dim for mpnet-base
            "compartment": compartment,
            "hierarchical_level": hierarchy.lower(),
            "metadata": {
                "domain": example.get('domain', 'science'),
                "source": 'OpenThoughts3',
                "difficulty": example.get('difficulty', 0),
                "technical_terms": tech_terms,
                "precision_required": precision,
                "fragility_score": None  # To be computed in training
            },
            "diffusion_config": {
                "noise_schedule": "cosine",
                "sigma_min": 0.02,
                "sigma_max": 0.5,
                "corruption_prob": 0.5
            }
        }
        samples.append(sample)

    return samples

# Load cached OpenThoughts3 dataset (streaming to handle size)
print("Loading OpenThoughts3-1.2M dataset...")
try:
    # Load directly from HF - uses your cache automatically
    dataset = load_dataset("open-thoughts/OpenThoughts3-1.2M", split="train", streaming=True)
    print(f"Loaded dataset in streaming mode.")

    # Verify structure by peeking at the first example
    try:
        first_example = next(iter(dataset))
        print("Dataset structure (first example):")
        print(f"  Sample keys: {list(first_example.keys())}")
        print(f"  Sample conversations length: {len(first_example.get('conversations', []))}")
    except StopIteration:
        print("Dataset is empty.")


except Exception as e:
    print(f"Error loading dataset: {e}")
    print("Make sure the dataset is properly cached and you have the 'datasets' library installed.")
    raise

# ===================================================================
# COMPREHENSIVE SCIENCE FILTER (85 keywords)
# ===================================================================

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

import re # Import re for regex in extract_technical_terms

def is_science_content(example):
    """Enhanced filter - checks domain, source, and content."""

    # Priority 1: Check domain field (most reliable)
    domain = str(example.get('domain', '')).lower()
    if 'science' in domain or 'sci' in domain:
        return True

    # Priority 2: Check source field
    source = str(example.get('source', '')).lower()
    for kw in ['chembio', 'chemistry', 'biology', 'physics', 'gpqa', 'jeebench', 'scibench']:
        if kw in source:
            return True

    # Priority 3: Check conversations content
    conversations = example.get('conversations', [])
    if conversations:
        full_text = ' '.join([str(c.get('value', '')).lower() for c in conversations])
        matches = sum(1 for kw in SCIENCE_KEYWORDS if kw in full_text)
        if matches >= 3:  # 3+ science keywords = likely science
            return True

    return False

print("\nFiltering for science content (streaming)...")
# Filtering directly on streaming dataset - this will process as we iterate later
science_dataset_stream = dataset.filter(is_science_content)
print("Science filter applied in streaming mode.")


# # ===================================================================
# # EXTRACT AND SEGMENT
# # ===================================================================

# print("\nExtracting and segmenting (streaming)...")
# all_segments = []
# processed_docs_counter = 0
# total_segments_counter = 0
# MAX_DOCS_TO_PROCESS = 10000 # Limit for testing/development

# for idx, example in enumerate(tqdm(science_dataset_stream.take(MAX_DOCS_TO_PROCESS))): # Process limited docs
#     segments = extract_and_segment(example, idx)
#     if segments:
#         all_segments.extend(segments)

#     processed_docs_counter += 1
#     if processed_docs_counter % 1000 == 0:
#          print(f"Processed {processed_docs_counter} documents.")


# total_segments_counter = len(all_segments)
# print(f"Total segments extracted from {processed_docs_counter} documents: {total_segments_counter}")

# # Convert to a Dataset for easier processing if needed, or continue with list
# # science_segments_dataset = Dataset.from_list(all_segments)


# # ===================================================================
# # CREATE STAGE 1: PRE-TRAINING
# # ===================================================================
# # This part now operates on the collected all_segments list

# print("\nCreating Stage 1...")

# stage1_data = []
# total_stage1_samples = 0
# MAX_STAGE1_SAMPLES = 50000 # Cap for Stage 1 samples

# for seg in tqdm(all_segments, desc="Processing segments for Stage 1"):
#     if total_stage1_samples >= MAX_STAGE1_SAMPLES:
#         break

#     # Encode embedding
#     embedding = None
#     if sonar_encoder:
#         try:
#             embedding = sonar_encoder.encode(seg['text']).tolist()
#         except Exception:
#             # Skip if encoding fails
#             continue
#     else:
#         # Skip if encoder failed to load
#         continue

#     precision = "fp32" if seg['compartment'] in ['FACTUAL', 'PROCEDURAL'] else "fp16"

#     sample = {
#         "id": seg['id'],
#         "text": seg['text'].strip(),
#         "sonar_embedding": embedding,
#         "compartment": seg['compartment'],
#         "hierarchical_level": seg['hierarchy'].lower(),
#         "metadata": {
#             "domain": seg['domain'],
#             "source": seg['source'],
#             "difficulty": seg['difficulty'],
#             "technical_terms": extract_technical_terms(seg['text']), # Extract terms here
#             "precision_required": precision,
#             "fragility_score": None  # To be computed in training
#         },
#         "diffusion_config": {
#             "noise_schedule": "cosine",
#             "sigma_min": 0.02,
#             "sigma_max": 0.5,
#             "corruption_prob": 0.5
#         }
#     }
#     stage1_data.append(sample)
#     total_stage1_samples += 1


# output_stage1 = "/content/drive/MyDrive/scm_project/datasets/stage1_pretrain.jsonl"
# with open(output_stage1, 'w', encoding='utf-8') as f:
#     for item in stage1_data:
#         f.write(json.dumps(item, ensure_ascii=False) + '\n')

# print(f"Stage 1: {len(stage1_data)} samples written to {output_stage1}")
# -------
# Save Stage 1 (append if resuming)
# stage1_file = f"{OUTPUT_PATH}/stage1_pretrain.jsonl"

# if os.path.exists(stage1_file):
#     print(f"Stage 1 file already exists: {stage1_file}")
#     existing_count = sum(1 for _ in open(stage1_file, 'r', encoding='utf-8'))
#     print(f"Existing Stage 1 samples: {existing_count}")

#     # Append new samples if any (from resume)
#     if stage1_data:  # Only if we added more during resume
#         with open(stage1_file, 'a', encoding='utf-8') as f:
#             for item in stage1_data:  # No cap, as existing already capped
#                 f.write(json.dumps(item, ensure_ascii=False) + '\n')
#         print(f"Appended {len(stage1_data)} new Stage 1 samples.")
#     total_stage1 = existing_count + len(stage1_data)
# else:
#     with open(stage1_file, 'w', encoding='utf-8') as f:
#         for item in stage1_data[:500000]:  # Cap fresh run
#             f.write(json.dumps(item, ensure_ascii=False) + '\n')
#     total_stage1 = len(stage1_data)
#     print(f"Stage 1: {total_stage1} samples → {stage1_file}")

# ===================================================================
# FIXED EXTRACTION & GROUPING (Per-Document Processing with Checkpointing)
# ===================================================================

print("\nExtracting & grouping (50k docs max)...")
MAX_DOCS = 5000
CHECKPOINT_INTERVAL = 50  # Save every 50 docs

# Try to load existing checkpoint
checkpoint_data = load_checkpoint("stage1_processing")
if checkpoint_data:
    doc_data = checkpoint_data['doc_data']
    stage1_data = checkpoint_data['stage1_data']
    doc_count = checkpoint_data['doc_count']
    science_count = checkpoint_data['science_count']
    start_idx = checkpoint_data['last_idx'] + 1
    print(f"Resuming from doc {start_idx} (processed: {doc_count} docs, {len(stage1_data)} Stage 1 samples)")
else:
    doc_data = {}
    stage1_data = []
    doc_count = 0
    science_count = 0
    start_idx = 0
    print("Starting fresh processing...")

print(f"\nProcessing up to {MAX_DOCS} docs with science filter...")
for idx, example in enumerate(tqdm(science_dataset_stream.take(MAX_DOCS), total=MAX_DOCS)):
    # Skip already processed docs
    if idx < start_idx:
        continue

    if doc_count >= MAX_DOCS:
        break

    science_count += 1
    full_text = extract_text_from_conversations(example)
    if not full_text or len(full_text) < 20:
        continue

    # Extract query
    query = ""
    conversations = example.get('conversations', [])
    for conv in conversations:
        if 'user' in str(conv.get('from', '')).lower():
            query = conv.get('value', '')
            break

    # Segment
    sentences = sent_tokenize(full_text[:5000])
    segments = []
    for i, sent in enumerate(sentences):
        if len(sent.strip()) < 15:
            continue
        comp = estimate_compartment_advanced(sent)
        hier = determine_hierarchy(sent)
        segments.append({
            'text': sent.strip(),
            'compartment': comp,
            'hierarchy': hier,
            'position': i
        })

    if segments:
        doc_data[idx] = {
            'segments': segments,
            'query': query or full_text[:200],
            'domain': example.get('domain', 'science'),
            'source': example.get('source', 'OpenThoughts3'),
            'difficulty': example.get('difficulty', 0)
        }

        # Add to Stage 1 (with embedding)
        for seg in segments:
            embedding = None
            if sonar_encoder:
                try:
                    embedding = sonar_encoder.encode(seg['text']).tolist()
                except:
                    continue
            if not embedding:
                continue

            tech_terms = extract_technical_terms(seg['text'])
            precision = "fp32" if seg['compartment'] in ['FACTUAL', 'PROCEDURAL'] else "fp16"

            stage1_data.append({
                "id": f"pretrain_{idx:06d}_{seg['position']}",
                "text": seg['text'],
                "sonar_embedding": embedding,
                "compartment": seg['compartment'],
                "hierarchical_level": seg['hierarchy'].lower(),
                "metadata": {
                    "domain": doc_data[idx]['domain'],
                    "source": doc_data[idx]['source'],
                    "technical_terms": tech_terms,
                    "precision_required": precision,
                    "fragility_score": None
                },
                "diffusion_config": {
                    "noise_schedule": "cosine",
                    "sigma_min": 0.02,
                    "sigma_max": 0.5,
                    "corruption_prob": 0.5
                }
            })

    doc_count += 1

    # Save checkpoint every CHECKPOINT_INTERVAL docs
    if doc_count % CHECKPOINT_INTERVAL == 0:
        save_checkpoint({
            'doc_data': doc_data,
            'stage1_data': stage1_data,
            'doc_count': doc_count,
            'science_count': science_count,
            'last_idx': idx
        }, "stage1_processing")
        print(f"Processed {doc_count} docs, {len(stage1_data)} Stage 1 samples so far.")

# Final checkpoint after loop completes
save_checkpoint({
    'doc_data': doc_data,
    'stage1_data': stage1_data,
    'doc_count': doc_count,
    'science_count': science_count,
    'last_idx': idx if 'idx' in locals() else 0
}, "stage1_processing")

print(f"\nProcessed {doc_count} science docs → {len(doc_data)} with segments.")

# Save Stage 1 (cap at 500k if needed)
stage1_file = f"{OUTPUT_PATH}/stage1_pretrain.jsonl"

# Check if Stage 1 already saved
if os.path.exists(stage1_file):
    print(f"Stage 1 file already exists: {stage1_file}")
    # Count existing lines
    with open(stage1_file, 'r', encoding='utf-8') as f:
        existing_count = sum(1 for _ in f)
    print(f"Existing Stage 1 samples: {existing_count}")
else:
    with open(stage1_file, 'w', encoding='utf-8') as f:
        for item in stage1_data[:500000]:  # Cap
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"Stage 1: {len(stage1_data)} samples → {stage1_file}")

# Save doc_data for Stage 2
save_checkpoint({'doc_data': doc_data}, "doc_data_for_stage2")

def reconstruct_doc_data_from_stage1(stage1_file):
    """Rebuild doc_data dict from stage1_pretrain.jsonl (fallback if pickle missing)."""
    doc_data = {}
    if not os.path.exists(stage1_file):
        return {}
    print("Reconstructing doc_data...")
    with open(stage1_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line.strip())
                if not item or 'id' not in item:
                    continue
                parts = item['id'].split('_')  # e.g., pretrain_000123_4 → idx='000123', pos=4
                if len(parts) < 3:
                    continue
                doc_id = parts[1]  # Padded idx like '000123'
                pos = int(parts[2])
                if doc_id not in doc_data:
                    doc_data[doc_id] = {'segments': [], 'query': '', 'domain': item['metadata']['domain'], 'source': item['metadata']['source'], 'difficulty': item['metadata']['difficulty']}
                seg = {
                    'text': item['text'],
                    'compartment': item['compartment'],
                    'hierarchy': item['hierarchical_level'].upper(),  # Match case
                    'position': pos
                }
                doc_data[doc_id]['segments'].append(seg)
                # Approx query from first seg text if empty
                if not doc_data[doc_id]['query'] and pos == 0:
                    doc_data[doc_id]['query'] = item['text'][:200]
            except (json.JSONDecodeError, IndexError, KeyError):
                continue  # Skip bad lines
    # Sort segments by position
    for d_id, data in doc_data.items():
        data['segments'].sort(key=lambda s: s['position'])
    print(f"Reconstructed {len(doc_data)} docs from {sum(1 for _ in open(stage1_file, 'r'))} Stage 1 samples.")
    return doc_data

# ===================================================================
# STAGE 2: SFT (Per-Document CoT with Checkpointing)
# ===================================================================

STAGE2_CHECKPOINT_INTERVAL = 20
print("\nCreating Stage 2...")
stage2_file = f"{OUTPUT_PATH}/stage2_sft.jsonl"
append_mode = False
stage2_data = []
processed_doc_ids = set()

existing_stage2_count = 0
if os.path.exists(stage2_file):
    with open(stage2_file, 'r', encoding='utf-8') as f:
        existing_stage2_count = sum(1 for _ in f)
    print(f"Stage 2 JSONL exists: {existing_stage2_count} samples (will append).")
    append_mode = True
else:
    print("Creating new Stage 2 JSONL.")
    stage2_checkpoint = load_checkpoint("stage2_progress")
    if stage2_checkpoint:
        stage2_data = stage2_checkpoint['stage2_data']
        processed_doc_ids = set(stage2_checkpoint['processed_doc_ids'])
        print(f"Resuming Stage 2 from checkpoint ({len(stage2_data)} samples completed)")
    else:
        stage2_data = []
        processed_doc_ids = set()
        print("Starting Stage 2 fresh...")

    if 'doc_data' not in locals() or not doc_data:
        doc_data_checkpoint = load_checkpoint("doc_data_for_stage2")
        if doc_data_checkpoint and 'doc_data' in doc_data_checkpoint:
            doc_data = doc_data_checkpoint['doc_data']
        else:
            doc_data = reconstruct_doc_data_from_stage1(stage1_file)
            if not doc_data:
                raise ValueError("Cannot reconstruct doc_data. Ensure Stage 1 JSONL exists.")

    if stage2_checkpoint:
        processed_count = len(processed_doc_ids)

    pbar = tqdm(doc_data.items(), desc="Creating Stage 2")
    for i, (idx, data) in enumerate(pbar):
        # Skip already processed
        if idx in processed_doc_ids:
            processed_count += 1  # Count skips for progress
            continue

        steps = []
        for seg in data['segments']:
            step_num = seg['position'] + 1
            precision = "fp32" if seg['compartment'] in ['FACTUAL', 'PROCEDURAL'] else "fp16"
            steps.append({
                "step": step_num,
                "compartment": seg['compartment'],
                "hierarchy": seg['hierarchy'],
                "text": seg['text'],
                "precision": precision,
                "importance_score": 0.9
            })

        if len(steps) >= 1:
            steps.sort(key=lambda x: x['step'])
            final_answer = '. '.join(step['text'] for step in steps) + '.'

            # Distributions
            comp_dist = {}
            hier_dist = {}
            for step in steps:
                comp_dist[step['compartment']] = comp_dist.get(step['compartment'], 0) + 1
                hier_dist[step['hierarchy']] = hier_dist.get(step['hierarchy'], 0) + 1

            stage2_data.append({
                "id": f"sft_{idx:06d}",
                "domain": data['domain'],
                "query": data['query'],
                "reasoning_chain": steps,
                "final_answer": final_answer,
                "metadata": {
                    "num_steps": len(steps),
                    "compartment_distribution": comp_dist,
                    "hierarchy_distribution": hier_dist,
                    "code_validated": False,
                    "quality_score": 0.85
                }
            })
            processed_doc_ids.add(idx)

        processed_count += 1
        if processed_count % STAGE2_CHECKPOINT_INTERVAL == 0:
            # Incremental save to JSONL (append mode)
            if append_mode:
                with open(stage2_file, 'a', encoding='utf-8') as f:
                    for new_item in stage2_data[-STAGE2_CHECKPOINT_INTERVAL:]:  # Last batch
                        f.write(json.dumps(new_item, ensure_ascii=False) + '\n')
            else:
                with open(stage2_file, 'w', encoding='utf-8') as f:  # First write
                    for item in stage2_data:
                        f.write(json.dumps(item, ensure_ascii=False) + '\n')
            print(f"Stage 2 incremental save: {len(stage2_data)} total samples.")

            # Pickle checkpoint (unchanged)
            save_checkpoint({
                'stage2_data': stage2_data,
                'processed_doc_ids': list(processed_doc_ids)
            }, "stage2_progress")

            import gc  # Free memory
            gc.collect()

        pbar.set_postfix({'processed': processed_count, 'new': len(stage2_data)})

if not append_mode or len(stage2_data) % STAGE2_CHECKPOINT_INTERVAL != 0:
    mode = 'a' if append_mode else 'w'
    with open(stage2_file, mode, encoding='utf-8') as f:
        start = existing_stage2_count if append_mode else 0
        for item in stage2_data[start:]:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
print(f"Stage 2: {len(stage2_data)} new samples → total ~{existing_stage2_count + len(stage2_data)}")

import time
import random

def generate_candidates_with_retry(query, num_candidates=3, model='deepseek-r1:1.5b', max_retries=3):
    """Generate candidates with exponential backoff retry."""

    for attempt in range(max_retries):
        try:
            # Increase timeout progressively
            timeout = 180 + (attempt * 60)  # 180s, 240s, 300s

            prompt = f"""Generate {num_candidates} different step-by-step science explanations for: {query}
Format each as: Step 1 [FACTUAL] [GRANULAR]: text | ... | Final Answer: answer"""

            print("prompt =", prompt)

            response = requests.post(
                'http://localhost:11434/api/generate',
                json={
                    'model': model,
                    'prompt': prompt,
                    'stream': False,
                    'options': {
                        'num_predict': 1024,
                        'temperature': 0.7,
                        'num_ctx': 4096  # Context window
                    }
                },
                timeout=timeout
            )
            
            print("response = ", response)

            if response.status_code == 200:
                raw = response.json().get('response', '')
                candidates = []
                c_list = raw.split('\n\n')[:num_candidates*2]

                for i in range(0, min(len(c_list), num_candidates*2), 2):
                    candidates.append({
                        'candidate_id': chr(65 + i//2),
                        'reasoning_chain': [
                            {'step': j+1, 'compartment': 'FACTUAL', 'hierarchy': 'GRANULAR',
                             'text': f"Step {j+1}"}
                            for j in range(3)
                        ],
                        'final_answer': f"Generated answer {i//2 + 1}",
                        'reward_score': 0.5 + (i//2)*0.2
                    })

                if len(candidates) >= 2:
                    return candidates[:3]

            # If response not 200 or insufficient candidates, retry
            raise ValueError(f"Status {response.status_code} or insufficient candidates")

        except (requests.Timeout, requests.ConnectionError, ValueError) as e:
            wait_time = (2 ** attempt) + random.uniform(0, 1)  # Exponential backoff
            print(f"  Retry {attempt+1}/{max_retries} for query (error: {type(e).__name__}). Waiting {wait_time:.1f}s...")
            time.sleep(wait_time)

            if attempt == max_retries - 1:
                print(f"  ⚠ Failed after {max_retries} attempts")
                return []  # Return empty after all retries exhausted

    return []


# ===================================================================
# STAGE 3: RLAIF (with Checkpointing)
# ===================================================================

print("\nCreating Stage 3 (RLAIF)...")

STAGE3_CHECKPOINT_INTERVAL = 20   # Save every 20 processed queries
STAGE3_CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "stage3_progress.pkl")
stage3_file = f"{OUTPUT_PATH}/stage3_rlaif.jsonl"

stage3_checkpoint = load_checkpoint("stage3_progress")
if stage3_checkpoint:
    stage3_data = stage3_checkpoint.get('stage3_data', [])
    processed_query_indices = set(stage3_checkpoint.get('processed_indices', []))
    print(f"Resuming Stage 3 from checkpoint ({len(stage3_data)} samples completed)")
else:
    stage3_data = []
    processed_query_indices = set()
    print("Starting Stage 3 fresh...")

if 'doc_data' not in locals() or not doc_data:
    doc_data_checkpoint = load_checkpoint("doc_data_for_stage2")
    if doc_data_checkpoint:
        doc_data = doc_data_checkpoint['doc_data']

sample_queries = [(idx, data['query']) for idx, data in list(doc_data.items())[:5000]]

BATCH_SIZE = 10
BATCH_DELAY = 30  # seconds cooldown between batches

total_processed = len(processed_query_indices)  # Count of processed queries so far

for batch_start in range(0, len(sample_queries), BATCH_SIZE):
    batch = sample_queries[batch_start:batch_start + BATCH_SIZE]

    for i, (doc_idx, query) in enumerate(batch):
        global_idx = batch_start + i
        if global_idx in processed_query_indices:
            continue

        if not query:
            continue

        time.sleep(0.5)  # Avoid API overload

        candidates = generate_candidates_with_retry(query)

        if len(candidates) >= 2:
            ai_feedback = {
                "ranking": [c['candidate_id'] for c in sorted(candidates, key=lambda x: x['reward_score'], reverse=True)],
                "criteria": {"logical_flow": [0.9, 0.8, 0.6], "accuracy": [0.95, 0.85, 0.7], "completeness": [0.9, 0.8, 0.5]},
                "critique": {c['candidate_id']: f"Quality: {c['reward_score']:.2f}" for c in candidates},
                "preference_pairs": [{"winner": candidates[0]['candidate_id'], "loser": candidates[j]['candidate_id']} for j in range(1, len(candidates))]
            }

            stage3_data.append({
                "id": f"rlaif_{global_idx:06d}",
                "query": query,
                "candidates": candidates,
                "ai_feedback": ai_feedback,
                "preference_pairs": ai_feedback['preference_pairs']
            })
            processed_query_indices.add(global_idx)
            total_processed += 1

            # Save checkpoint at intervals
            if total_processed % STAGE3_CHECKPOINT_INTERVAL == 0:
                # Append new samples to JSONL
                with open(stage3_file, 'a', encoding='utf-8') as f:
                    for item in stage3_data[-STAGE3_CHECKPOINT_INTERVAL:]:
                        f.write(json.dumps(item, ensure_ascii=False) + '\n')

                save_checkpoint({
                    'stage3_data': stage3_data,
                    'processed_indices': list(processed_query_indices)
                }, "stage3_progress")

                import gc
                gc.collect()

    if batch_start + BATCH_SIZE < len(sample_queries):
        print(f"Batch complete. Cooling down for {BATCH_DELAY}s...")
        time.sleep(BATCH_DELAY)

# Final save: overwrite full JSONL for consistency
with open(stage3_file, 'w', encoding='utf-8') as f:
    for item in stage3_data:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

save_checkpoint({
    'stage3_data': stage3_data,
    'processed_indices': list(processed_query_indices)
}, "stage3_progress")

print(f"Stage 3 complete: {len(stage3_data)} samples → {stage3_file}")


print(f"\n✓ Pipeline Complete! Check {OUTPUT_PATH} for JSONL files.")
print(f"Science docs processed: {science_count} | Multi-step CoT: {sum(1 for d in stage2_data if d['metadata']['num_steps'] >= 2)}")