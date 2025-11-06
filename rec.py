#/usr/bin/env python3
"""
Rebuild Stage 1 checkpoint (doc_data) from existing JSONL file.

FIXES:
- Handles JSONL files without newlines (single-line format)
- Windows console encoding issues (removes emoji)
- Division by zero when doc_data is empty
"""

import json
import pickle
import os
import sys
from pathlib import Path
from collections import defaultdict
import logging
from tqdm import tqdm

# ═══════════════════════════════════════════════════════════════════
# CONFIGURATION - MATCH YOUR MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════

OUTPUT_PATH = Path(r"C:\Users\Khwaish\Google Drive Streaming\My Drive\scm_project\datasets\processed")
CHECKPOINT_DIR = Path(r"C:\Users\Khwaish\Google Drive Streaming\My Drive\scm_project\checkpoints")

STAGE1_JSONL = OUTPUT_PATH / "stage1_pretrain.jsonl"
CHECKPOINT_NAME = "stage1_processing"
DOC_DATA_CHECKPOINT = "doc_data_for_stage2"

# FIX: Setup logging with UTF-8 encoding (Windows compatibility)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# Configure file handler
file_handler = logging.FileHandler(
    CHECKPOINT_DIR / "rebuild_checkpoint.log",
    encoding='utf-8'
)
file_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))

# Configure console handler with UTF-8
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))

# Try to set console to UTF-8 (Windows)
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass  # Python < 3.7

logging.basicConfig(
    level=logging.INFO,
    handlers=[file_handler, console_handler]
)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════
# MAIN REBUILD LOGIC
# ═══════════════════════════════════════════════════════════════════

def parse_sample_id(sample_id):
    """
    Extract document index and segment position from Stage 1 sample ID.
    
    Format: pretrain_{doc_idx:06d}_{seg_position}
    Example: pretrain_000042_003 → doc_idx=42, seg_position=3
    """
    try:
        parts = sample_id.split('_')
        doc_idx = int(parts[1])
        seg_position = int(parts[2])
        return doc_idx, seg_position
    except (IndexError, ValueError) as e:
        logger.warning(f"Invalid sample ID format: {sample_id} - {e}")
        return None, None


def rebuild_doc_data_from_jsonl(jsonl_path):
    """
    Reconstruct doc_data dictionary from Stage 1 JSONL file.
    
    FIX: Handles both newline-separated AND single-line JSONL formats.
    
    Returns:
        doc_data: {doc_idx: {'segments': [...], 'query': str, 'domain': str, ...}}
        stats: {'total_samples', 'unique_docs', 'total_segments'}
    """
    logger.info("="*70)
    logger.info("REBUILDING doc_data FROM STAGE 1 JSONL")
    logger.info("="*70)
    
    if not jsonl_path.exists():
        logger.error(f"[ERROR] JSONL file not found: {jsonl_path}")
        raise FileNotFoundError(f"Cannot rebuild: {jsonl_path} does not exist")
    
    file_size_gb = jsonl_path.stat().st_size / (1024**3)
    logger.info(f"JSONL file: {jsonl_path}")
    logger.info(f"File size: {file_size_gb:.2f} GB")
    
    # FIX: Check if file has newlines
    logger.info("\nDetecting JSONL format...")
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        first_mb = f.read(1024 * 1024)  # Read first 1MB
        newline_count = first_mb.count('\n')
    
    if newline_count == 0:
        logger.warning("[WARNING] JSONL file has NO newlines - all samples on one line!")
        logger.warning("This may be caused by incorrect JSONL writing (missing \\n)")
        logger.info("Attempting to parse as single-line concatenated JSON...")
        use_streaming_parser = True
    else:
        logger.info(f"[OK] JSONL has newlines (detected {newline_count} in first 1MB)")
        use_streaming_parser = False
    
    # Step 2: Group samples by document
    logger.info("\nGrouping samples by document...")
    doc_segments = defaultdict(list)  # {doc_idx: [list of segments]}
    doc_metadata = {}  # {doc_idx: {'query', 'domain', 'source', 'difficulty'}}
    
    total_samples = 0
    parse_errors = 0
    
    if use_streaming_parser:
        # FIX: Parse single-line JSONL (streaming JSON decoder)
        logger.info("Using streaming JSON parser (no newlines detected)...")
        
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            buffer = ""
            pbar = tqdm(desc="Reading JSONL", unit=" samples")
            
            while True:
                chunk = f.read(1024 * 1024)  # Read 1MB chunks
                if not chunk:
                    break
                
                buffer += chunk
                
                # Try to extract complete JSON objects
                while True:
                    try:
                        # Find end of JSON object
                        decoder = json.JSONDecoder()
                        obj, idx = decoder.raw_decode(buffer)
                        
                        # Process sample
                        total_samples += 1
                        pbar.update(1)
                        
                        try:
                            doc_idx, seg_position = parse_sample_id(obj['id'])
                            if doc_idx is not None:
                                segment = {
                                    'text': obj['text'],
                                    'compartment': obj['compartment'],
                                    'hierarchy': obj['hierarchical_level'].upper(),
                                    'position': seg_position
                                }
                                doc_segments[doc_idx].append(segment)
                                
                                if doc_idx not in doc_metadata:
                                    doc_metadata[doc_idx] = {
                                        'query': obj.get('query', ''),
                                        'domain': obj['metadata']['domain'],
                                        'source': obj['metadata']['source'],
                                        'difficulty': obj['metadata']['difficulty']
                                    }
                        except (KeyError, AttributeError) as e:
                            parse_errors += 1
                            if parse_errors < 10:
                                logger.warning(f"Skipping malformed sample: {e}")
                        
                        # Remove processed JSON from buffer
                        buffer = buffer[idx:].lstrip()
                    
                    except json.JSONDecodeError:
                        # No complete JSON object yet, need more data
                        break
            
            pbar.close()
    
    else:
        # Standard newline-separated JSONL
        logger.info("Counting samples...")
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            total_samples = sum(1 for _ in f if _.strip())
        
        logger.info(f"Total samples: {total_samples:,}")
        
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, total=total_samples, desc="Reading JSONL"):
                if not line.strip():
                    continue
                
                try:
                    sample = json.loads(line.strip())
                    
                    doc_idx, seg_position = parse_sample_id(sample['id'])
                    if doc_idx is None:
                        continue
                    
                    segment = {
                        'text': sample['text'],
                        'compartment': sample['compartment'],
                        'hierarchy': sample['hierarchical_level'].upper(),
                        'position': seg_position
                    }
                    
                    doc_segments[doc_idx].append(segment)
                    
                    if doc_idx not in doc_metadata:
                        doc_metadata[doc_idx] = {
                            'query': sample.get('query', ''),
                            'domain': sample['metadata']['domain'],
                            'source': sample['metadata']['source'],
                            'difficulty': sample['metadata']['difficulty']
                        }
                
                except (json.JSONDecodeError, KeyError) as e:
                    parse_errors += 1
                    if parse_errors < 10:
                        logger.warning(f"Skipping malformed line: {e}")
    
    # Log parse errors
    if parse_errors > 0:
        logger.warning(f"[WARNING] {parse_errors} samples skipped due to parse errors")
    
    # Step 3: Reconstruct doc_data structure
    logger.info("\nReconstructing doc_data structure...")
    doc_data = {}
    
    for doc_idx in sorted(doc_segments.keys()):
        segments = sorted(doc_segments[doc_idx], key=lambda x: x['position'])
        
        doc_data[doc_idx] = {
            'segments': segments,
            'query': doc_metadata[doc_idx]['query'],
            'domain': doc_metadata[doc_idx]['domain'],
            'source': doc_metadata[doc_idx]['source'],
            'difficulty': doc_metadata[doc_idx]['difficulty']
        }
    
    # FIX: Handle empty doc_data
    if len(doc_data) == 0:
        logger.error("[ERROR] No valid documents found in JSONL!")
        logger.error("Possible causes:")
        logger.error("  1. JSONL file is corrupted")
        logger.error("  2. Sample ID format doesn't match pretrain_{doc_idx}_{seg_position}")
        logger.error("  3. All samples failed to parse")
        raise ValueError("Cannot proceed: doc_data is empty after parsing")
    
    # Step 4: Compute statistics
    stats = {
        'total_samples': total_samples,
        'unique_docs': len(doc_data),
        'total_segments': sum(len(doc['segments']) for doc in doc_data.values()),
        'avg_segments_per_doc': sum(len(doc['segments']) for doc in doc_data.values()) / max(len(doc_data), 1)  # FIX: Prevent division by zero
    }
    
    logger.info("\n" + "="*70)
    logger.info("REBUILD STATISTICS")
    logger.info("="*70)
    logger.info(f"Total samples processed: {stats['total_samples']:,}")
    logger.info(f"Unique documents: {stats['unique_docs']:,}")
    logger.info(f"Total segments: {stats['total_segments']:,}")
    logger.info(f"Avg segments/doc: {stats['avg_segments_per_doc']:.1f}")
    logger.info(f"Parse errors: {parse_errors}")
    logger.info("="*70)
    
    return doc_data, stats


def save_checkpoint(data, checkpoint_name, checkpoint_dir):
    """Save checkpoint with error handling."""
    checkpoint_path = checkpoint_dir / f"{checkpoint_name}.pkl"
    
    try:
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(data, f)
        
        file_size_mb = checkpoint_path.stat().st_size / (1024**2)
        logger.info(f"[OK] Checkpoint saved: {checkpoint_name}.pkl ({file_size_mb:.1f} MB)")
        return True
    
    except Exception as e:
        logger.error(f"[ERROR] Failed to save checkpoint {checkpoint_name}: {e}")
        return False


def validate_doc_data(doc_data):
    """Validate reconstructed doc_data structure."""
    logger.info("\nValidating doc_data structure...")
    
    issues = []
    
    # Check for empty doc_data
    if not doc_data:
        issues.append("doc_data is empty")
        logger.error("[ERROR] doc_data is empty - cannot proceed")
        return False
    
    # Sample 10 random documents
    sample_size = min(10, len(doc_data))
    sample_docs = list(doc_data.keys())[:sample_size]
    
    for doc_idx in sample_docs:
        doc = doc_data[doc_idx]
        
        # Check required keys
        required_keys = ['segments', 'query', 'domain', 'source', 'difficulty']
        missing = [k for k in required_keys if k not in doc]
        if missing:
            issues.append(f"Doc {doc_idx}: Missing keys {missing}")
        
        # Check segments structure
        if 'segments' in doc:
            if not doc['segments']:
                issues.append(f"Doc {doc_idx}: segments is empty")
            else:
                seg = doc['segments'][0]
                required_seg_keys = ['text', 'compartment', 'hierarchy', 'position']
                missing_seg = [k for k in required_seg_keys if k not in seg]
                if missing_seg:
                    issues.append(f"Doc {doc_idx}: Segment missing keys {missing_seg}")
    
    if issues:
        logger.warning("[WARNING] Validation found issues:")
        for issue in issues:
            logger.warning(f"  - {issue}")
        return False
    else:
        logger.info("[OK] Validation passed (sampled 10 documents)")
        return True


def main():
    """Main rebuild workflow."""
    logger.info("\n" + "="*70)
    logger.info("STAGE 1 CHECKPOINT REBUILD UTILITY")
    logger.info("="*70)
    logger.info(f"Input JSONL: {STAGE1_JSONL}")
    logger.info(f"Output dir: {CHECKPOINT_DIR}")
    logger.info("="*70 + "\n")
    
    # Step 1: Check if checkpoints already exist
    stage1_pkl = CHECKPOINT_DIR / f"{CHECKPOINT_NAME}.pkl"
    doc_data_pkl = CHECKPOINT_DIR / f"{DOC_DATA_CHECKPOINT}.pkl"
    
    if stage1_pkl.exists():
        file_size_mb = stage1_pkl.stat().st_size / (1024**2)
        logger.warning(f"[WARNING] {CHECKPOINT_NAME}.pkl already exists ({file_size_mb:.1f} MB)")
        
        if file_size_mb < 10:
            logger.warning("  File is suspiciously small - likely corrupted")
            response = input("  Overwrite? (yes/no): ").strip().lower()
            if response != 'yes':
                logger.info("Aborting rebuild")
                return
        else:
            logger.info("  File seems valid (>10MB)")
            response = input("  Overwrite anyway? (yes/no): ").strip().lower()
            if response != 'yes':
                logger.info("Aborting rebuild")
                return
    
    # Step 2: Rebuild doc_data from JSONL
    try:
        doc_data, stats = rebuild_doc_data_from_jsonl(STAGE1_JSONL)
    except Exception as e:
        logger.error(f"[ERROR] Rebuild failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 3: Validate reconstructed data
    if not validate_doc_data(doc_data):
        logger.error("[ERROR] Validation failed - aborting")
        return
    
    # Step 4: Compute checkpoint metadata
    doc_count = len(doc_data)
    last_idx = max(doc_data.keys())
    doc_ids_processed = list(doc_data.keys())
    
    logger.info("\nPreparing checkpoint data...")
    logger.info(f"  doc_count: {doc_count}")
    logger.info(f"  last_idx: {last_idx}")
    logger.info(f"  doc_ids count: {len(doc_ids_processed)}")
    
    # Step 5: Save stage1_processing checkpoint (FULL doc_data with segments)
    checkpoint_full = {
        'doc_data': doc_data,
        'stage1_data': [],
        'doc_count': doc_count,
        'science_count': doc_count,
        'last_idx': last_idx,
        'doc_ids_processed': doc_ids_processed
    }
    
    logger.info("\nSaving stage1_processing checkpoint...")
    if not save_checkpoint(checkpoint_full, CHECKPOINT_NAME, CHECKPOINT_DIR):
        logger.error("[ERROR] Failed to save stage1_processing checkpoint")
        return
    
    # Step 6: Save doc_data_for_stage2 checkpoint
    logger.info("Saving doc_data_for_stage2 checkpoint...")
    if not save_checkpoint({'doc_data': doc_data}, DOC_DATA_CHECKPOINT, CHECKPOINT_DIR):
        logger.error("[ERROR] Failed to save doc_data_for_stage2 checkpoint")
        return
    
    # Step 7: Final verification
    logger.info("\n" + "="*70)
    logger.info("REBUILD COMPLETE")
    logger.info("="*70)
    logger.info(f"[OK] {CHECKPOINT_NAME}.pkl created")
    logger.info(f"[OK] {DOC_DATA_CHECKPOINT}.pkl created")
    logger.info(f"[OK] Ready for Stage 2 processing")
    logger.info("="*70 + "\n")
    
    logger.info("Next steps:")
    logger.info("1. Verify checkpoint files exist in: " + str(CHECKPOINT_DIR))
    logger.info("2. Check file sizes (should be 100-500MB)")
    logger.info("3. Run your main pipeline (Stage 2 will load from checkpoint)")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n\n[WARNING] Rebuild interrupted by user")
    except Exception as e:
        logger.error(f"\n\n[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
