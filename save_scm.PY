#!/usr/bin/env python3
"""
SCM Model Export Script
Exports trained SCM model to production-ready formats (.pt, .h5, .pkl)
"""

import torch
import pickle
import json
from pathlib import Path
import logging
from scm_main import SCMConfig, SCMTransformer, SONARDecoder

def export_scm_model(
    checkpoint_path: str,
    export_dir: str = "G:/My Drive/scm_project/exported_models",
    formats: list = ['pt', 'pkl']  # h5 not recommended for PyTorch
):
    """
    Export trained SCM model to multiple formats
    
    Args:
        checkpoint_path: Path to best checkpoint (e.g., "stage3_best.pt")
        export_dir: Directory to save exported models
        formats: List of formats ['pt', 'pkl']
    """
    logger = logging.getLogger(__name__)
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    
    # Load config and model
    config = SCMConfig()
    model = SCMTransformer(config).to('cuda')
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cuda')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    logger.info(f"✅ Loaded model from {checkpoint_path}")
    
    # ========== FORMAT 1: PyTorch (.pt) - RECOMMENDED ==========
    if 'pt' in formats:
        pt_path = export_dir / "scm_production.pt"
        
        # Export model + config + metadata
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': {
                'SONAR_DIM': config.SONAR_DIM,
                'COMPRESSED_DIM': config.COMPRESSED_DIM,
                'N_HEADS': config.N_HEADS,
                'N_LAYERS': config.N_LAYERS,
                'MAX_SEQUENCE_LENGTH': config.MAX_SEQUENCE_LENGTH,
                'FFN_MULTIPLIER': config.FFN_MULTIPLIER,
                'DROPOUT': config.DROPOUT,
                'DOMAIN_VOCAB_SIZES': config.DOMAIN_VOCAB_SIZES
            },
            'compartment_to_idx': checkpoint['compartment_to_idx'],
            'hierarchy_to_idx': checkpoint['hierarchy_to_idx'],
            'training_metadata': {
                'final_loss': checkpoint.get('best_loss', 0),
                'total_steps': checkpoint.get('global_step', 0),
                'stage': checkpoint.get('stage', 'stage3')
            }
        }, pt_path)
        
        logger.info(f"✅ Exported PyTorch model: {pt_path}")
    
    # ========== FORMAT 2: Pickle (.pkl) ==========
    if 'pkl' in formats:
        pkl_path = export_dir / "scm_production.pkl"
        
        # Move model to CPU for portability
        model_cpu = model.cpu()
        
        with open(pkl_path, 'wb') as f:
            pickle.dump({
                'model': model_cpu,
                'config': config,
                'compartment_to_idx': checkpoint['compartment_to_idx'],
                'hierarchy_to_idx': checkpoint['hierarchy_to_idx']
            }, f)
        
        logger.info(f"✅ Exported Pickle model: {pkl_path}")
    
    # ========== EXPORT DOMAIN VOCABULARY ==========
    vocab_path = export_dir / "domain_vocabulary.pt"
    torch.save({
        'prototypes': model.domain_vocab.prototypes,
        'domains': model.domain_vocab.domains
    }, vocab_path)
    
    logger.info(f"✅ Exported domain vocabulary: {vocab_path}")
    
    # ========== EXPORT DECODER CORPUS (if exists) ==========
    decoder_checkpoint = Path(checkpoint_path).parent / "sonar_decoder.pt"
    if decoder_checkpoint.exists():
        decoder_data = torch.load(decoder_checkpoint)
        decoder_export = export_dir / "sonar_decoder.pt"
        torch.save(decoder_data, decoder_export)
        logger.info(f"✅ Exported decoder corpus: {decoder_export}")
    
    logger.info(f"\n🎉 Model export complete! Files saved to {export_dir}")
    
    return export_dir

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Export best model from Stage 3
    export_scm_model(
        checkpoint_path="G:/My Drive/scm_project/checkpoints/scm_training/stage3_best.pt",
        export_dir="G:/My Drive/scm_project/scm_production",
        formats=['pt', 'pkl']
    )
