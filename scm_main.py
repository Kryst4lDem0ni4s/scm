#!/usr/bin/env python3
"""
Small Concept Model (SCM) Training Pipeline
Architecture: Domain-specialized concept-level reasoning with compressed SONAR embeddings

Implements:
- Concept Compression Layer (1024 → 384 dims)
- Domain-Specific Concept Transformer
- Efficient Autoregressive Generation
- Multi-Stage Training (Pre-training + SFT + RLAIF)

Dataset Compatibility:
- Stage 1: Diffusion pre-training (stage1_pretrain.jsonl)
- Stage 2: Supervised fine-tuning (stage2_sft.jsonl)
- Stage 3: RLAIF optimization (stage3_rlaif.jsonl)

# concept clustering and sonar space only work on linux (colab)
!pip install fairseq2 --extra-index-url https://fair.pkg.atmeta.com/fairseq2/whl/pt2.9.0/cu128
!pip install sonar-space
"""

import math
import os
import subprocess
import sys
import json
import pickle
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import gc

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm
from torch.optim.swa_utils import AveragedModel, SWALR

# ═══════════════════════════════════════════════════════════════════
# CONFIGURATION - MODIFY THESE PATHS FOR YOUR SYSTEM
# ═══════════════════════════════════════════════════════════════════

def setup_cuda():
    """Ensure PyTorch with CUDA 12.x is installed"""
    try:
        import torch
        if torch.cuda.is_available():
            cuda_version = torch.version.cuda
            major_version = int(cuda_version.split('.')[0]) if cuda_version else 0
            if major_version >= 12:
                print(f"✅ CUDA {cuda_version} detected")
                print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
                return True
    except ImportError:
        pass
    
    # Install PyTorch with CUDA 12.1
    print("⚠ Installing PyTorch with CUDA 12.9 support...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "--upgrade",
        "torch", 
        "torchvision", 
        "torchaudio",
        "--extra-index-url", "https://download.pytorch.org/whl/cu128"
    ])
    
    import torch
    assert torch.cuda.is_available(), "❌ CUDA installation failed"
    print(f"✅ PyTorch {torch._version_} installed with CUDA {torch.version.cuda}")
    return True

# Run setup
setup_cuda()

# Force CUDA as default device
torch.set_default_device('cuda')
# print(f"Default device: {torch.get_default_device()}")

# ═══════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

class SCMConfig:
    """SCM Architecture Configuration"""
    
    # Model Architecture
    SONAR_DIM = 1024  # Input SONAR embedding dimension
    COMPRESSED_DIM = 384  # Compressed concept dimension (256/384/512)
    N_HEADS = 6  # Multi-head attention heads
    N_LAYERS = 8  # Transformer layers
    FFN_MULTIPLIER = 2  # FFN hidden dimension multiplier
    DROPOUT = 0.1  # Dropout rate
    MAX_SEQUENCE_LENGTH = 128  # Max concepts per sequence
    
    # Training Hyperparameters
    BATCH_SIZE = 16  # Training batch size
    LEARNING_RATE = 3e-4  # Initial learning rate
    
    # For Stage 2/3, use lower initial LR
    LEARNING_RATE_STAGE1 = 3e-4  # Stage 1
    LEARNING_RATE_STAGE2 = 1e-5  # Stage 2 (lower!)
    LEARNING_RATE_STAGE3 = 5e-6  # Stage 3 (even lower)

    
    WEIGHT_DECAY = 0.01  # L2 regularization
    MAX_EPOCHS = 10  # Maximum training epochs
    WARMUP_STEPS = None  # Learning rate warmup steps
    GRADIENT_CLIP = 5.0  # Gradient clipping threshold
    
    # Loss Weights
    ALPHA_CONCEPT = 1.0  # Concept prediction loss weight
    BETA_DOMAIN = 0.5  # Domain-specific loss weight
    GAMMA_EFFICIENCY = 0.1  # Efficiency regularization weight
    
    # Precision-Aware Training
    FP32_PRECISION_WEIGHT = 2.0  # Weight for FP32-required samples
    FP16_PRECISION_WEIGHT = 1.0  # Weight for FP16-acceptable samples
    
    # Data Paths
    BASE_PATH = Path(r"G:\My Drive\scm_project")
    STAGE1_JSONL = BASE_PATH / "datasets" / "processed" / "stage1_pretrain.jsonl"
    STAGE2_JSONL = BASE_PATH / "datasets" / "processed" / "stage2_sft.jsonl"
    STAGE3_JSONL = BASE_PATH / "datasets" / "processed" / "stage3_rlaif.jsonl"
    CHECKPOINT_DIR = BASE_PATH / "checkpoints" / "scm_training"
    LOG_DIR = BASE_PATH / "logs" / "scm_training"
    
    # Valid Labels
    VALID_COMPARTMENTS = {'FACTUAL', 'PROCEDURAL', 'EPISODIC', 'CONTEXTUAL', 'CONCEPTUAL'}
    VALID_HIERARCHIES = {'GRANULAR', 'INTERMEDIATE', 'GENERAL'}
    VALID_PRECISIONS = {'fp32', 'fp16'}
    
    RUN_LR_FINDER = True # Change after fresh runs
    
    # Device
    DEVICE = torch.device("cuda")
    print("Device:", str(DEVICE))
    if str(DEVICE)!="cuda":
        torch.set_default_device('cuda')
        DEVICE = "cuda"
    
    DOMAIN_VOCAB_SIZES = {
        'general': 8000,
        'science': 5000,
        'medical': 5000,
        'legal': 3000,
        'code': 2000
    }
    
    ACCUMULATION_STEPS = 4
    
    @classmethod
    def validate(cls):
        """Validate configuration"""
        assert cls.SONAR_DIM == 1024, "SCM requires 1024-dim SONAR embeddings"
        assert cls.COMPRESSED_DIM in [256, 384, 512], "Compressed dim must be 256/384/512"
        assert cls.STAGE1_JSONL.exists(), f"Stage 1 dataset not found: {cls.STAGE1_JSONL}"
        cls.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        cls.LOG_DIR.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════
# LOGGING SETUP
# ═══════════════════════════════════════════════════════════════════

def setup_logging(config: SCMConfig) -> logging.Logger:
    """Setup logging with file and console handlers"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = config.LOG_DIR / f"scm_training_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("="*80)
    logger.info("SMALL CONCEPT MODEL (SCM) TRAINING PIPELINE")
    logger.info("="*80)
    logger.info(f"Device: {config.DEVICE}")
    logger.info(f"SONAR Dim: {config.SONAR_DIM}")
    logger.info(f"Compressed Dim: {config.COMPRESSED_DIM}")
    logger.info(f"Model Layers: {config.N_LAYERS}")
    logger.info(f"Attention Heads: {config.N_HEADS}")
    logger.info(f"Batch Size: {config.BATCH_SIZE}")
    logger.info(f"Learning Rate: {config.LEARNING_RATE}")
    logger.info("="*80)
    
    return logger

# ═══════════════════════════════════════════════════════════════════
# MODEL ARCHITECTURE
# ═══════════════════════════════════════════════════════════════════

class ConceptCompressor(nn.Module):
    """
    Concept Compression Layer: 1024-dim SONAR → 384-dim compressed concepts
    
    Implements bidirectional compression/expansion with domain-specific bias.
    """
    
    def __init__(self, sonar_dim: int = 1024, compressed_dim: int = 384):
        super().__init__()
        self.sonar_dim = sonar_dim
        self.compressed_dim = compressed_dim
        
        # Compression pathway
        self.compress = nn.Sequential(
            nn.Linear(sonar_dim, compressed_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(compressed_dim * 2, compressed_dim)
        )
        
        # Expansion pathway
        self.expand = nn.Sequential(
            nn.Linear(compressed_dim, sonar_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(sonar_dim * 2, sonar_dim)
        )
        
        # Domain-specific bias (learnable)
        self.domain_bias = nn.Parameter(torch.zeros(1, 1, compressed_dim))
        
        # Layer normalization
        self.norm_compress = nn.LayerNorm(compressed_dim)
        self.norm_expand = nn.LayerNorm(sonar_dim)
    
    def compress_concept(self, sonar_embedding: torch.Tensor) -> torch.Tensor:
        """
        Compress SONAR embedding to concept space
        
        Args:
            sonar_embedding: [batch, seq_len, 1024]
        
        Returns:
            compressed: [batch, seq_len, 384]
        """
        
        compressed = self.compress(sonar_embedding) + self.domain_bias
        return self.norm_compress(compressed)
    
    def expand_concept(self, compressed_concept: torch.Tensor) -> torch.Tensor:
        """
        Expand compressed concept back to SONAR space
        
        Args:
            compressed_concept: [batch, seq_len, 384]
        
        Returns:
            expanded: [batch, seq_len, 1024]
        """
        # Remove domain bias before expansion
        debiased = compressed_concept - self.domain_bias
        expanded = self.expand(debiased)
        return self.norm_expand(expanded)
    
    def reconstruction_loss(self, sonar_embedding: torch.Tensor) -> torch.Tensor:
        """Compute reconstruction loss for compression training"""
        compressed = self.compress_concept(sonar_embedding)
        reconstructed = self.expand_concept(compressed)
        return F.mse_loss(reconstructed, sonar_embedding)

class SONARDecoder(nn.Module):
    """
    SONAR Concept-to-Text Decoder
    
    Decodes compressed concepts back to text via:
    1. Expand to full SONAR space (384 → 1024)
    2. Retrieve nearest sentences from training corpus (kNN)
    3. Optional: Fine-tune a small decoder for generation
    """
    
    def __init__(
        self, 
        compressor: ConceptCompressor,
        corpus_embeddings: Optional[torch.Tensor] = None,
        corpus_texts: Optional[List[str]] = None
    ):
        """
        Args:
            compressor: Trained concept compressor
            corpus_embeddings: [N, 1024] SONAR embeddings of training sentences
            corpus_texts: [N] Corresponding text strings
        """
        super().__init__()
        self.compressor = compressor
        
        # kNN retrieval corpus
        self.register_buffer('corpus_embeddings', corpus_embeddings)
        self.corpus_texts = corpus_texts
    
    def decode_to_sonar(self, compressed_concepts: torch.Tensor) -> torch.Tensor:
        """
        Expand compressed concepts to full SONAR space
        
        Args:
            compressed_concepts: [batch, seq_len, 384]
        
        Returns:
            sonar_embeddings: [batch, seq_len, 1024]
        """
        return self.compressor.expand_concept(compressed_concepts)

    
    def retrieve_text(
        self, 
        sonar_embeddings: torch.Tensor,
        k: int = 1,
        temperature: float = 1.0
    ) -> List[List[str]]:
        """
        Retrieve nearest sentences from corpus using kNN
        
        Args:
            sonar_embeddings: [batch, seq_len, 1024]
            k: Number of nearest neighbors to retrieve
            temperature: Softmax temperature for probabilistic selection
        
        Returns:
            retrieved_texts: List[List[str]] of shape [batch, seq_len]
        """
        # Vectorized similarity computation
        query_norm = F.normalize(sonar_embeddings, dim=-1)  # [B, L, 1024]
        corpus_norm = F.normalize(self.corpus_embeddings, dim=-1)  # [N, 1024]
        
        # Batch matmul: [B, L, 1024] @ [1024, N] = [B, L, N]
        similarities = torch.matmul(query_norm, corpus_norm.t())
        
        # Get top-k indices: [B, L, k]
        _, topk_indices = similarities.topk(k, dim=-1)
        
        # Convert to text
        results = []
        for b in range(similarities.shape[0]):
            batch_results = []
            for s in range(similarities.shape[1]):
                best_idx = topk_indices[b, s, 0].item()
                batch_results.append(self.corpus_texts[best_idx])
            results.append(batch_results)
        
        return results
    
    def build_corpus(
        self, 
        sonar_encoder,  # External SONAR encoder
        text_corpus: List[str],
        batch_size: int = 32
    ):
        """
        Build kNN corpus by encoding all training sentences
        
        Args:
            sonar_encoder: Pre-trained SONAR TextToEmbeddingModelPipeline
            text_corpus: List of training sentences
            batch_size: Encoding batch size
        """
        from tqdm import tqdm
        import numpy as np
        
        embeddings = []
        for i in tqdm(range(0, len(text_corpus), batch_size), desc="Building corpus"):
            batch_texts = text_corpus[i:i+batch_size]
            
            try:
                # Get embeddings from SONAR
                batch_embs = sonar_encoder.predict(batch_texts, source_lang='eng_Latn')
                
                # ✅ FIX: Handle both Tensor and NumPy array returns
                if isinstance(batch_embs, torch.Tensor):
                    # Already a tensor - just move to CPU if needed
                    embeddings.append(batch_embs.cpu().float())
                elif isinstance(batch_embs, np.ndarray):
                    # NumPy array - convert to tensor
                    embeddings.append(torch.from_numpy(batch_embs).float())
                else:
                    raise TypeError(f"Unexpected type: {type(batch_embs)}")
                    
            except Exception as e:
                print(f"⚠️ Failed to encode batch {i}-{i+batch_size}: {e}")
                # Fallback: create random embeddings for failed batch
                embeddings.append(torch.randn(len(batch_texts), 1024))
        
        if len(embeddings) == 0:
            raise RuntimeError("No embeddings were generated. Check SONAR encoder.")
        
        # Concatenate all embeddings
        self.corpus_embeddings = torch.cat(embeddings, dim=0).float()
        self.corpus_texts = text_corpus
        
        print(f"✅ Corpus built: {len(text_corpus)} sentences, {self.corpus_embeddings.shape}")

class DomainBiasedAttention(nn.Module):
    """
    Custom Multi-Head Attention with Domain Bias Injection
    
    Implements: Attention(Q,K,V) = softmax(QK^T/√d_comp + B_domain)V
    """
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Q, K, V projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # ✅ Domain-specific attention bias (learnable per head)
        # Shape: [n_heads, 1, 1] - broadcasts to [n_heads, seq_len, seq_len]
        self.domain_attn_bias = nn.Parameter(torch.zeros(n_heads, 1, 1))
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
    
    def forward(
        self, 
        x: torch.Tensor, 
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch, seq_len, d_model]
            attn_mask: [seq_len, seq_len] causal mask (True = masked)
            key_padding_mask: [batch, seq_len] padding mask (True = masked)
        
        Returns:
            output: [batch, seq_len, d_model]
            attn_weights: [batch, n_heads, seq_len, seq_len]
        """
        batch_size, seq_len, _ = x.shape
        
        # Project and reshape to [batch, n_heads, seq_len, d_k]
        Q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Compute attention scores: QK^T / √d_k
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # [B, n_heads, L, L]
        
        # ✅ INJECT DOMAIN BIAS (before softmax!)
        scores = scores + self.domain_attn_bias  # Broadcasts to [B, n_heads, L, L]
        
        # Apply causal mask (if provided)
        if attn_mask is not None:
            # attn_mask: [L, L] → expand to [1, 1, L, L]
            scores = scores.masked_fill(attn_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        # Apply padding mask (if provided)
        if key_padding_mask is not None:
            # key_padding_mask: [B, L] → expand to [B, 1, 1, L]
            scores = scores.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        
        # Softmax and apply to values
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Weighted sum of values
        attn_out = torch.matmul(attn_weights, V)  # [B, n_heads, L, d_k]
        
        # Reshape back to [B, L, d_model]
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        # Final projection
        output = self.out_proj(attn_out)
        
        return output, attn_weights


class ConceptTransformerLayer(nn.Module):
    """
    Transformer layer with domain-biased attention
    """
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, ffn_mult: int = 2):
        super().__init__()
        
        # ✅ Use custom domain-biased attention
        self.self_attn = DomainBiasedAttention(d_model, n_heads, dropout)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_mult * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_mult * d_model, d_model)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        x: torch.Tensor, 
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Self-attention with residual
        attn_out, _ = self.self_attn(x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # FFN with residual
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        
        return x

class SCMTransformer(nn.Module):
    """
    Small Concept Model Transformer
    
    Efficient concept-level autoregressive reasoning in compressed semantic space.
    """
    
    def __init__(self, config: SCMConfig):
        super().__init__()
        self.config = config
        
        # Concept compressor
        self.compressor = ConceptCompressor(
            config.SONAR_DIM, 
            config.COMPRESSED_DIM
        )
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(
            torch.randn(1, config.MAX_SEQUENCE_LENGTH, config.COMPRESSED_DIM) * 0.02
        )
        
        # Transformer layers
        self.layers = nn.ModuleList([
            ConceptTransformerLayer(
                config.COMPRESSED_DIM, 
                config.N_HEADS, 
                config.DROPOUT,
                ffn_mult=config.FFN_MULTIPLIER
            )
            for _ in range(config.N_LAYERS)
        ])
        
        # Output layer
        self.output_norm = nn.LayerNorm(config.COMPRESSED_DIM)
        
        # Compartment/Hierarchy classification heads
        self.compartment_head = nn.Linear(
            config.COMPRESSED_DIM, 
            len(config.VALID_COMPARTMENTS)
        )
        self.hierarchy_head = nn.Linear(
            config.COMPRESSED_DIM, 
            len(config.VALID_HIERARCHIES)
        )
        
        # Concept prediction head (for autoregressive generation)
        self.concept_head = nn.Linear(
            config.COMPRESSED_DIM, 
            config.COMPRESSED_DIM
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Create causal mask
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(config.MAX_SEQUENCE_LENGTH, config.MAX_SEQUENCE_LENGTH), diagonal=1).bool()
        )
        
        self.domain_vocab = DomainConceptVocabulary(
            config.COMPRESSED_DIM,
            config.DOMAIN_VOCAB_SIZES
        )
    
    def _init_weights(self, module):
        """Initialize weights with proper scaling"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    def forward(
        self, 
        sonar_embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_causal_mask: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            sonar_embeddings: [batch, seq_len, 1024] SONAR embeddings
            attention_mask: [batch, seq_len] padding mask (1 = valid, 0 = padding)
        
        Returns:
            dict with keys:
                - compressed_concepts: [batch, seq_len, 384]
                - concept_predictions: [batch, seq_len, 384]
                - compartment_logits: [batch, seq_len, 5]
                - hierarchy_logits: [batch, seq_len, 3]
        """
        batch_size, seq_len, _ = sonar_embeddings.shape
        
        # Compress to concept space
        compressed = self.compressor.compress_concept(sonar_embeddings)  # [B, L, 384]
        
        # Add positional encoding
        x = compressed + self.pos_encoding[:, :seq_len, :]
        
        # Prepare masks
        if use_causal_mask:
            causal_mask = self.causal_mask[:seq_len, :seq_len].to(x.device)
        else:
            causal_mask = None  # ✅ No causal mask for Stage 1
        
        key_padding_mask = None if attention_mask is None else (attention_mask == 0)
        
        # Transformer layers
        for layer in self.layers:
            x = layer(x, attn_mask=causal_mask, key_padding_mask=key_padding_mask)
        
        x = self.output_norm(x)
        
        # Output heads
        concept_pred = self.concept_head(x)  # Next concept prediction
        compartment_logits = self.compartment_head(x)
        hierarchy_logits = self.hierarchy_head(x)
        
        return {
            'compressed_concepts': compressed,
            'concept_predictions': concept_pred,
            'compartment_logits': compartment_logits,
            'hierarchy_logits': hierarchy_logits
        }
    
    def generate(
        self, 
        initial_concepts: torch.Tensor,
        max_length: int = 20,
        temperature: float = 0.7,
        domain_bias: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Autoregressive concept generation
        
        Args:
            initial_concepts: [batch, init_len, 1024] Initial SONAR embeddings
            max_length: Maximum sequence length to generate
            temperature: Sampling temperature
            domain_bias: Optional domain-specific attention bias
        
        Returns:
            generated_concepts: [batch, max_length, 384] Generated compressed concepts
        """
        self.eval()
        batch_size = initial_concepts.shape[0]
        
        # Compress initial concepts
        concepts = self.compressor.compress_concept(initial_concepts)
        
        with torch.no_grad():
            for _ in range(max_length - concepts.shape[1]):
                # ✅ Expand concepts to SONAR space for forward pass
                expanded = self.compressor.expand_concept(concepts)
                original_biases = []
                # ✅ Apply domain bias if provided
                if domain_bias is not None:
                    # Inject domain bias into attention layers
                    for layer in self.layers:
                        if hasattr(layer.self_attn, 'domain_attn_bias'):
                            # Temporarily override bias
                            original_biases.append(layer.self_attn.domain_attn_bias.data.clone())
                            layer.self_attn.domain_attn_bias.data = domain_bias

                # Forward pass
                outputs = self.forward(expanded, use_causal_mask=True)
                
                # ✅ Restore original bias
                if domain_bias is not None:
                    for i, layer in enumerate(self.layers):
                        if hasattr(layer.self_attn, 'domain_attn_bias'):
                            layer.self_attn.domain_attn_bias.data = original_biases[i]
                
                # Get next concept prediction
                next_concept_logits = outputs['concept_predictions'][:, -1, :]
                
                # ✅ Sample from distribution (or use argmax for greedy)
                if temperature > 0:
                    # Sample from Gaussian distribution around prediction
                    noise = torch.randn_like(next_concept_logits) * temperature
                    next_concept = next_concept_logits + noise
                else:
                    # Greedy (deterministic)
                    next_concept = next_concept_logits
                
                # Append to sequence
                concepts = torch.cat([concepts, next_concept.unsqueeze(1)], dim=1)
                
                # Early stopping (if next concept is near-zero)
                if torch.norm(next_concept, dim=-1).mean() < 0.1:
                    break
        
        return concepts

class DomainConceptVocabulary(nn.Module):
    """
    Domain-Specific Concept Vocabulary
    
    Maintains learnable concept prototypes per domain and enforces
    concepts stay close to domain clusters via cosine similarity loss.
    """
    
    def __init__(self, compressed_dim: int, vocab_size_per_domain: Dict[str, int]):
        """
        Args:
            compressed_dim: Size of compressed concept embeddings (384)
            vocab_size_per_domain: e.g., {'general': 8000, 'medical': 5000, 'legal': 3000}
        """
        super().__init__()
        self.compressed_dim = compressed_dim
        self.domains = list(vocab_size_per_domain.keys())
        
        # Learnable concept prototypes per domain
        self.prototypes = nn.ParameterDict({
            domain: nn.Parameter(torch.randn(size, compressed_dim) * 0.02)
            for domain, size in vocab_size_per_domain.items()
        })
        
        # Normalize prototypes (unit vectors for cosine similarity)
        with torch.no_grad():
            for domain in self.prototypes:
                self.prototypes[domain].data = F.normalize(self.prototypes[domain].data, dim=-1)
    
    def get_prototypes(self, domain: str) -> torch.Tensor:
        """Get normalized prototypes for a domain"""
        if domain not in self.prototypes:
            domain = 'general'  # Fallback
        return F.normalize(self.prototypes[domain], dim=-1)
    
    def compute_domain_loss(
        self, 
        concepts: torch.Tensor, 
        domains: List[str],
        temperature: float = 0.1
    ) -> torch.Tensor:
        """
        Compute domain vocabulary alignment loss
        
        Args:
            concepts: [batch, seq_len, compressed_dim]
            domains: List of domain strings (length = batch)
            temperature: Softmax temperature for soft assignment
        
        Returns:
            loss: Scalar loss encouraging concepts to align with domain prototypes
        """
        batch_size, seq_len, _ = concepts.shape
        device = concepts.device
        
        # Normalize concepts
        concepts_norm = F.normalize(concepts, dim=-1)  # [B, L, D]
        
        # Compute loss per sample
        total_loss = 0.0
        for b, domain in enumerate(domains):
            # Get prototypes for this domain
            prototypes = self.get_prototypes(domain).to(device)  # [V, D]
            
            # Compute cosine similarity: [L, V]
            similarities = torch.matmul(
                concepts_norm[b],  # [L, D]
                prototypes.t()     # [D, V]
            )  # [L, V]
            
            # Soft assignment loss (negative entropy for diversity + max similarity)
            # Encourage each concept to be similar to at least one prototype
            max_sim, _ = similarities.max(dim=-1)  # [L]
            loss_alignment = -max_sim.mean()  # Maximize similarity to closest prototype
            
            # Encourage diversity (concepts shouldn't collapse to single prototype)
            probs = F.softmax(similarities / temperature, dim=-1)  # [L, V]
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
            loss_diversity = -entropy  # Maximize entropy
            
            total_loss += loss_alignment + 0.1 * loss_diversity
        
        return total_loss / batch_size
    
    def cluster_concepts(
        self, 
        concepts: torch.Tensor, 
        domain: str,
        n_clusters: Optional[int] = None
    ) -> torch.Tensor:
        """
        Cluster concepts using k-means (post-training analysis)
        
        Args:
            concepts: [N, compressed_dim] Collected concepts from training
            domain: Domain name
            n_clusters: Override vocabulary size
        
        Returns:
            cluster_centers: [n_clusters, compressed_dim]
        """
        from sklearn.cluster import KMeans
        
        n_clusters = n_clusters or self.prototypes[domain].shape[0]
        
        # Run k-means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(concepts.cpu().numpy())
        
        # Update prototypes with cluster centers
        cluster_centers = torch.from_numpy(kmeans.cluster_centers_).float()
        with torch.no_grad():
            self.prototypes[domain].data = F.normalize(cluster_centers, dim=-1)
        
        return cluster_centers

# ═══════════════════════════════════════════════════════════════════
# DATASET LOADERS
# ═══════════════════════════════════════════════════════════════════

class Stage1Dataset(Dataset):
    """
    Stage 1 Pre-training Dataset (Diffusion Denoising)
    
    Loads from stage1_pretrain.jsonl with SONAR embeddings and diffusion noise.
    """
    def __init__(self, jsonl_path: Path, max_samples: Optional[int] = None, logger: Optional[logging.Logger] = None):
        self.samples = []
        if logger is None:
            logger = logging.getLogger(__name__)
                
        logger.info(f"Loading Stage 1 dataset from {jsonl_path}")
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(tqdm(f, desc="Loading Stage 1")):
                if max_samples and i >= max_samples:
                    break
                
                try:
                    sample = json.loads(line.strip())
                    
                    # Validate required fields
                    if 'clean_embedding' not in sample or 'noisy_embedding' not in sample:
                        continue
                    
                    # Validate embedding dimensions
                    if len(sample['clean_embedding']) != 1024 or len(sample['noisy_embedding']) != 1024:
                        continue

                    
                    self.samples.append({
                        'clean_embedding': np.array(sample['clean_embedding'], dtype=np.float32),
                        'noisy_embedding': np.array(sample['noisy_embedding'], dtype=np.float32),
                        'timestep': sample['timestep'],
                        'alpha_t': sample['alpha_t'],
                        'compartment': sample['compartment'],
                        'hierarchy': sample['hierarchical_level'].upper(),
                        'precision': sample['metadata']['precision_required'],
                        'importance': sample['metadata']['importance_score']
                    })
                
                except (json.JSONDecodeError, KeyError, ValueError) as e:
                    continue
        
        logger.info(f"Loaded {len(self.samples)} Stage 1 samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        return {
            'clean_embedding': torch.from_numpy(sample['clean_embedding']),
            'noisy_embedding': torch.from_numpy(sample['noisy_embedding']),
            'timestep': torch.tensor(sample['timestep'], dtype=torch.long),
            'alpha_t': torch.tensor(sample['alpha_t'], dtype=torch.float32),
            'compartment': sample['compartment'],
            'hierarchy': sample['hierarchy'],
            'precision': sample['precision'],
            'importance': torch.tensor(sample['importance'], dtype=torch.float32)
        }


class Stage2Dataset(Dataset):
    """
    Stage 2 Supervised Fine-Tuning Dataset (Chain-of-Thought)
    
    Loads multi-step reasoning chains from stage2_sft.jsonl.
    """
    
    def __init__(self, jsonl_path: Path, max_samples: Optional[int] = None, logger: Optional[logging.Logger] = None):
        self.samples = []
        if logger is None:
            logger = logging.getLogger(__name__)
        
        logger.info(f"Loading Stage 2 dataset from {jsonl_path}")
        
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(tqdm(f, desc="Loading Stage 2")):
                if max_samples and i >= max_samples:
                    break
                    
                try:
                    sample = json.loads(line.strip())
                    
                    # Extract reasoning chain
                    reasoning_chain = sample['reasoning_chain']
                    
                    if len(reasoning_chain) < 2:  # Need at least 2 steps
                        continue
                    
                    # Validate all steps have embeddings
                    valid = True
                    for step in reasoning_chain:
                        if 'sonar_embedding' not in step or len(step['sonar_embedding']) != 1024:
                            valid = False
                            break
                    
                    if not valid:
                        continue
                    
                    self.samples.append({
                        'query': sample['query'],
                        'reasoning_chain': reasoning_chain,
                        'final_answer': sample['final_answer'],
                        'domain': sample['domain']
                    })
                
                except (json.JSONDecodeError, KeyError) as e:
                    continue
        
        logger.info(f"Loaded {len(self.samples)} Stage 2 samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        chain = sample['reasoning_chain']
        
        # Extract embeddings and metadata
        embeddings = np.array([step['sonar_embedding'] for step in chain], dtype=np.float32)
        compartments = [step['compartment'] for step in chain]
        hierarchies = [step['hierarchy'] for step in chain]
        precisions = [step['precision'] for step in chain]
        importances = np.array([step['importance_score'] for step in chain], dtype=np.float32)
        
        return {
            'embeddings': torch.from_numpy(embeddings),  # [seq_len, 1024]
            'compartments': compartments,
            'hierarchies': hierarchies,
            'precisions': precisions,
            'importances': torch.from_numpy(importances),
            'query': sample['query'],
            'domain': sample['domain']
        }

class Stage3RLAIFDataset(Dataset):
    """
    Stage 3: Reinforcement Learning from AI Feedback (RLAIF)
    
    Loads query-response pairs with AI-generated quality scores.
    """
    
    def __init__(
        self, 
        jsonl_path: Path, 
        max_samples: Optional[int] = None,
        logger: Optional[logging.Logger] = None
    ):
        self.samples = []
        if logger is None:
            logger = logging.getLogger(__name__)
        
        logger.info(f"Loading Stage 3 RLAIF dataset from {jsonl_path}")
        
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(tqdm(f, desc="Loading Stage 3")):
                if max_samples and i >= max_samples:
                    break
                
                try:
                    sample = json.loads(line.strip())
                    
                    # ✅ FIX: Handle candidates structure from generated dataset
                    if 'query' not in sample or 'candidates' not in sample:
                        continue

                    # Extract best candidate (highest reward_score)
                    candidates = sample.get('candidates', [])
                    if len(candidates) == 0:
                        continue

                    best_candidate = max(candidates, key=lambda x: x.get('reward_score', 0))
                    response_chain = best_candidate.get('reasoning_chain', [])
                    score = best_candidate.get('reward_score', 0.5)

                    if len(response_chain) < 2:
                        continue

                    # ✅ FIX: Check if sonar_embedding exists AND is not None
                    valid = all(
                        'sonar_embedding' in step 
                        and step['sonar_embedding'] is not None  # ✅ Check for None
                        and len(step['sonar_embedding']) == 1024 
                        for step in response_chain
                    )
                    
                    if not valid:
                        continue

                    self.samples.append({
                        'query': sample['query'],
                        'response_chain': response_chain,  # ✅ Extracted from best candidate
                        'score': score,                     # ✅ Extracted from best candidate
                        'domain': sample.get('domain', 'general')  # ✅ May be missing in old data
                    })
                
                except (json.JSONDecodeError, KeyError) as e:
                    continue
        
        logger.info(f"Loaded {len(self.samples)} Stage 3 RLAIF samples")
    
    # def _extract_best_candidate(self, sample):
    #     """
    #     Extract best candidate from multi-candidate format.
    #     Handles both old and new dataset formats.
    #     """
    #     # New format: {"candidates": [...]}
    #     if 'candidates' in sample:
    #         candidates = sample['candidates']
    #         if len(candidates) > 0:
    #             best = max(candidates, key=lambda x: x.get('reward_score', 0))
    #             return {
    #                 'response_chain': best.get('reasoning_chain', []),
    #                 'score': best.get('reward_score', 0.5),
    #                 'domain': sample.get('domain', 'general')
    #             }
        
    #     # Old format: direct keys
    #     return {
    #         'response_chain': sample.get('response_chain', []),
    #         'score': sample.get('score', 0.5),
    #         'domain': sample.get('domain', 'general')
    #     }

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        chain = sample['response_chain']
        
        embeddings = np.array([step['sonar_embedding'] for step in chain], dtype=np.float32)
        compartments = [step['compartment'] for step in chain]
        hierarchies = [step['hierarchy'] for step in chain]
        
        return {
            'embeddings': torch.from_numpy(embeddings),
            'compartments': compartments,
            'hierarchies': hierarchies,
            'score': torch.tensor(sample['score'], dtype=torch.float32),
            'query': sample['query'],
            'domain': sample['domain']
        }


def collate_stage3(batch):
    """Collate function for Stage 3 RLAIF"""
    max_len = max(item['embeddings'].shape[0] for item in batch)
    
    padded_embeddings = []
    attention_masks = []
    compartments_padded = []
    hierarchies_padded = []
    scores = []
    
    for item in batch:
        seq_len = item['embeddings'].shape[0]
        pad_len = max_len - seq_len
        
        padded_emb = F.pad(item['embeddings'], (0, 0, 0, pad_len))
        padded_embeddings.append(padded_emb)
        
        mask = torch.cat([torch.ones(seq_len), torch.zeros(pad_len)])
        attention_masks.append(mask)
        
        compartments_padded.append(item['compartments'] + [''] * pad_len)
        hierarchies_padded.append(item['hierarchies'] + [''] * pad_len)
        scores.append(item['score'])
    
    return {
        'embeddings': torch.stack(padded_embeddings),
        'attention_mask': torch.stack(attention_masks),
        'compartments': compartments_padded,
        'hierarchies': hierarchies_padded,
        'scores': torch.stack(scores),
        'queries': [item['query'] for item in batch],
        'domains': [item['domain'] for item in batch]
    }

def collate_stage1(batch):
    """Collate function for Stage 1 (single embeddings)"""
    return {
        'clean_embedding': torch.stack([item['clean_embedding'] for item in batch]),
        'noisy_embedding': torch.stack([item['noisy_embedding'] for item in batch]),
        'timestep': torch.stack([item['timestep'] for item in batch]),
        'alpha_t': torch.stack([item['alpha_t'] for item in batch]),
        'compartments': [item['compartment'] for item in batch],
        'hierarchies': [item['hierarchy'] for item in batch],
        'precisions': [item['precision'] for item in batch],
        'importances': torch.stack([item['importance'] for item in batch])
    }


def collate_stage2(batch):
    """Collate function for Stage 2 (variable-length sequences)"""
    # Find max sequence length
    max_len = max(item['embeddings'].shape[0] for item in batch)
    
    # Pad sequences
    padded_embeddings = []
    attention_masks = []
    compartments_padded = []
    hierarchies_padded = []
    precisions_padded = []
    importances_padded = []
    
    for item in batch:
        seq_len = item['embeddings'].shape[0]
        pad_len = max_len - seq_len
        
        # Pad embeddings
        padded_emb = F.pad(item['embeddings'], (0, 0, 0, pad_len))
        padded_embeddings.append(padded_emb)
        
        # Create attention mask (1 = valid, 0 = padding)
        mask = torch.cat([torch.ones(seq_len), torch.zeros(pad_len)])
        attention_masks.append(mask)
        
        # Pad metadata (use empty strings for padding)
        compartments_padded.append(item['compartments'] + [''] * pad_len)
        hierarchies_padded.append(item['hierarchies'] + [''] * pad_len)
        precisions_padded.append(item['precisions'] + [''] * pad_len)
        
        # Pad importances
        padded_imp = F.pad(item['importances'], (0, pad_len))
        importances_padded.append(padded_imp)
    
    return {
        'embeddings': torch.stack(padded_embeddings),  # [batch, max_len, 1024]
        'attention_mask': torch.stack(attention_masks),  # [batch, max_len]
        'compartments': compartments_padded,
        'hierarchies': hierarchies_padded,
        'precisions': precisions_padded,
        'importances': torch.stack(importances_padded),
        'queries': [item['query'] for item in batch],
        'domains': [item['domain'] for item in batch]
    }

# ═══════════════════════════════════════════════════════════════════
# LOSS FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing
    
    def forward(self, pred, target):
        n_classes = pred.size(-1)
        log_probs = F.log_softmax(pred, dim=-1)
        
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (n_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        
        return torch.mean(torch.sum(-true_dist * log_probs, dim=-1))

class SCMLoss(nn.Module):
    """
    SCM Multi-Objective Loss Function
    
    L_SCM = α × L_concept + β × L_domain + γ × L_efficiency
    """
    
    def __init__(self, config: SCMConfig):
        super().__init__()
        self.config = config
        
        # Label mappings in order (alphabetical, deterministic)
        COMPARTMENT_ORDER = ['CONCEPTUAL', 'CONTEXTUAL', 'EPISODIC', 'FACTUAL', 'PROCEDURAL']
        HIERARCHY_ORDER = ['GENERAL', 'GRANULAR', 'INTERMEDIATE']

        self.compartment_to_idx = {c: i for i, c in enumerate(COMPARTMENT_ORDER)}
        self.hierarchy_to_idx = {h: i for i, h in enumerate(HIERARCHY_ORDER)}
        self.ce_loss = LabelSmoothingCrossEntropy(smoothing=0.1)
        
    def forward(
        self, 
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        stage: str = 'stage1'
    ) -> Dict[str, torch.Tensor]:
        """
        Compute multi-objective loss
        
        Args:
            outputs: Model outputs
            targets: Ground truth targets
            stage: 'stage1' or 'stage2'
        
        Returns:
            dict with loss components
        """
        if stage == 'stage1':
            return self._stage1_loss(outputs, targets)
        elif stage == 'stage2':
            return self._stage2_loss(outputs, targets)
        else:
            raise ValueError(f"Unknown stage: {stage}")
    
    def _stage1_loss(self, outputs, targets):
        """Stage 1: Denoising + Reconstruction Loss"""
        # Concept reconstruction loss
        compressed = outputs['compressed_concepts']  # [B, 1, 384]
        clean_embedding = targets['clean_embedding'].unsqueeze(1)  # [B, 1, 1024]
        
        # Expand compressed concept back to SONAR space
        reconstructed = targets['compressor'].expand_concept(compressed)
        l_concept = F.mse_loss(reconstructed, clean_embedding)
        
        # Domain classification loss (compartment + hierarchy)
        comp_logits = outputs['compartment_logits'].squeeze(1)  # [B, 5]
        hier_logits = outputs['hierarchy_logits'].squeeze(1)  # [B, 3]
        
        # Convert labels to indices
        comp_labels = torch.tensor(
            [self.compartment_to_idx.get(c, 0) for c in targets['compartments']],
            device=comp_logits.device
        )
        hier_labels = torch.tensor(
            [self.hierarchy_to_idx.get(h, 0) for h in targets['hierarchies']],
            device=hier_logits.device
        )
        
        
        l_comp = self.ce_loss(comp_logits, comp_labels)
        l_hier = self.ce_loss(hier_logits, hier_labels)
        l_domain = (l_comp + l_hier) / 2
        
        # L_efficiency = λ × param_count + μ × inference_time
        # Efficiency loss (penalize large embeddings)
        l_efficiency = torch.mean(compressed ** 2)  # ✅ Mean squared magnitude
        
        # Total loss
        total_loss = (
            self.config.ALPHA_CONCEPT * l_concept +
            self.config.BETA_DOMAIN * l_domain +
            self.config.GAMMA_EFFICIENCY * l_efficiency
        )
        
        return {
            'loss': total_loss,
            'l_concept': l_concept.item(),
            'l_domain': l_domain.item(),
            'l_efficiency': l_efficiency.item()
        }
    
    def _stage2_loss(self, outputs, targets):
        """Stage 2: Autoregressive Concept Prediction"""
        # Shift targets by 1
        # ✅ FIX: Get compressed concepts (B, L, 384)
        all_concepts = outputs['compressed_concepts']  # (B, L, 384)
        
        # ✅ FIX: Slice predictions to exclude last position (B, L-1, 384)
        # concept_predictions has shape (B, L, 384), we need (B, L-1, 384)
        pred_concepts = outputs['concept_predictions'][:, :-1, :]  # (B, L-1, 384)
        
        # ✅ FIX: Target is next concept (shift by 1)
        target_concepts = all_concepts[:, 1:, :]  # (B, L-1, 384)
        
        # ✅ FIX: Mask also needs to match (B, L-1, 1)
        mask = targets['attention_mask'][:, 1:].unsqueeze(-1)  # (B, L-1, 1)
        
        # Precision-aware weighting
        batch_size, seq_len = targets['attention_mask'].shape
        precision_weights = torch.ones(batch_size, seq_len, device=pred_concepts.device)
        
        # Create FP32 mask
        fp32_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        for b in range(batch_size):
            for s in range(seq_len):
                if targets['precisions'][b][s] == 'fp32':
                    fp32_mask[b, s] = True
        
        fp32_mask = fp32_mask.to(pred_concepts.device)
        precision_weights[fp32_mask] = self.config.FP32_PRECISION_WEIGHT
        
        # ✅ Apply weights to mask (combine attention + precision)
        weighted_mask = mask * precision_weights[:, :-1].unsqueeze(-1)  # [B, L-1, 1]
        
        # Concept prediction loss (with precision weighting)
        l_concept = F.mse_loss(
            pred_concepts * mask,
            target_concepts * mask,
            reduction='sum'
        ) / (mask.sum() + 1e-8)  # ✅ Normalize by valid positions only  # ✅ Correct normalization with epsilon
        
        # Domain classification loss (same as before)
        comp_logits = outputs['compartment_logits']  # (B, seq_len, 5)
        hier_logits = outputs['hierarchy_logits']  # (B, seq_len, 3)
        
        batch_size, seq_len = targets['attention_mask'].shape
        
        comp_labels = []
        hier_labels = []
        valid_positions = []
        
        for b in range(batch_size):
            for s in range(seq_len):
                if targets['attention_mask'][b, s] > 0:  # ✅ Only valid positions
                    comp_labels.append(
                        self.compartment_to_idx.get(targets['compartments'][b][s], 0)
                    )
                    hier_labels.append(
                        self.hierarchy_to_idx.get(targets['hierarchies'][b][s], 0)
                    )
                    valid_positions.append(b * seq_len + s)
        
        if len(valid_positions) == 0:
            # No valid positions (shouldn't happen)
            l_domain = torch.tensor(0.0, device=comp_logits.device)
        else:
            # Flatten and select only valid positions
            comp_logits_flat = comp_logits.view(-1, len(self.compartment_to_idx))
            hier_logits_flat = hier_logits.view(-1, len(self.hierarchy_to_idx))
            
            comp_logits_valid = comp_logits_flat[valid_positions]
            hier_logits_valid = hier_logits_flat[valid_positions]
            
            comp_labels_tensor = torch.tensor(comp_labels, device=comp_logits.device)
            hier_labels_tensor = torch.tensor(hier_labels, device=hier_logits.device)
            
            l_comp = F.cross_entropy(comp_logits_valid, comp_labels_tensor)
            l_hier = F.cross_entropy(hier_logits_valid, hier_labels_tensor)
            l_domain = (l_comp + l_hier) / 2
        
        # Total loss
        total_loss = (
            self.config.ALPHA_CONCEPT * l_concept +
            self.config.BETA_DOMAIN * l_domain
        )
        
        return {
            'loss': total_loss,
            'l_concept': l_concept.item(),
            'l_domain': l_domain.item(),
            'precision_weight': precision_weights.mean().item()
        }

# ═══════════════════════════════════════════════════════════════════
# TRAINER
# ═══════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════
# TRAINING UTILITIES
# ═══════════════════════════════════════════════════════════════════

class EarlyStopping:
    """Early stopping to prevent overfitting"""
    def __init__(self, patience: int = 5, min_delta: float = 0.001, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.logger = logging.getLogger(__name__)
        
    def __call__(self, val_loss: float) -> bool:
        score = -val_loss if self.mode == 'min' else val_loss
        
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            self.logger.info(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
        
        return self.early_stop

class ModelEMA:
    """Exponential Moving Average of model parameters"""
    def __init__(self, model, decay=0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        """Apply EMA weights for evaluation"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        """Restore original weights"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]

class SCMMetrics:
    """Track multiple evaluation metrics"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.total_loss = 0.0
        self.concept_mse = 0.0
        self.domain_acc = 0.0
        self.embedding_sim = 0.0
        self.count = 0
    
    def update(self, loss_val, l_concept):
        self.total_loss += loss_val
        self.concept_mse += l_concept
        self.count += 1
    
    def compute(self):
        if self.count == 0:
            return {'loss': 0.0, 'perplexity': float('inf'), 'concept_mse': 0.0}
        
        avg_loss = self.total_loss / self.count
        
        if isinstance(avg_loss, torch.Tensor):
            avg_loss_value = avg_loss.cpu().item()
        else:
            avg_loss_value = avg_loss
            
        return {
            'loss': avg_loss_value,
            'perplexity': np.exp(avg_loss_value),
            'concept_mse': self.concept_mse / self.count
        }

class SCMTrainer:
    """SCM Training Pipeline"""
    
    def __init__(self, config: SCMConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.device = config.DEVICE
        
        # Initialize model
        self.model = SCMTransformer(config).to(self.device)
        self.logger.info(f"Model initialized: {sum(p.numel() for p in self.model.parameters())/1e6:.1f}M parameters")
        
        # Initialize optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
        
        # ✅ NEW: Improved scheduler with warmup + plateau detection
        # from torch.optim.lr_scheduler import ReduceLROnPlateau
        # self.main_scheduler = ReduceLROnPlateau(
        #     self.optimizer, mode='min', factor=0.5, patience=3, verbose=True
        # )
        
        from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

        self.main_scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=500,      # Restart every 500 steps
            T_mult=2,     # Double period after each restart
            eta_min=1e-6  # Minimum LR
        )
        
        self.warmup_steps = config.WARMUP_STEPS
        self.warmup_scheduler = None  # Will be set in training
        
        # Initialize loss function
        self.criterion = SCMLoss(config)
        
        # Training state
        self.global_step = 0
        self.best_loss = float('inf')
        
        # self.scaler = torch.GradScaler()  
        # ✅ ADD: Mixed precision scaler
        # self.use_amp = torch.cuda.is_available()  # Auto-enable on GPU
        torch.set_default_device('cuda')
        torch.get_default_device()
        
        self.use_amp = True
        self.scaler = torch.amp.GradScaler() if self.use_amp else torch.GradScaler()

        if self.use_amp:
            self.logger.info("✅ Mixed precision training enabled (FP16)")
            
        # ✅ NEW: EMA and early stopping
        self.ema = ModelEMA(self.model, decay=0.9999)
        self.early_stopping = {}  # Per-stage early stopping
        self.grad_norm_history = []
    
    def set_warmup_steps(self, total_steps: int):
        """Set warmup to 10% of total training steps"""
        self.warmup_steps = max(100, int(0.1 * total_steps))
        self.logger.info(f"Warmup steps: {self.warmup_steps} (10% of {total_steps})")
       
       
    def find_optimal_lr(self, dataset, num_steps=100):
        """LR range test (Leslie Smith method)"""
        from torch.utils.data import DataLoader
        
        if self.device.type == 'cuda':
            generator = torch.Generator(device=self.device)
        else:
            generator = None

        temp_loader = DataLoader(
            dataset, 
            batch_size=self.config.BATCH_SIZE,
            shuffle=False, 
            collate_fn=collate_stage1,
            num_workers=0,
            pin_memory=False,  # ✅ Must be False (data already on CUDA)
            generator=generator  # ✅ Add CUDA generator
        )
        
        start_lr, end_lr = 1e-7, 1e-2
        lr_mult = (end_lr / start_lr) ** (1 / num_steps)
        lr = start_lr
        
        lrs, losses = [], []
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        for i, batch in enumerate(temp_loader):
            if i >= num_steps:
                break
            
            self.optimizer.zero_grad()
            clean_emb = batch['clean_embedding'].unsqueeze(1).to(self.device)
            noisy_emb = batch['noisy_embedding'].unsqueeze(1).to(self.device)
            
            outputs = self.model(noisy_emb, use_causal_mask=False)
            loss_dict = self.criterion(
                outputs,
                {
                    'clean_embedding': clean_emb.squeeze(1),
                    'compartments': batch['compartments'],
                    'hierarchies': batch['hierarchies'],
                    'precisions': batch['precisions'],
                    'compressor': self.model.compressor
                },
                stage='stage1'
            )
            
            loss = loss_dict['loss']
            loss.backward()
            self.optimizer.step()
            
            lrs.append(lr)
            losses.append(loss.item())
            
            lr *= lr_mult
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        
        # Find LR with steepest descent
        gradients = np.gradient(losses)
        optimal_idx = np.argmin(gradients)
        optimal_lr = lrs[optimal_idx]
        
        self.logger.info(f"📊 Optimal LR: {optimal_lr:.2e} (current: {self.config.LEARNING_RATE:.2e})")
        
        return optimal_lr, lrs, losses

    def enable_swa(self, swa_start_epoch=2):
        """Enable Stochastic Weight Averaging"""
        self.swa_model = AveragedModel(self.model)
        self.swa_scheduler = SWALR(
            self.optimizer,
            swa_lr=self.config.LEARNING_RATE * 0.1
        )
        self.swa_start_epoch = swa_start_epoch
        self.logger.info(f"✅ SWA enabled (starts epoch {swa_start_epoch})")
 
    def _validate_epoch(self, val_loader, stage: str = 'stage1') -> float:
        """Run validation epoch"""
        self.model.eval()
        val_metrics = SCMMetrics()
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation", leave=False):
                if stage == 'stage1':
                    clean_emb = batch['clean_embedding'].unsqueeze(1).to(self.device)
                    noisy_emb = batch['noisy_embedding'].unsqueeze(1).to(self.device)
                    
                    outputs = self.model(noisy_emb, use_causal_mask=False)
                    loss_dict = self.criterion(
                        outputs,
                        {
                            'clean_embedding': clean_emb.squeeze(1),
                            'compartments': batch['compartments'],
                            'hierarchies': batch['hierarchies'],
                            'precisions': batch['precisions'],
                            'compressor': self.model.compressor
                        },
                        stage='stage1'
                    )
                    
                    val_metrics.update(loss_dict['loss'], loss_dict['l_concept'])
                
                elif stage == 'stage2':
                    embeddings = batch['embeddings'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    
                    outputs = self.model(embeddings, attention_mask=attention_mask)
                    loss_dict = self.criterion(
                        outputs,
                        {
                            'attention_mask': attention_mask,
                            'compartments': batch['compartments'],
                            'hierarchies': batch['hierarchies'],
                            'precisions': batch['precisions']
                        },
                        stage='stage2'
                    )
                    
                    val_metrics.update(loss_dict['loss'], loss_dict['l_concept'])
        
        self.model.train()
        computed_metrics = val_metrics.compute()
        perplexity = computed_metrics['perplexity']

        self.logger.info(f"Val Loss: {computed_metrics['loss']:.4f}, Perplexity: {perplexity:.2f}")

        return computed_metrics['loss']

    def _validate_stage3(self, val_loader) -> float:
        """Run validation for Stage 3 RLAIF"""
        self.model.eval()
        val_metrics = SCMMetrics()
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation", leave=False):
                embeddings = batch['embeddings'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                scores = batch['scores'].to(self.device)
                
                outputs = self.model(embeddings, attention_mask=attention_mask)
                
                # Compute concept loss
                pred_concepts = outputs['concept_predictions'][:, :-1, :]
                target_concepts = outputs['compressed_concepts'][:, 1:, :]
                mask = attention_mask[:, 1:].unsqueeze(-1)
                
                l_concept = F.mse_loss(
                    pred_concepts * mask,
                    target_concepts * mask,
                    reduction='sum'
                ) / (mask.sum() + 1e-8)
                
                # Weight by reward
                reward_weights = 1.0 - scores
                weighted_loss = l_concept * reward_weights.mean()
                
                val_metrics.update(weighted_loss.item(), l_concept.item())
        
        self.model.train()
        computed_metrics = val_metrics.compute()
        perplexity = computed_metrics['perplexity']
        
        self.logger.info(f"Val Loss: {computed_metrics['loss']:.4f}, Perplexity: {perplexity:.2f}")
        
        return computed_metrics['loss']

        
    def train_stage1(self, num_epochs: int = 50, max_samples: Optional[int] = None, 
                 resume: bool = True, val_split: float = 0.15):
        """
        Stage 1: Denoising Pre-training
        
        Trains concept compression and denoising in compressed space.
        """
        self.logger.info("="*80)
        self.logger.info("STAGE 1: DENOISING PRE-TRAINING")
        self.logger.info("="*80)
        
        full_dataset = Stage1Dataset(
            self.config.STAGE1_JSONL, 
            max_samples=max_samples, 
            logger=self.logger
        )
        
        start_epoch = 0
        if resume:
            resume_info = self.auto_resume("stage1")
            if resume_info:
                start_epoch = resume_info['epoch']
                # full_dataset = Stage1Dataset(self.config.STAGE1_JSONL, max_samples=max_samples, logger=self.logger)
    
        train_size = int((1 - val_split) * len(full_dataset))
        val_size = len(full_dataset) - train_size
        
        # ✅ Create CUDA-compatible generator
        if self.device.type == 'cuda':
            generator = torch.Generator(device='cuda').manual_seed(42)
        else:
            generator = torch.Generator().manual_seed(42)

        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, 
            [train_size, val_size],
            generator=generator  # ✅ CUDA generator
        )

        self.logger.info(f"Split: {len(train_dataset)} train, {len(val_dataset)} val")
        
        train_loader = DataLoader(
            train_dataset, batch_size=self.config.BATCH_SIZE, shuffle=False,
            collate_fn=collate_stage1, num_workers=0, pin_memory=False
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.config.BATCH_SIZE, shuffle=False,
            collate_fn=collate_stage1, num_workers=0, pin_memory=False
        )
        
        total_steps = len(train_loader) * num_epochs // self.config.ACCUMULATION_STEPS
        self.set_warmup_steps(total_steps)
        
        # ✅ Initialize early stopping
        early_stopping = EarlyStopping(patience=5, min_delta=0.001)
        
        self.model.train()
        
        for epoch in range(start_epoch, num_epochs):
            epoch_loss = 0.0
            epoch_metrics = {'l_concept': 0.0, 'l_domain': 0.0, 'l_efficiency': 0.0}
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
            self.optimizer.zero_grad()
            
            for batch_idx, batch in enumerate(pbar):  
                # Move to device
                clean_emb = batch['clean_embedding'].unsqueeze(1).to(self.device)  # [B, 1, 1024]
                noisy_emb = batch['noisy_embedding'].unsqueeze(1).to(self.device)
                
                # Forward pass (use noisy embeddings as input)
                with torch.amp.autocast(device_type=str(self.device), enabled=self.use_amp):  # Enable FP16
                    outputs = self.model(noisy_emb, use_causal_mask=False)
                    outputs['model'] = self.model  # Pass model for reconstruction
                    
                    # Compute loss
                    loss_dict = self.criterion(
                        outputs,
                        {
                            'clean_embedding': clean_emb.squeeze(1),
                            'compartments': batch['compartments'],
                            'hierarchies': batch['hierarchies'],
                            'precisions': batch['precisions'],
                            'compressor': self.model.compressor if not isinstance(self.model, nn.DataParallel) else self.model.module.compressor
                        },
                        stage='stage1'
                    )
                    
                    loss = loss_dict['loss'] / self.config.ACCUMULATION_STEPS

                # Backward (accumulate gradients)
                if self.use_amp:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                if (batch_idx + 1) % self.config.ACCUMULATION_STEPS == 0:
                    if self.use_amp:
                        self.scaler.unscale_(self.optimizer)
                        
                        # Check gradient norm
                        total_norm = torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), float('inf')
                        )
                        
                        # ✅ Skip only if infinite/NaN
                        if torch.isinf(total_norm) or torch.isnan(total_norm):
                            self.logger.warning(f"⚠️ Skipping step - grad norm: {total_norm}")
                            self.optimizer.zero_grad()
                            self.scaler.update()
                            continue
                        
                        # ✅ Always clip (even if large)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), 5.0  # More aggressive clipping
                        )
                        
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.config.GRADIENT_CLIP
                        )
                        self.optimizer.step()
                        
                    if epoch >= self.swa_start_epoch:
                        self.swa_model.update_parameters(self.model)
                        self.swa_scheduler.step()
                        
   
                    # After line 1481 (after optimizer.step())
                    if self.global_step % 1000 == 0:
                        # Gradually increase weight decay
                        new_wd = min(0.1, self.config.WEIGHT_DECAY * (1 + self.global_step / 100000))
                        for param_group in self.optimizer.param_groups:
                            param_group['weight_decay'] = new_wd

                    
                    self.optimizer.zero_grad()
                    
                    # ✅ Warmup scheduler
                    if self.global_step < self.warmup_steps:
                        warmup_factor = (self.global_step + 1) / self.warmup_steps
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = self.config.LEARNING_RATE_STAGE1 * warmup_factor
                    
                    # ✅ Update EMA
                    self.ema.update()
                
                self.global_step += 1
                
                
                # Update metrics
                epoch_loss += loss.item() * self.config.ACCUMULATION_STEPS
                for key in epoch_metrics:
                    epoch_metrics[key] += loss_dict[key]
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': loss.item() * self.config.ACCUMULATION_STEPS,
                    'l_concept': loss_dict['l_concept'],
                    'lr': self.optimizer.param_groups[0]['lr']
                })
                
                if batch_idx > 0 and batch_idx % 500 == 0:
                    self.save_checkpoint(
                        f"stage1_epoch{epoch+1}_batch{batch_idx}.pt",
                        epoch=epoch,
                        batch_idx=batch_idx,
                        stage="stage1"
                    )
                            
            # Epoch summary
            avg_train_loss = epoch_loss / len(train_loader)
            self.ema.apply_shadow()
            val_loss = self._validate_epoch(val_loader, stage='stage1')
            self.ema.restore()
        
            self.logger.info(f"Epoch {epoch+1} - Train: {avg_train_loss:.4f}  (PPL: {np.exp(avg_train_loss):.2f}), Val: {val_loss:.4f} (PPL: {np.exp(val_loss):.2f})")
            
            # ✅ Update scheduler based on validation loss
            self.main_scheduler.step()
            
            # Save best model
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.save_checkpoint(f"stage1_best.pt", epoch=epoch+1, stage="stage1")
            
            # ✅ Early stopping check
            if early_stopping(val_loss):
                self.logger.info(f"⏹️ Early stopping at epoch {epoch+1}")
                break
        
        self.logger.info("✅ Stage 1 training complete")
        torch.optim.swa_utils.update_bn(train_loader, self.swa_model)

        self.save_checkpoint(f"stage1_final_epoch.pt", stage="stage1")
        
    def train_stage2(self, num_epochs: int = 40, max_samples: Optional[int] = None, 
            resume: bool = True, val_split: float = 0.15):        
        """
        Stage 2: Supervised Fine-Tuning
        
        Trains autoregressive concept prediction on reasoning chains.
        """
        self.logger.info("="*80)
        self.logger.info("STAGE 2: SUPERVISED FINE-TUNING")
        self.logger.info("="*80)
        
        full_dataset = Stage2Dataset(
            self.config.STAGE2_JSONL, 
            max_samples=max_samples, 
            logger=self.logger
        )
        
        start_epoch = 0
        if resume:
            resume_info = self.auto_resume("stage2")
            if resume_info:
                start_epoch = resume_info['epoch']
        
        # full_dataset = Stage2Dataset(self.config.STAGE2_JSONL, max_samples=max_samples, logger=self.logger)
        train_size = int((1 - val_split) * len(full_dataset))
        val_size = len(full_dataset) - train_size
        # ✅ Create CUDA-compatible generator
        if self.device.type == 'cuda':
            generator = torch.Generator(device='cuda').manual_seed(42)
        else:
            generator = torch.Generator().manual_seed(42)

        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, 
            [train_size, val_size],
            generator=generator  # ✅ CUDA generator
        )

        
        self.logger.info(f"Split: {len(train_dataset)} train, {len(val_dataset)} val")
        
        train_loader = DataLoader(
            train_dataset, batch_size=self.config.BATCH_SIZE, shuffle=False,
            collate_fn=collate_stage2, num_workers=0, pin_memory=False
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.config.BATCH_SIZE, shuffle=False,
            collate_fn=collate_stage2, num_workers=0, pin_memory=False
        )
        
        early_stopping = EarlyStopping(patience=5, min_delta=0.001)
        self.model.train()
        
        # ✅ Set Stage 2 LR ONCE at start (before training loop)
        target_lr = 1e-4  # Good starting point for Stage 2
        self.logger.info(f"🔧 Setting Stage 2 LR: {target_lr:.2e}")
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = target_lr
        
        # self.logger.info("🔧 Reducing LR for Stage 2 (long sequences)")
        # for param_group in self.optimizer.param_groups:
        #     param_group['lr'] = self.config.LEARNING_RATE_STAGE2 * 0.1  # 3e-5
        # self.logger.info(f"Stage 2 LR: {self.optimizer.param_groups[0]['lr']:.2e}")
        
        for epoch in range(start_epoch, num_epochs):
            epoch_loss = 0.0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
            self.optimizer.zero_grad()
            
            for batch_idx, batch in enumerate(pbar):
                embeddings = batch['embeddings'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                with torch.amp.autocast(device_type='cuda', enabled=self.use_amp):
                    outputs = self.model(embeddings, attention_mask=attention_mask)
                    
                    domain_vocab_loss = self.model.domain_vocab.compute_domain_loss(
                        outputs['compressed_concepts'], batch['domains'], temperature=0.1
                    )
                    
                    loss_dict = self.criterion(
                        outputs,
                        {
                            'attention_mask': attention_mask,
                            'compartments': batch['compartments'],
                            'hierarchies': batch['hierarchies'],
                            'precisions': batch['precisions']
                        },
                        stage='stage2'
                    )
                    
                    loss = (loss_dict['loss'] + 0.1 * domain_vocab_loss) / self.config.ACCUMULATION_STEPS
                
                if self.use_amp:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                    
                # After loss.backward() (line ~1948):
                if (batch_idx + 1) % self.config.ACCUMULATION_STEPS == 0:
                    if self.use_amp:
                        self.scaler.unscale_(self.optimizer)
                        
                        # ✅ Check gradient norm before clipping
                        total_norm = torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), float('inf')
                        )
                        
                        if total_norm > 1000.0 or torch.isinf(total_norm) or torch.isnan(total_norm):  # ✅ Skip if gradients explode
                            self.logger.warning(f"⚠️ Skipping step - grad norm: {total_norm:.2f}")
                            self.optimizer.zero_grad()  # ✅ Clear gradients
                            self.scaler.update()
                            continue
                    
                        # Now apply gradient clipping
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.config.GRADIENT_CLIP
                        )
                        
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        # Non-AMP path
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.config.GRADIENT_CLIP
                        )
                        self.optimizer.step()
                    
                    self.optimizer.zero_grad()
                
                # Update EMA
                self.ema.update()
                self.global_step += 1

                # Update metrics
                epoch_loss += loss.item() * self.config.ACCUMULATION_STEPS
                pbar.set_postfix(loss=loss.item() * self.config.ACCUMULATION_STEPS)

                
                if batch_idx > 0 and batch_idx % 500 == 0:
                    self.save_checkpoint(
                        f"stage2_epoch{epoch+1}_batch{batch_idx}.pt",
                        epoch=epoch, batch_idx=batch_idx, stage="stage2"
                    )
            
            # Validate
            avg_train_loss = epoch_loss / len(train_loader)
            self.ema.apply_shadow()
            val_loss = self._validate_epoch(val_loader, stage='stage2')
            self.ema.restore()
            
            self.logger.info(f"Epoch {epoch+1} - Train: {avg_train_loss:.4f}, Val: {val_loss:.4f}")
            self.main_scheduler.step()
                        
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.save_checkpoint(f"stage2_best.pt", epoch=epoch+1, stage="stage2")
            
            if early_stopping(val_loss):
                self.logger.info(f"⏹️ Early stopping at epoch {epoch+1}")
                break
        
        self.logger.info("✅ Stage 2 training complete")
        self.save_checkpoint(f"stage2_final_epoch.pt", stage="stage2")
        
    def train_stage3(self, num_epochs: int = 5, max_samples: Optional[int] = None, 
                 resume: bool = True, val_split: float = 0.15):
        """
        Stage 3: RLAIF (Reinforcement Learning from AI Feedback)
        
        Optimizes model using reward signals from AI-generated quality scores.
        """
        self.logger.info("="*80)
        self.logger.info("STAGE 3: RLAIF TRAINING")
        self.logger.info("="*80)
        
        full_dataset = Stage3RLAIFDataset(
            self.config.STAGE3_JSONL, 
            max_samples=max_samples, 
            logger=self.logger
        )
        
        start_epoch = 0
        if resume:
            resume_info = self.auto_resume("stage3")
            if resume_info:
                start_epoch = resume_info['epoch']
        
        # Load dataset
        # full_dataset = Stage3RLAIFDataset(
        #     self.config.STAGE3_JSONL, 
        #     max_samples=max_samples,
        #     logger=self.logger
        # )
        train_size = int((1 - val_split) * len(full_dataset))
        val_size = len(full_dataset) - train_size
        # ✅ Create CUDA-compatible generator
        if self.device.type == 'cuda':
            generator = torch.Generator(device='cuda').manual_seed(42)
        else:
            generator = torch.Generator().manual_seed(42)

        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, 
            [train_size, val_size],
            generator=generator  # ✅ CUDA generator
        )

        self.logger.info(f"Split: {len(train_dataset)} train, {len(val_dataset)} val")
        
        train_loader = DataLoader(
            train_dataset, batch_size=self.config.BATCH_SIZE, shuffle=False,
            collate_fn=collate_stage3, num_workers=0, pin_memory=False
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.config.BATCH_SIZE, shuffle=False,
            collate_fn=collate_stage3, num_workers=0, pin_memory=False
        )
        
        early_stopping = EarlyStopping(patience=5, min_delta=0.001)
        
        self.model.train()
        
        self.logger.info("🔧 Reducing LR for Stage 3 (long sequences)")
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.config.LEARNING_RATE_STAGE3 * 0.1  # 3e-5
        self.logger.info(f"Stage 3 LR: {self.optimizer.param_groups[0]['lr']:.2e}")
        
        for epoch in range(start_epoch, num_epochs):
            epoch_loss = 0.0
            epoch_reward = 0.0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
            
            self.optimizer.zero_grad()
            
            for batch_idx, batch in enumerate(pbar):
                embeddings = batch['embeddings'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                scores = batch['scores'].to(self.device)  # AI quality scores
                
                # Forward pass
                with torch.amp.autocast(device_type=str(self.device), enabled=self.use_amp): # Enable FP16
                    outputs = self.model(embeddings, attention_mask=attention_mask)
                    domain_vocab = (
                        self.model.domain_vocab if not isinstance(self.model, nn.DataParallel)
                        else self.model.module.domain_vocab
                    )
                    # ✅ ADD: Domain vocabulary loss for RLAIF
                    domain_vocab_loss = domain_vocab.compute_domain_loss(
                        outputs['compressed_concepts'],
                        batch['domains'],
                        temperature=0.1
                    )

                    # Compute concept prediction quality
                    pred_concepts = outputs['concept_predictions'][:, :-1, :]
                    target_concepts = outputs['compressed_concepts'][:, 1:, :]
                    mask = attention_mask[:, 1:].unsqueeze(-1)
                    
                    # Concept loss
                    l_concept = F.mse_loss(
                        pred_concepts * mask,
                        target_concepts * mask,
                        reduction='sum'
                    ) / (mask.sum() + 1e-8)
                    
                    # ✅ REWARD-WEIGHTED LOSS (RLAIF)
                    # High-quality responses get lower loss weight (already good)
                    # Low-quality responses get higher weight (need improvement)
                    reward_weights = 1.0 - scores  # Invert: low score = high weight
                    # weighted_loss = (l_concept * reward_weights.mean()) + 0.05 * domain_vocab_loss
                
                # Backward pass
                self.optimizer.zero_grad()
                
                # Scale loss
                base_loss = (l_concept * reward_weights.mean()) + 0.05 * domain_vocab_loss

                # Scale for accumulation
                weighted_loss = base_loss / self.config.ACCUMULATION_STEPS

                # Backward
                if self.use_amp:
                    self.scaler.scale(weighted_loss).backward()
                else:
                    weighted_loss.backward()

                # Update every ACCUMULATION_STEPS
                if (self.global_step + 1) % self.config.ACCUMULATION_STEPS == 0:
                    if self.use_amp:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.GRADIENT_CLIP)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.GRADIENT_CLIP)
                        self.optimizer.step()
                    
                    self.optimizer.zero_grad()
                self.ema.update()
                
                # Update metrics
                epoch_loss += weighted_loss.item()
                epoch_reward += scores.mean().item()
                
                pbar.set_postfix({
                    'loss': weighted_loss.item(),
                    'avg_score': scores.mean().item(),
                    'lr': self.optimizer.param_groups[0]['lr']
                })
                
                self.global_step += 1
                
                if batch_idx > 0 and batch_idx % 500 == 0:
                    self.save_checkpoint(
                        f"stage3_epoch{epoch+1}_batch{batch_idx}.pt",
                        epoch=epoch,
                        batch_idx=batch_idx,
                        stage="stage3"
                    )
            
            # Epoch summary
            avg_train_loss = epoch_loss / len(train_loader)
            self.ema.apply_shadow()
            val_loss = self._validate_stage3(val_loader)  # Need to add this method
            self.ema.restore()
            
            self.logger.info(f"Epoch {epoch+1} - Train: {avg_train_loss:.4f}, Val: {val_loss:.4f}")
            self.main_scheduler.step()
            
            # Save checkpoint
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.save_checkpoint(f"stage3_best.pt", epoch=epoch+1, stage="stage3")
            
            if early_stopping(val_loss):
                self.logger.info(f"⏹️ Early stopping at epoch {epoch+1}")
                break
        
        self.logger.info("✅ Stage 3 RLAIF training complete")
        
        # Always save final checkpoint
        self.save_checkpoint(f"stage3_final_epoch.pt", stage="stage3")
    
    # def save_checkpoint(self, filename: str):
    #     """Save model checkpoint"""
    #     checkpoint_path = self.config.CHECKPOINT_DIR / filename
    #     torch.save({
    #         'model_state_dict': self.model.state_dict(),
    #         'optimizer_state_dict': self.optimizer.state_dict(),
    #         'scheduler_state_dict': self.scheduler.state_dict(),
    #         'global_step': self.global_step,
    #         'best_loss': self.best_loss,
    #         'config': self.config,
    #         'compartment_to_idx': self.criterion.compartment_to_idx,  # ✅ Save mappings
    #         'hierarchy_to_idx': self.criterion.hierarchy_to_idx
    #     }, checkpoint_path)
    #     self.logger.info(f"💾 Checkpoint saved: {checkpoint_path}")
      
      
    def evaluate_test_set(self, test_jsonl: str, stage: str = 'stage1'):
        """Final evaluation on held-out test set"""
        self.logger.info("="*80)
        self.logger.info(f"TEST SET EVALUATION - {stage.upper()}")
        self.logger.info("="*80)
        
        if stage == 'stage1':
            test_dataset = Stage1Dataset(test_jsonl, logger=self.logger)
            collate_fn = collate_stage1
        elif stage == 'stage2':
            test_dataset = Stage2Dataset(test_jsonl, logger=self.logger)
            collate_fn = collate_stage2
        else:
            test_dataset = Stage3RLAIFDataset(test_jsonl, logger=self.logger)
            collate_fn = collate_stage3
        
        test_loader = DataLoader(
            test_dataset, batch_size=self.config.BATCH_SIZE,
            shuffle=False, collate_fn=collate_fn, num_workers=0
        )
        
        # Use EMA weights for final evaluation
        self.ema.apply_shadow()
        
        if stage == 'stage3':
            test_loss = self._validate_stage3(test_loader)
        else:
            test_loss = self._validate_epoch(test_loader, stage=stage)
        
        self.ema.restore()
        
        test_perplexity = np.exp(test_loss)
        
        self.logger.info(f"📊 Test Loss: {test_loss:.4f}")
        self.logger.info(f"📊 Test Perplexity: {test_perplexity:.2f}")
        
        return test_loss, test_perplexity
  
    def save_checkpoint(self, filename: str, epoch: int = 0, batch_idx: int = 0, stage: str = "unknown"):
        """Save model checkpoint with full training state"""
        checkpoint_path = self.config.CHECKPOINT_DIR / filename
        torch.save({
            # Model state
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.swa_scheduler.state_dict(),
            
            # Training progress
            'global_step': self.global_step,
            'best_loss': self.best_loss,
            'epoch': epoch,
            'batch_idx': batch_idx,
            'stage': stage,
            
            # Config and mappings
            'config': self.config,
            'compartment_to_idx': self.criterion.compartment_to_idx,
            'hierarchy_to_idx': self.criterion.hierarchy_to_idx,
            
            # Metadata
            'timestamp': datetime.now().isoformat()
        }, checkpoint_path)
        self.logger.info(f"💾 Checkpoint saved: {checkpoint_path} (epoch {epoch}, step {self.global_step})")

    
    # def load_checkpoint(self, filename: str):
    #     """Load model checkpoint"""
    #     checkpoint_path = self.config.CHECKPOINT_DIR / filename
    #     checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
    #     self.model.load_state_dict(checkpoint['model_state_dict'])
    #     self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #     self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    #     self.global_step = checkpoint['global_step']
    #     self.best_loss = checkpoint['best_loss']
        
    #     self.logger.info(f"📂 Checkpoint loaded: {checkpoint_path}")


    def load_checkpoint(self, filename: str) -> Dict:
        """Load model checkpoint and return training state"""
        checkpoint_path = self.config.CHECKPOINT_DIR / filename
        
        if not checkpoint_path.exists():
            self.logger.warning(f"⚠️ Checkpoint not found: {checkpoint_path}")
            return None
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # Restore model and optimizer
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.swa_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Restore training state
        self.global_step = checkpoint.get('global_step', 0)
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        
        self.logger.info(f"📂 Checkpoint loaded: {checkpoint_path}")
        self.logger.info(f"   Resumed from epoch {checkpoint.get('epoch', 50)}, step {self.global_step}")
        
        return {
            'epoch': checkpoint.get('epoch', 50),
            'batch_idx': checkpoint.get('batch_idx', 0),
            'stage': checkpoint.get('stage', 'unknown')
        }
    
    def auto_resume(self, stage: str) -> Optional[Dict]:
        """
        Automatically find and load the latest checkpoint for a given stage
        
        Returns:
            Dict with {'epoch', 'batch_idx', 'stage'} if resumed, None otherwise
        """
        # Look for checkpoints matching this stage
        checkpoint_pattern = f"{stage}_*.pt"
        checkpoints = list(self.config.CHECKPOINT_DIR.glob(checkpoint_pattern))
        
        if not checkpoints:
            self.logger.info(f"No checkpoints found for {stage}, starting fresh")
            return None
        
        # Find latest checkpoint by modification time
        latest_checkpoint = max(checkpoints, key=lambda p: p.stat().st_mtime)
        
        self.logger.info(f"🔄 Resuming from: {latest_checkpoint.name}")
        resume_info = self.load_checkpoint(latest_checkpoint.name)
        
        return resume_info


def post_training_clustering(trainer: SCMTrainer, logger: logging.Logger):
    """
    Post-training: Cluster compressed concepts per domain and update prototypes
    """
    logger.info("\n" + "="*80)
    logger.info("POST-TRAINING: CLUSTERING DOMAIN CONCEPTS")
    logger.info("="*80)
    
    # Collect concepts from validation data
    trainer.model.eval()
    
    # Dictionary to store concepts per domain
    domain_concepts = {domain: [] for domain in trainer.config.DOMAIN_VOCAB_SIZES.keys()}
    
    # Load Stage 2 dataset (validation split)
    dataset = Stage2Dataset(
        trainer.config.STAGE2_JSONL,
        max_samples=5000,  # Use subset for clustering
        logger=logger
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=collate_stage2,
        num_workers=0
    )
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Collecting concepts"):
            embeddings = batch['embeddings'].to(trainer.device)
            attention_mask = batch['attention_mask'].to(trainer.device)
            
            # Forward pass
            outputs = trainer.model(embeddings, attention_mask=attention_mask)
            compressed = outputs['compressed_concepts']  # [B, L, 384]
            
            # Group by domain
            for b, domain in enumerate(batch['domains']):
                valid_mask = attention_mask[b] > 0
                valid_concepts = compressed[b, valid_mask]  # [valid_len, 384]
                
                if domain in domain_concepts:
                    domain_concepts[domain].append(valid_concepts.cpu())
    
    # Cluster concepts per domain
    for domain, concept_list in domain_concepts.items():
        if len(concept_list) == 0:
            logger.warning(f"No concepts collected for domain: {domain}")
            continue
        
        # Concatenate all concepts
        all_concepts = torch.cat(concept_list, dim=0)  # [N, 384]
        logger.info(f"Domain '{domain}': {all_concepts.shape[0]} concepts collected")
        
        # Run k-means clustering
        cluster_centers = trainer.model.domain_vocab.cluster_concepts(
            all_concepts,
            domain,
            n_clusters=None  # Use default vocab size
        )
        
        logger.info(f"  Clustered into {cluster_centers.shape[0]} prototypes")
    
    logger.info("✅ Concept clustering complete")
    
    # Save updated model with clustered prototypes
    trainer.save_checkpoint("final_with_clusters.pt")
    
def build_decoder_corpus(trainer: SCMTrainer, logger: logging.Logger):
    """
    Build SONAR decoder corpus from training data
    
    Requires: sentence_transformers library with SONAR encoder
    """
    logger.info("\n" + "="*80)
    logger.info("BUILDING SONAR DECODER CORPUS")
    logger.info("="*80)
    
    # try:
    #     from sonar.inference_pipelines import SpeechToEmbeddingModelPipeline
    #     sonar_encoder = SpeechToEmbeddingModelPipeline(
    #         encoder="sonar_text_encoder",
    #         device=str(trainer.device)
    #     )
    #     logger.info("✅ SONAR encoder loaded")
    # except ImportError:
    #     logger.warning("⚠️ sonar-expressive not installed.")
    #     logger.warning("Install with: pip install sonar-expressive")
    #     return None
    
    try:
        from sonar.inference_pipelines import TextToEmbeddingModelPipeline  # ✅ Text pipeline
        sonar_encoder = TextToEmbeddingModelPipeline(
            encoder="text_sonar_basic_encoder",  # ✅ Correct encoder name
            tokenizer="text_sonar_basic_encoder",
            device=trainer.device
        )
        logger.info("✅ SONAR encoder loaded")
    except Exception:
        logger.warning("⚠️ sonar-space not installed.")
        logger.warning("Install with: pip install sonar-space")
        return None
    
    # Collect all unique sentences from training data
    text_corpus = set()
    
    # Load Stage 1 data (for sentence collection)
    logger.info("Collecting sentences from Stage 1 dataset...")
    with open(trainer.config.STAGE1_JSONL, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Stage 1 sentences"):
            try:
                sample = json.loads(line.strip())
                if 'text' in sample:
                    text_corpus.add(sample['text'])
            except:
                continue
    
    # Load Stage 2 data
    logger.info("Collecting sentences from Stage 2 dataset...")
    with open(trainer.config.STAGE2_JSONL, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Stage 2 sentences"):
            try:
                sample = json.loads(line.strip())
                for step in sample.get('reasoning_chain', []):
                    if 'thinking_step' in step:
                        text_corpus.add(step['thinking_step'])
            except:
                continue
    
    text_corpus = list(text_corpus)
    logger.info(f"Collected {len(text_corpus)} unique sentences")
    
    # Create decoder
    decoder = SONARDecoder(trainer.model.compressor)
    
    # Build corpus
    decoder.build_corpus(sonar_encoder, text_corpus, batch_size=32)
    
    # Save decoder
    decoder_path = trainer.config.CHECKPOINT_DIR / "sonar_decoder.pt"
    torch.save({
        'corpus_embeddings': decoder.corpus_embeddings,
        'corpus_texts': decoder.corpus_texts
    }, decoder_path)
    
    logger.info(f"✅ SONAR decoder corpus saved: {decoder_path}")
    return decoder

# ═══════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════

def main():
    """Complete SCM training pipeline with all advanced features"""
    # Initialize config
    config = SCMConfig()
    config.validate()
    
    # Setup logging
    logger = setup_logging(config)
    
    # Initialize trainer
    trainer = SCMTrainer(config, logger)
    
    trainer.enable_swa(swa_start_epoch=2)  # Start SWA from epoch 2

    # Then training
    logger.info("\n🚀 Stage 1: Denoising Pre-training")
    # trainer.train_stage1(num_epochs=50, max_samples=12000)
    
    # Calculate total steps across all stages
    stage1_steps = (12000 / config.BATCH_SIZE) * 3 / config.ACCUMULATION_STEPS
    stage2_steps = (6000 / config.BATCH_SIZE) * 5 / config.ACCUMULATION_STEPS
    stage3_steps = (3000 / config.BATCH_SIZE) * 3 / config.ACCUMULATION_STEPS

    total_steps = stage1_steps + stage2_steps + stage3_steps
    trainer.set_warmup_steps(int(total_steps))

    logger.info(f"Total training steps: {int(total_steps)}, Warmup: {trainer.warmup_steps}")

    if config.RUN_LR_FINDER:  # Add this flag to SCMConfig
        full_dataset = Stage1Dataset(config.STAGE1_JSONL, max_samples=1000)
        optimal_lr, lrs, losses = trainer.find_optimal_lr(full_dataset)
        trainer.config.LEARNING_RATE = optimal_lr
        logger.info(f"✅ Optimal LR found: {optimal_lr:.2e}")

    
    # Training pipeline
    try:
        # Stage 1: Denoising Pre-training
        # logger.info("\n🚀 Stage 1: Denoising Pre-training")
        # trainer.train_stage1(num_epochs=50, max_samples=12000)
        
        # # Stage 2: Supervised Fine-Tuning
        # logger.info("\n🚀 Stage 2: Supervised Fine-Tuning")
        # trainer.train_stage2(num_epochs=40, max_samples=6000)
        
        # # ✅ Stage 3: RLAIF Training
        # logger.info("\n🚀 Stage 3: RLAIF Training")
        # trainer.train_stage3(num_epochs=5, max_samples=3000)
        
        post_training_clustering(trainer, logger)
        
        
        # ✅ Build SONAR decoder corpus (post-training)
        logger.info("\n🔧 Building SONAR decoder corpus...")
        # TODO: Load your SONAR encoder here
        # sonar_encoder = load_sonar_encoder()
        # text_corpus = load_training_corpus()
        decoder = build_decoder_corpus(trainer, logger)
        # decoder = SONARDecoder(trainer.model.compressor)
        # decoder.build_corpus(sonar_encoder, text_corpus)
        
        logger.info("\n" + "="*80)
        logger.info("✅ SCM TRAINING COMPLETE")
        logger.info("="*80)
        logger.info(f"Best Loss: {trainer.best_loss:.4f}")
        logger.info(f"Total Steps: {trainer.global_step}")
        logger.info(f"Checkpoints: {config.CHECKPOINT_DIR}")
        
    except KeyboardInterrupt:
        logger.warning("\n⚠️ Training interrupted")
        trainer.save_checkpoint("interrupted.pt")
    except Exception as e:
        logger.error(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()
