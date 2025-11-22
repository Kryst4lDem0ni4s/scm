#!/usr/bin/env python3
"""
SCM Model Inference Test Suite
Comprehensive testing for trained SCM model with SONAR encoder and decoder
"""

import subprocess
import sys
import torch
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import numpy as np
import traceback
from datetime import datetime
import json
# Import model classes
from scm_main import SCMConfig, SCMTransformer


# ═══════════════════════════════════════════════════════════════════
# ENVIRONMENT SETUP
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
    
    print("⚠️  Installing PyTorch with CUDA 12.8 support...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "--upgrade",
        "torch", "torchvision", "torchaudio",
        "--extra-index-url", "https://download.pytorch.org/whl/cu128"
    ])
    
    import torch
    assert torch.cuda.is_available(), "❌ CUDA installation failed"
    print(f"✅ PyTorch {torch.__version__} installed with CUDA {torch.version.cuda}")
    return True


# Run setup
setup_cuda()
torch.set_default_device('cuda')


# ═══════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

class InferenceConfig:
    """Paths configuration for inference"""
    
    # Model checkpoint (auto-detect best available)
    CHECKPOINT_DIR = Path("G:/My Drive/scm_project/checkpoints/scm_training")
    
    # Decoder corpus
    DECODER_PATH = Path("G:/My Drive/scm_project/checkpoints/scm_training/sonar_decoder.pt")
    
    # Device
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# ═══════════════════════════════════════════════════════════════════
# INFERENCE ENGINE
# ═══════════════════════════════════════════════════════════════════

class SCMInferenceTester:
    """Complete inference testing for SCM"""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        decoder_path: Optional[str] = None,
        device: str = 'cuda'
    ):
        """
        Initialize inference tester
        
        Args:
            model_path: Path to model checkpoint (auto-detected if None)
            decoder_path: Path to decoder corpus
            device: 'cuda' or 'cpu'
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"\n{'='*80}")
        print("🚀 SCM INFERENCE TESTER")
        print(f"{'='*80}")
        print(f"🔧 Using device: {self.device}")
        
        # Auto-detect model if not provided
        if model_path is None:
            model_path = self._find_best_checkpoint()
        
        # Load model
        self._load_model(model_path)
        
        # Load decoder
        self._load_decoder(decoder_path)
        
        # Initialize SONAR encoder
        self._init_sonar_encoder()
    
    def _find_best_checkpoint(self) -> str:
        """Auto-detect best checkpoint"""
        config = InferenceConfig()
        priority_checkpoints = [
            "stage3_best.pt",
            "stage2_best.pt",
            "final_with_clusters.pt",
            "stage1_best.pt"
        ]
        
        for ckpt_name in priority_checkpoints:
            ckpt_path = config.CHECKPOINT_DIR / ckpt_name
            if ckpt_path.exists():
                print(f"✅ Found checkpoint: {ckpt_name}")
                return str(ckpt_path)
        
        # Fallback: any .pt file
        pt_files = list(config.CHECKPOINT_DIR.glob("*.pt"))
        if pt_files:
            ckpt = sorted(pt_files)[0]
            print(f"⚠️  Using fallback: {ckpt.name}")
            return str(ckpt)
        
        raise FileNotFoundError(f"No checkpoints found in {config.CHECKPOINT_DIR}")
    
    def _load_model(self, model_path: str):
        """Load trained SCM model"""
        print(f"\n📂 Loading model from: {model_path}")
        
        # Import model classes
        from scm_main import SCMConfig, SCMTransformer
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # Reconstruct config
        config = SCMConfig()
        
        # ✅ FIX: Handle both dict and SCMConfig object
        if 'config' in checkpoint:
            checkpoint_config = checkpoint['config']
            
            # Case 1: Config is a dictionary
            if isinstance(checkpoint_config, dict):
                for key, value in checkpoint_config.items():
                    if hasattr(config, key):
                        setattr(config, key, value)
            
            # Case 2: Config is an SCMConfig object
            elif hasattr(checkpoint_config, '__dict__'):
                for key, value in checkpoint_config.__dict__.items():
                    if hasattr(config, key) and not key.startswith('_'):
                        setattr(config, key, value)
            
            # Case 3: Config is already an SCMConfig instance (use directly)
            elif isinstance(checkpoint_config, SCMConfig):
                config = checkpoint_config
            
            else:
                print(f"⚠️  Unknown config type: {type(checkpoint_config)}, using defaults")
        
        # Initialize model
        self.model = SCMTransformer(config).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        self.config = config
        self.compartment_to_idx = checkpoint.get('compartment_to_idx', {})
        self.hierarchy_to_idx = checkpoint.get('hierarchy_to_idx', {})
        
        # Reverse mappings
        self.idx_to_compartment = {v: k for k, v in self.compartment_to_idx.items()}
        self.idx_to_hierarchy = {v: k for k, v in self.hierarchy_to_idx.items()}
        
        param_count = sum(p.numel() for p in self.model.parameters()) / 1e6
        print(f"✅ Model loaded: {param_count:.1f}M parameters")

    
    def _load_decoder(self, decoder_path: Optional[str]):
        """Load SONAR decoder corpus (Windows + Google Drive fix)"""
        import io
        
        if decoder_path is None:
            decoder_path = InferenceConfig().DECODER_PATH
        
        decoder_path = Path(decoder_path)
        
        if not decoder_path.exists():
            print(f"⚠️  Decoder corpus not found: {decoder_path}")
            print("   Text generation will be unavailable")
            self.corpus_embeddings = None
            self.corpus_texts = None
            self.decoder_available = False
            return
        
        print(f"\n📂 Loading decoder corpus from: {decoder_path}")
        
        try:
            # ✅ FIX: Read entire file into memory buffer first
            print("   Reading file into memory...")
            with open(decoder_path, 'rb') as f:
                file_buffer = io.BytesIO(f.read())
            
            print("   Deserializing PyTorch checkpoint...")
            decoder_data = torch.load(
                file_buffer,
                map_location=self.device,
                weights_only=False
            )
            
            self.corpus_embeddings = decoder_data['corpus_embeddings'].to(self.device)
            self.corpus_texts = decoder_data['corpus_texts']
            
            print(f"✅ Decoder loaded: {len(self.corpus_texts)} sentences")
            self.decoder_available = True
            
        except Exception as e:
            traceback.print_exc()
            print(f"⚠️  Failed to load decoder: {e}")
            print(f"   File path: {decoder_path}")
            print("   Text generation will be unavailable")
            self.corpus_embeddings = None
            self.corpus_texts = None
            self.decoder_available = False
    
    def _init_sonar_encoder(self):
        """Initialize SONAR text encoder"""
        try:
            from sonar.inference_pipelines import TextToEmbeddingModelPipeline
            
            print("\n📥 Loading SONAR encoder...")
            self.encoder = TextToEmbeddingModelPipeline(
                encoder="text_sonar_basic_encoder",
                tokenizer="text_sonar_basic_encoder",
                device=self.device
            )
            print("✅ SONAR encoder initialized")
            self.encoder_available = True
            
        except Exception as e:
            print(f"⚠️  SONAR encoder unavailable: {e}")
            print("   Using dummy embeddings for testing")
            self.encoder = None
            self.encoder_available = False

    def encode_text(self, texts: List[str]) -> torch.Tensor:
        """Convert text to SONAR embeddings"""
        if self.encoder_available:
            embeddings = self.encoder.predict(texts, source_lang='eng_Latn')
            
            # Handle both Tensor and NumPy returns
            if isinstance(embeddings, torch.Tensor):
                return embeddings.to(self.device)
            elif isinstance(embeddings, np.ndarray):
                return torch.from_numpy(embeddings).float().to(self.device)
            else:
                raise TypeError(f"Unexpected embedding type: {type(embeddings)}")
        else:
            # Dummy embeddings for testing
            return torch.randn(len(texts), 1024, device=self.device)
    
    def decode_to_text(
        self,
        sonar_embeddings: torch.Tensor,
        k: int = 3
    ) -> List[Tuple[List[str], List[float]]]:
        """
        Decode SONAR embeddings to text using kNN retrieval
        
        Returns:
            List of (texts, similarities) for each position
        """
        if not self.decoder_available:
            return [([f"<no decoder - position {i}>"], [0.0]) for i in range(sonar_embeddings.shape[0])]
        
        # Normalize
        query_norm = F.normalize(sonar_embeddings, dim=-1)  # [L, 1024]
        corpus_norm = F.normalize(self.corpus_embeddings, dim=-1)  # [N, 1024]
        
        # Compute similarities
        similarities = torch.matmul(query_norm, corpus_norm.t())  # [L, N]
        
        # Get top-k
        topk_sims, topk_indices = similarities.topk(k, dim=-1)  # [L, k]
        
        results = []
        for seq_idx in range(topk_indices.shape[0]):
            step_texts = []
            step_sims = []
            
            for k_idx in range(k):
                corpus_idx = topk_indices[seq_idx, k_idx].item()
                sim_score = topk_sims[seq_idx, k_idx].item()
                text = self.corpus_texts[corpus_idx]
                
                step_texts.append(text)
                step_sims.append(sim_score)
            
            results.append((step_texts, step_sims))
        
        return results
    
    def analyze_concept(self, text: str) -> Dict:
        """Perform comprehensive concept analysis"""
        # Encode text
        sonar_emb = self.encode_text([text]).unsqueeze(1)  # [1, 1, 1024]
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(sonar_emb, use_causal_mask=False)
        
        # Extract predictions
        comp_logits = outputs['compartment_logits'][0, 0]
        hier_logits = outputs['hierarchy_logits'][0, 0]
        compressed = outputs['compressed_concepts'][0, 0]
        
        comp_probs = F.softmax(comp_logits, dim=-1)
        hier_probs = F.softmax(hier_logits, dim=-1)
        
        comp_idx = comp_logits.argmax().item()
        hier_idx = hier_logits.argmax().item()
        
        comp_pred = self.idx_to_compartment.get(comp_idx, 'UNKNOWN')
        hier_pred = self.idx_to_hierarchy.get(hier_idx, 'UNKNOWN')
        
        # Domain alignment
        domain_scores = {}
        for domain in self.model.domain_vocab.domains:
            prototypes = self.model.domain_vocab.get_prototypes(domain)
            sims = F.cosine_similarity(
                compressed.unsqueeze(0),
                prototypes,
                dim=-1
            )
            domain_scores[domain] = sims.max().item()
        
        return {
            'text': text,
            'compartment': comp_pred,
            'compartment_confidence': comp_probs[comp_idx].item(),
            'hierarchy': hier_pred,
            'hierarchy_confidence': hier_probs[hier_idx].item(),
            'domain_alignment': domain_scores,
            'all_compartment_probs': {
                self.idx_to_compartment.get(i, f'UNK_{i}'): comp_probs[i].item()
                for i in range(len(comp_probs))
            }
        }
    
    def generate_reasoning_chain(
        self,
        prompt: str,
        max_steps: int = 5,
        temperature: float = 0.7,
        top_k: int = 3
    ) -> List[Dict]:
        """Generate autoregressive reasoning chain"""
        print(f"\n🧠 Generating reasoning chain...")
        print(f"   Prompt: '{prompt}'")
        print(f"   Max steps: {max_steps}, Temperature: {temperature}")
        
        # Encode prompt
        prompt_sonar = self.encode_text([prompt]).unsqueeze(1)  # [1, 1, 1024]
        
        # Generate concepts
        with torch.no_grad():
            generated_compressed = self.model.generate(
                initial_concepts=prompt_sonar,
                max_length=max_steps,
                temperature=temperature
            )  # [1, max_steps, 384]
        
        # Expand to SONAR space
        expanded_sonar = self.model.compressor.expand_concept(
            generated_compressed
        )  # [1, max_steps, 1024]
        
        # Decode to text
        reasoning_steps = []
        decoded_results = self.decode_to_text(expanded_sonar[0], k=top_k)
        
        for step_idx, (texts, sims) in enumerate(decoded_results):
            step_info = {
                'step': step_idx + 1,
                'top_k_retrievals': [
                    {'text': t, 'similarity': float(s)}
                    for t, s in zip(texts, sims)
                ],
                'best_text': texts[0],
                'best_similarity': float(sims[0])
            }
            reasoning_steps.append(step_info)
        
        return reasoning_steps
    
    def print_analysis(self, result: Dict):
        """Pretty print analysis results"""
        print(f"\n📊 CONCEPT ANALYSIS")
        print(f"   Text: {result['text'][:70]}...")
        print(f"\n   🏷️  Compartment: {result['compartment']} "
              f"(conf: {result['compartment_confidence']:.3f})")
        print(f"   📏 Hierarchy: {result['hierarchy']} "
              f"(conf: {result['hierarchy_confidence']:.3f})")
        
        print(f"\n   🎯 Domain Alignment:")
        sorted_domains = sorted(
            result['domain_alignment'].items(),
            key=lambda x: x[1],
            reverse=True
        )
        for domain, score in sorted_domains:
            bar = '█' * int(score * 20)
            print(f"      {domain:15s}: {bar} {score:.3f}")
    
    def print_reasoning(self, reasoning: List[Dict]):
        """Pretty print reasoning chain"""
        for step in reasoning:
            print(f"\n📍 Step {step['step']}:")
            for rank, retr in enumerate(step['top_k_retrievals']):
                marker = "🥇" if rank == 0 else "🥈" if rank == 1 else "🥉"
                print(f"   {marker} [{retr['similarity']:.4f}] {retr['text'][:75]}...")


# ═══════════════════════════════════════════════════════════════════
# TEST SUITE
# ═══════════════════════════════════════════════════════════════════

def run_test_suite():
    """Run comprehensive inference tests"""
    
    # Initialize tester
    tester = SCMInferenceTester()
    
    # TEST 1: Concept Analysis
    print(f"\n{'='*80}")
    print("TEST 1: CONCEPT ANALYSIS")
    print(f"{'='*80}")
    
    test_texts = [
        "Photosynthesis is the process by which plants convert light energy into chemical energy.",
        "The Eiffel Tower is located in Paris, France.",
        "To solve this equation, first isolate the variable on one side."
    ]
    
    for text in test_texts:
        analysis = tester.analyze_concept(text)
        tester.print_analysis(analysis)
    
    # TEST 2: Reasoning Generation
    print(f"\n{'='*80}")
    print("TEST 2: REASONING CHAIN GENERATION")
    print(f"{'='*80}")
    
    prompts = [
        "How does a neural network learn?",
        "What causes photosynthesis?"
    ]
    
    for prompt in prompts:
        reasoning = tester.generate_reasoning_chain(
            prompt=prompt,
            max_steps=5,
            temperature=0.7,
            top_k=3
        )
        tester.print_reasoning(reasoning)
    
    # TEST 3: Domain Specificity
    print(f"\n{'='*80}")
    print("TEST 3: DOMAIN SPECIFICITY")
    print(f"{'='*80}")
    
    domain_tests = {
        'Science': "The mitochondria is the powerhouse of the cell.",
        'Medical': "Administer 500mg of ibuprofen every 6 hours.",
        'General': "I enjoy reading books in my free time."
    }
    
    for domain, text in domain_tests.items():
        print(f"\n📂 Expected Domain: {domain}")
        result = tester.analyze_concept(text)
        top_domain = max(result['domain_alignment'].items(), key=lambda x: x[1])
        print(f"   Predicted: {result['compartment']} / {result['hierarchy']}")
        print(f"   Top domain: {top_domain[0]} ({top_domain[1]:.3f})")
    
    print(f"\n{'='*80}")
    print("✅ ALL TESTS COMPLETE")
    print(f"{'='*80}")


# ═══════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    try:
        run_test_suite()
    except Exception as e:
        print(f"\n❌ Error during inference: {e}")
        import traceback
        traceback.print_exc()
