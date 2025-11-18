#!/usr/bin/env python3
"""
SCM Inference Pipeline
Load trained model and run inference on custom inputs
"""

import torch
import torch.nn.functional as F
from typing import List, Dict, Optional
from pathlib import Path
import numpy as np

class SCMInference:
    """Production inference wrapper for SCM"""
    
    def __init__(
        self,
        model_path: str,
        decoder_path: Optional[str] = None,
        device: str = 'cuda'
    ):
        """
        Args:
            model_path: Path to exported .pt model
            decoder_path: Path to SONAR decoder corpus (optional)
            device: 'cuda' or 'cpu'
        """
        self.device = torch.device(device)
        
        # Load model
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Reconstruct model from config
        from scm_main import SCMConfig, SCMTransformer
        
        config = SCMConfig()
        # Override with saved config
        for key, value in checkpoint['config'].items():
            setattr(config, key, value)
        
        self.model = SCMTransformer(config).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        self.config = config
        self.compartment_to_idx = checkpoint['compartment_to_idx']
        self.hierarchy_to_idx = checkpoint['hierarchy_to_idx']
        
        # Load decoder if available
        if decoder_path and Path(decoder_path).exists():
            decoder_data = torch.load(decoder_path, map_location=self.device)
            self.corpus_embeddings = decoder_data['corpus_embeddings']
            self.corpus_texts = decoder_data['corpus_texts']
            print(f"✅ Loaded decoder with {len(self.corpus_texts)} sentences")
        else:
            self.corpus_embeddings = None
            self.corpus_texts = None
            print("⚠️ No decoder corpus loaded (kNN retrieval unavailable)")
        
        print(f"✅ SCM model loaded ({sum(p.numel() for p in self.model.parameters())/1e6:.1f}M params)")
    
    def encode_text_to_sonar(self, texts: List[str]) -> torch.Tensor:
        """
        Convert text to SONAR embeddings (requires SONAR encoder)
        
        For production, you need to integrate your SONAR encoder here.
        """
        try:
            from sonar.inference_pipelines import TextToEmbeddingModelPipeline
            
            encoder = TextToEmbeddingModelPipeline(
                encoder="text_sonar_basic_encoder",
                tokenizer="text_sonar_basic_encoder",
                device=self.device
            )
            
            embeddings = encoder.predict(texts, source_lang='eng_Latn')
            return torch.from_numpy(embeddings).float().to(self.device)
        
        except Exception as e:
            print(f"⚠️ SONAR encoder not available: {e}")
            print("Using random embeddings for demo (replace with real SONAR)")
            # DEMO ONLY: Random embeddings
            return torch.randn(len(texts), 1024).to(self.device)
    
    def compress_concepts(self, sonar_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compress SONAR embeddings to concept space
        
        Args:
            sonar_embeddings: [batch, seq_len, 1024]
        
        Returns:
            compressed: [batch, seq_len, 384]
        """
        with torch.no_grad():
            return self.model.compressor.compress_concept(sonar_embeddings)
    
    def generate_reasoning(
        self,
        initial_text: str,
        max_steps: int = 10,
        temperature: float = 0.7,
        domain: str = 'general'
    ) -> List[str]:
        """
        Generate autoregressive reasoning chain
        
        Args:
            initial_text: Starting prompt
            max_steps: Max reasoning steps to generate
            temperature: Sampling temperature (0 = greedy, >0 = stochastic)
            domain: Domain context ('general', 'science', 'medical', etc.)
        
        Returns:
            List of generated reasoning steps (as text if decoder available)
        """
        # Encode initial text
        initial_sonar = self.encode_text_to_sonar([initial_text])  # [1, 1024]
        initial_sonar = initial_sonar.unsqueeze(1)  # [1, 1, 1024]
        
        # Generate concepts
        generated_concepts = self.model.generate(
            initial_concepts=initial_sonar,
            max_length=max_steps,
            temperature=temperature
        )
        
        # Decode back to text (if decoder available)
        if self.corpus_embeddings is not None:
            expanded = self.model.compressor.expand_concept(generated_concepts)
            reasoning_steps = self.decode_to_text(expanded[0])  # [seq_len, 1024]
            return reasoning_steps
        else:
            # Return compressed concepts (numerical)
            return [f"Concept_{i}: {c.cpu().numpy()[:5]}..." 
                    for i, c in enumerate(generated_concepts[0])]
    
    def decode_to_text(self, sonar_embeddings: torch.Tensor, k: int = 1) -> List[str]:
        """
        Decode SONAR embeddings to text using kNN retrieval
        
        Args:
            sonar_embeddings: [seq_len, 1024]
            k: Number of nearest neighbors
        
        Returns:
            List of retrieved sentences
        """
        if self.corpus_embeddings is None:
            raise ValueError("No decoder corpus loaded")
        
        # Normalize embeddings
        query_norm = F.normalize(sonar_embeddings, dim=-1)  # [L, 1024]
        corpus_norm = F.normalize(self.corpus_embeddings, dim=-1)  # [N, 1024]
        
        # Compute similarities
        similarities = torch.matmul(query_norm, corpus_norm.t())  # [L, N]
        
        # Get top-k indices
        _, topk_indices = similarities.topk(k, dim=-1)  # [L, k]
        
        # Retrieve text
        results = []
        for idx in topk_indices:
            best_idx = idx[0].item()
            results.append(self.corpus_texts[best_idx])
        
        return results
    
    def analyze_concepts(self, text: str) -> Dict:
        """
        Analyze text and return concept-level insights
        
        Returns:
            Dict with:
                - compressed_concepts: [seq_len, 384]
                - compartment_predictions: List[str]
                - hierarchy_predictions: List[str]
                - domain_alignment: Dict[str, float]
        """
        # Encode
        sonar_emb = self.encode_text_to_sonar([text]).unsqueeze(1)  # [1, 1, 1024]
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(sonar_emb, use_causal_mask=False)
        
        # Get predictions
        comp_logits = outputs['compartment_logits'][0, 0]  # [5]
        hier_logits = outputs['hierarchy_logits'][0, 0]  # [3]
        
        comp_probs = F.softmax(comp_logits, dim=-1)
        hier_probs = F.softmax(hier_logits, dim=-1)
        
        # Map to labels
        idx_to_comp = {v: k for k, v in self.compartment_to_idx.items()}
        idx_to_hier = {v: k for k, v in self.hierarchy_to_idx.items()}
        
        comp_pred = idx_to_comp[comp_logits.argmax().item()]
        hier_pred = idx_to_hier[hier_logits.argmax().item()]
        
        # Domain alignment
        compressed = outputs['compressed_concepts'][0, 0]  # [384]
        domain_scores = {}
        
        for domain in self.model.domain_vocab.domains:
            prototypes = self.model.domain_vocab.get_prototypes(domain)
            similarities = F.cosine_similarity(
                compressed.unsqueeze(0), 
                prototypes, 
                dim=-1
            )
            domain_scores[domain] = similarities.max().item()
        
        return {
            'compressed_concepts': compressed.cpu().numpy(),
            'compartment': comp_pred,
            'hierarchy': hier_pred,
            'compartment_confidence': comp_probs.max().item(),
            'hierarchy_confidence': hier_probs.max().item(),
            'domain_alignment': domain_scores
        }

# ========== USAGE EXAMPLE ==========
if __name__ == "__main__":
    # Load model
    scm = SCMInference(
        model_path="./scm_production/scm_production.pt",
        decoder_path="./scm_production/sonar_decoder.pt",
        device='cuda'
    )
    
    # Test 1: Analyze single concept
    print("\n" + "="*80)
    print("TEST 1: Concept Analysis")
    print("="*80)
    
    analysis = scm.analyze_concepts("Photosynthesis is the process by which plants convert light energy into chemical energy.")
    
    print(f"Compartment: {analysis['compartment']} (conf: {analysis['compartment_confidence']:.3f})")
    print(f"Hierarchy: {analysis['hierarchy']} (conf: {analysis['hierarchy_confidence']:.3f})")
    print(f"Domain alignment: {analysis['domain_alignment']}")
    
    # Test 2: Generate reasoning chain
    print("\n" + "="*80)
    print("TEST 2: Reasoning Generation")
    print("="*80)
    
    reasoning = scm.generate_reasoning(
        initial_text="How does a computer processor work?",
        max_steps=5,
        temperature=0.5,
        domain='science'
    )
    
    for i, step in enumerate(reasoning):
        print(f"Step {i+1}: {step}")
