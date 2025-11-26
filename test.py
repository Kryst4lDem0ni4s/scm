import logging
import os
import sonar

os.environ["FAIRSEQ2_EXTENSION_TRACE"] = "True"

try:
    # Run this in WSL (Linux shell) ONLY!
    from sonar.inference_pipelines import TextToEmbeddingModelPipeline
    self_encoder = TextToEmbeddingModelPipeline(
        encoder="text_sonar_basic_encoder",
        tokenizer="text_sonar_basic_encoder"
    )
    self_encoder_type = "sonar"
    print("SONAR encoder loaded (1024-dim)")
except (ImportError, Exception) as e:
    print(f"SONAR unavailable: {e}")

    # On Windows fallback: LaBSE
    try:
        from sentence_transformers import SentenceTransformer
        selfencoder = SentenceTransformer('sentence-transformers/LaBSE')
        self_encoder_type = "labse"
        logging.info("LaBSE encoder loaded (1024-dim)")
    except Exception as labse_e:
        logging.error(f"LaBSE unavailable: {labse_e}")
