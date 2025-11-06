CHECKPOINT_DIR = "C:/Users/Khwaish/Google Drive Streaming/My Drive/scm_project/checkpoints/stage1_processing.pkl"

import pickle
with open('checkpoints/stage1_processing.pkl', 'rb') as f:
    data = pickle.load(f)

# Check doc_data has segments
doc0 = data['doc_data'][0]
print(f"Doc 0 segments: {len(doc0['segments'])}")  # Should be 10-20
print(f"First segment: {doc0['segments'][0]}")  # Should have 'text', 'compartment', 'hierarchy'

doc_data = data['doc_data']
print(f"Loaded {len(doc_data)} documents for Stage 2")
# Should NOT crash with "Cannot proceed: No doc_data available"
