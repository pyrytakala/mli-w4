# Image processing constants
IMAGE_CHANNELS = 3
IMAGE_SIZE = 224
IMAGE_MEAN = [0.48145466, 0.4578275, 0.40821073]
IMAGE_STD = [0.26862954, 0.26130258, 0.27577711]

# Model architecture constants
CLIP_HIDDEN_SIZE = 768  # hidden size of CLIP image model (from CLIP)
EMBEDDING_SIZE = 512  # word embedding (from CLIP)
NUM_HEADS = 8
FEEDFORWARD_DIM = 512
NUM_DECODER_LAYERS = 6

# Training constants
DEFAULT_BATCH_SIZE = 32
DEFAULT_TRAIN_SPLIT = 0.8
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_NUM_EPOCHS = 20
MAX_SEQUENCE_LENGTH = 77  # Total sequence length including image patches and caption tokens
VALIDATION_FREQUENCY = 1000  # Validate and log examples every 100 minibatches

# CLIP model name
CLIP_MODEL_NAME = 'openai/clip-vit-base-patch32' 

# Tokenizer constants
START_TOKEN = "<|startoftext|>"
END_TOKEN = "<|endoftext|>"
PAD_TOKEN = "<|pad|>"

# Random seed for reproducibility
RANDOM_SEED = 42

# Generation parameters
GENERATION_TEMPERATURE = 1.0