TRAIN_FILES = "/w266/project/mscoco/image_vectors_and_captions/train-?????-of-00256"
VAL_FILES = "/w266/project/mscoco/image_vectors_and_captions/val-?????-of-00004"
TEST_FILES = "/w266/project/mscoco/image_vectors_and_captions/test-?????-of-00008"

CHECKPOINT_DIR = "/w266/project/mscoco/rnn_model_checkpoints"
MAX_CHECKPOINTS_TO_KEEP=5

EVAL_DIR="/w266/project/mscoco/model_eval"
EVAL_INTERVAL_SECS=600 # "Interval between evaluation runs."
NUM_EVAL_EXAMPLES=10132
MIN_GLOBAL_STEP=5000 # "Minimum global step to run evaluation."

NUM_TRAIN_EXAMPLES = 586363

IMAGE_VECTOR_SIZE = 2048
HIDDEN_UNITS = 512
VOCAB_SIZE = 12000
BATCH_SIZE = 32
LOG_EVERY_N_STEPS = 1
NUMBER_OF_STEPS = 1000000

#VOCAB_FILE="/home/subhashini.r/mscoco/annotations/word_counts.txt"
VOCAB_FILE="/w266/project/mscoco/annotations/word_counts.txt"

INFERENCE_FILES=""
#INFERENCE_IMAGE_FILES_DIR="/home/subhashini.r/mscoco/images/validation/label"
INFERENCE_IMAGE_FILES_DIR="/w266/project/mscoco/images/validation/label"
#INFERENCE_VECTOR_FILES_DIR="/home/subhashini.r/mscoco/inception_image_vectors/validation"
INFERENCE_VECTOR_FILES_DIR="/w266/project/mscoco/inception_image_vectors/validation"
