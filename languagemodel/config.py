#BASE_PATH="/home/subhashini.r/"
BASE_PATH="/w266/project/"
TRAIN_FILES = BASE_PATH + "mscoco/image_vectors_and_captions/train-?????-of-00256"
VAL_FILES = BASE_PATH + "mscoco/image_vectors_and_captions/val-?????-of-00004"
TEST_FILES = BASE_PATH + "mscoco/image_vectors_and_captions/test-?????-of-00008"
PROCESSED_TEST_FILE= BASE_PATH + "mscoco/predictions/test_data.json"
PREDICTIONS_FILE= BASE_PATH + "mscoco/predictions/test_data_predictions.json"

#folder to pick up checkpoint for re-training
CHECKPOINT_DIR = BASE_PATH + "mscoco/rnn_model_checkpoint"
#checkpoint file to pick up for evaluation and inference
CHECKPOINT_FILE = BASE_PATH + "mscoco/rnn_model_checkpoints/model.ckpt-171230"
MAX_CHECKPOINTS_TO_KEEP=200

EVAL_DIR= BASE_PATH + "mscoco/rnn_model_eval"
EVAL_INTERVAL_SECS=600 # "Interval between evaluation runs."
NUM_EVAL_EXAMPLES=10132
MIN_GLOBAL_STEP=5000 # "Minimum global step to run evaluation."

NUM_TRAIN_EXAMPLES = 586363

IMAGE_VECTOR_SIZE = 2048
HIDDEN_UNITS = 512
VOCAB_SIZE = 12000
BATCH_SIZE = 32
LOG_EVERY_N_STEPS = 50
NUMBER_OF_STEPS = 1000000

#VOCAB_FILE="/home/subhashini.r/mscoco/annotations/word_counts.txt"
VOCAB_FILE= BASE_PATH + "mscoco/annotations/word_counts.txt"

INFERENCE_IMAGE_FILES_DIR= BASE_PATH + "mscoco/images/validation/label"
INFERENCE_VECTOR_FILES_DIR= BASE_PATH + "mscoco/inception_image_vectors/validation"
