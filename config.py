""" A neural chatbot using sequence to sequence model with
attentional decoder.

This is based on Google Translate Tensorflow model
https://github.com/tensorflow/models/blob/master/tutorials/rnn/translate/

Sequence to sequence model by Cho et al.(2014)

Created by Chip Huyen (chiphuyen@cs.stanford.edu)
CS20: "TensorFlow for Deep Learning Research"
cs20.stanford.edu

This file contains the hyperparameters for the model.

See README.md for instruction on how to run the starter code.
"""

# parameters for processing the dataset
DATA_PATH = 'cornell_movie_dialogs_corpus/cornell movie-dialogs corpus'
CONVO_FILE = 'movie_conversations.txt'
LINE_FILE = 'movie_lines.txt'
OUTPUT_FILE = 'output_wannta.txt'
PROCESSED_PATH = ''
CPT_PATH = '/data/chatbot/checkpoints'

THRESHOLD = 2

PAD_ID = 0
UNK_ID = 3
START_ID = 10000
EOS_ID = 10000

TESTSET_SIZE = 25000

# BUCKETS = [(19, 19), (28, 28), (33, 33), (40, 43), (50, 53), (60, 63)]
BUCKETS = [(3, 3), (7, 7), (15, 15), (20, 23), (40, 43), (50, 53)]
#BUCKETS = [(19, 19), (28, 28)]

CONTRACTIONS = [("i ' m ", "i 'm "), ("' d ", "'d "), ("' s ", "'s "),
				("don ' t ", "do n't "), ("didn ' t ", "did n't "), ("doesn ' t ", "does n't "),
				("can ' t ", "ca n't "), ("shouldn ' t ", "should n't "), ("wouldn ' t ", "would n't "),
				("' ve ", "'ve "), ("' re ", "'re "), ("in ' ", "in' ")]

NUM_LAYERS = 3
HIDDEN_SIZE = 256
BATCH_SIZE = 32

LR = 0.5
MAX_GRAD_NORM = 5.0

NUM_SAMPLES = 256
ENC_VOCAB = 8603
DEC_VOCAB = 8603
