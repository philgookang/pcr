# ##################################
# DATASET
# ##################################

# term frequency threshold
# minimun count required
term_frequency_threshold = 5

# size of dataset
validation_and_test_dataset_size = 5000

# dataset type
# 1 - MSCOCO 2017
# 2 - Flickr8k
# 3 - Flickr30k
dataset_type = 1


# ##################################
# PRETRAIN & TRAINING & TEST
# ##################################

# number of epochs
cnn_rnn_number_epochs = 1000

# image random crop size
image_crop_size = 224

# number of epochs
pretrain_number_epochs = 9

# log step size
log_step = 5


# ##################################
# DATA LOADER
# ##################################

# number of process worker (load data from disk to gpu)
number_of_workers =  32
# number_of_workers =  0 # for bidirectional

# data loader shuffle dataset
is_shuffle = False

# size of each training cycle
pretrain_batch_size = 128

# size of each training cycle
train_batch_size = 192
# train_batch_size = 16 # for bidirectional
