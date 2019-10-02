# ##################################
# Convolutional Neural Network
# ##################################

# fully connected layer output size
cnn_output_size = 1024

# ##################################
# Recurrent Neural Network
# ##################################

# Using bi-direction RNN
use_bi_direction_lstm = False

# LSTM input size
rnn_embed_size = 1024 # 5120

# dimension of lstm hidden states
rnn_lstm_hidden_size = 512

# number of layers in lstm
rnn_lstm_number_of_layers = 3

# generation sentence method
rnn_inference = "sample" # "sample" "beam_search"

# number of beam search K
rnn_beam_search_width = 3

# dropout
rnn_lstm_dropout = 0
