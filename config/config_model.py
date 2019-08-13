# ##################################
# Convolutional Neural Network
# ##################################

# fully connected layer output size
cnn_output_size = 1024

# PoS CNN output combine method
# 1 - concatenation    set rnn output 5120
# 2 - Average matrix   set rnn output 1024
cnn_output_combine_methods = 1


# ##################################
# Recurrent Neural Network
# ##################################

# LSTM input size
rnn_embed_size = 5120 # 1024

# dimension of lstm hidden states
rnn_lstm_hidden_size = 512

# number of layers in lstm
rnn_lstm_number_of_layers = 3

# generation sentence method
rnn_inference = "sample" # "sample" "beam_search"

# number of beam search K
rnn_beam_search_width = 3
