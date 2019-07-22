model_file = {
    "noun"          : { "pretrain" : "cnn_noun_pretrain.pt", "train" : "cnn_noun_train.pt" },
    "pronoun"       : { "pretrain" : "cnn_pronoun_pretrain.pt", "train" : "cnn_pronoun_train.pt" },
    "verb"          : { "pretrain" : "cnn_verb_pretrain.pt", "train" : "cnn_verb_train.pt" },
    "adjective"     : { "pretrain" : "cnn_adjective_pretrain.pt", "train" : "cnn_adjective_train.pt" },
    "adverb"        : { "pretrain" : "cnn_adverb_pretrain.pt", "train" : "cnn_adverb_train.pt" },
    "conjunction"   : { "pretrain" : "cnn_conjunction_pretrain.pt", "train" : "cnn_conjunction_train.pt" },
    "preposition"   : { "pretrain" : "cnn_preposition_pretrain.pt", "train" : "cnn_preposition_train.pt" },
    "interjection"  : { "pretrain" : "cnn_interjection_pretrain.pt", "train" : "cnn_interjection_train.pt" },
    "decoder"       : { "train" : "rnn_decoder.pt" }
}

dataset_file = {
    "pretrain"          : "cnn_pretrain_dataset.pkl",
    "train"             : "cnn_rnn_train_dataset.pkl",
    "test"              : "cnn_rnn_test_dataset.pkl",
    "validation"        : "cnn_rnn_validation_dataset.pkl"
}
