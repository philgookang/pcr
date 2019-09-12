from config.config_train import dataset_type

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
    "validation"        : "cnn_rnn_validation_dataset.pkl",
    "result"            : "cnn_rnn_caption_result.pkl"
}

if dataset_type == 2:
    dataset_file["train"] =         "cnn_rnn_train_dataset_flickr8k.pkl"
    dataset_file["test"] =          "cnn_rnn_test_dataset_flickr8k.pkl"
    dataset_file["validation"] =   "cnn_rnn_validation_dataset_flickr8k.pkl"
    dataset_file["result"] =        "cnn_rnn_caption_result_flickr8k.pkl"

dataset_skip_file = {
    "train"             : "skip_train.csv",
    "validation"        : "skip_validation.csv",
    "noun"              : "skip_noun.csv",
    "pronoun"           : "skip_pronoun.csv",
    "verb"              : "skip_verb.csv",
    "adjective"         : "skip_adjective.csv",
    "adverb"            : "skip_adverb.csv",
    "conjunction"       : "skip_conjunction.csv",
    "preposition"       : "skip_preposition.csv",
    "interjection"      : "skip_interjection.csv"
}
