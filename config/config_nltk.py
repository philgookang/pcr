
nltk_to_pos = {
    "noun"          : ["CD", "NN", "NNS", "NNP", "NNPS", "PDT"],
    "pronoun"       : ["CD", "PRP", "PRP$", "WP", "WP$"],
    "verb"          : ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"],
    "adjective"     : ["CD", "DT", "JJ", "JJR", "JJS"],
    "adverb"        : ["CD", "RB", "RBR", "RBS", "WRB"],
    "conjunction"   : ["CC", "IN"],
    "preposition"   : ["IN"],
    "interjection"  : ["UH"]
}


pos_to_nltk = {
    "NN" : "noun", "NNS" : "noun", "NNP" : "noun", "NNPS" : "noun", "PDT" : "noun",
    "PRP" : "pronoun", "PRP$" : "pronoun", "WP" : "pronoun", "WP$" : "pronoun",
    "VB" : "verb", "VBD" : "verb", "VBG" : "verb", "VBN" : "verb", "VBP" : "verb", "VBZ" : "verb",
    "DT" : "adjective", "JJ" : "adjective", "JJR" : "adjective", "JJS" : "adjective",
    "RB" : "adverb", "RBR" : "adverb", "RBS" : "adverb", "WRB" : "adverb",
}
