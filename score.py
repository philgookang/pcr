from config import *
from helper import *
from evaluation import *

import pickle
import sys


def run_score_on_file(filename):

    with open(filename, "rb") as f:
        dataset = pickle.load(f)

    run_score(dataset)


def run_score(dataset):
    # get bleu score
    bleu = Bleu()
    bscore = bleu.compute(dataset)

    # get cider score
    cider = Cider()
    cscore = cider.compute(dataset)

    # get rouge score
    rouge = Rouge()
    rscore = rouge.compute(dataset)

    print("Scores")
    print("BLEU: ", bscore)
    print("CIDEr: ", cscore)
    print("ROUGE: ", rscore)


if __name__ == "__main__":

    if len(sys.argv) >= 2:

        # get filename
        filename = sys.argv[1]

        # open file
        run_score_on_file(filename)
