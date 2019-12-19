from config import *
from helper import *
from evaluation import *
from evaluation.spice.spice import Spice

import pickle
import sys

def run_score_on_file(filename):

    with open(filename, "rb") as f:
        dataset = pickle.load(f)

    run_score(dataset)

def run_score(dataset):

    # get spice score
    sscore = run_score_spice(dataset)

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
    print("SPICE: ", sscore)

def run_score_spice(dataset):

    result_lst = {}

    for id, item in enumerate(dataset):
        result_lst["image_" + str(id)] = item

    # fixed the hypothesis/caption wrong lableing
    reference = { }
    hypothesis = { }
    no = 0
    for filename in result_lst:
        for ref in result_lst[filename]["reference"]:
            reference[no] = [" ".join(ref)]
            hypothesis[no] = [" ".join(result_lst[filename]["hypothesis"])]
            no += 1

    # get spice score
    spice = Spice()
    sscore = spice.compute_score(reference, hypothesis)

    return sscore[0]

if __name__ == "__main__":

    if len(sys.argv) >= 2:

        # get filename
        filename = sys.argv[1]

        # open file
        run_score_on_file(filename)
