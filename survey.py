from config import *
from helper import *
from evaluation import *

import pickle
import sys

from shutil import copyfile
from tqdm import tqdm


if __name__ == "__main__":

    # with open('result/survey_result.pkl', 'rb') as f:
    #     dataset = pickle.load(f)
    #
    # for percent in dataset:
    #     print(percent, len(dataset[percent]))

    with open("result/survey/pcr_coco2017.pkl", "rb") as f:
        dataset = pickle.load(f)

    percentile_count = {
        10 : [], 20: [], 30: [], 40: [], 50: [], 60: [], 70: [], 80: [], 90: [], 100 : []
    }

    # tqdm
    id = 0
    for item in tqdm(dataset):

        bleu = Bleu()
        bscore = bleu.compute([item])

        for i in range(10, 110, 10):
            score = bscore[3] * 100
            if score <= i:
                percentile_count[i].append({
                    "reference" : item["reference"],
                    "pcr" : {
                        "score" : score,
                        "caption" : item['hypothesis']
                    }
                })
                # print(percentile_count)
                # print(phil)
                break

        id += 1

    with open('result/survey_result.pkl', 'wb') as f:
        pickle.dump(percentile_count, f)
