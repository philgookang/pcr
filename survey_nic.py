from config import *
from helper import *
from evaluation import *

import pickle
import sys

from shutil import copyfile
from tqdm import tqdm


import os

if __name__ == "__main__":


    with open("result/survey_result.pkl", "rb") as f:
        survey_db = pickle.load(f)

    for i in tqdm(survey_db):
        print(i)
        with open(str(i) + ".txt", "w")  as f:
            for item in tqdm(survey_db[i]):
                f.write("{0} {1}\n".format(item["pcr"]["score"], item["pcr"]["caption"]))

    print(phil)

    # with open('result/survey_result.pkl', 'rb') as f:
    #     dataset = pickle.load(f)
    #
    # for percent in dataset:
    #     print(percent, len(dataset[percent]))

    test_dataset = load_dataset('cnn_rnn_test_dataset.pkl')

    with open("result/survey_result.pkl", "rb") as f:
        survey_db = pickle.load(f)

    with open("result/survey/nic_coco2017.pkl", "rb") as f:
        dataset = pickle.load(f)

    # tqdm
    id = 0
    for item in tqdm(dataset):

        bleu = Bleu()
        bscore = bleu.compute([item])

        for i in range(10, 110, 10):
            score = bscore[3] * 100
            if score <= i:
                for previous_item in survey_db[i]:
                    if previous_item["reference"][0] == item["reference"][0] and previous_item["reference"][1] == item["reference"][1]:

                        def get_file(ref1, ref2, ref3):
                            for fff in test_dataset:
                                org_test = test_dataset[fff]
                                if org_test[0] == ref1 and org_test[1] == ref2 and org_test[2] == ref3:
                                    return fff

                        filename = get_file(previous_item["reference"][0], previous_item["reference"][1], previous_item["reference"][2])

                        lst = []
                        lst.append('GT: ' + ' '.join(previous_item["reference"][0]))
                        lst.append('PCR: ' + ' '.join(previous_item['pcr']['caption']))
                        lst.append('NIC: ' + ' '.join(item['hypothesis']))
                        lst.append( str(previous_item['pcr']['score']) + " " + str(score))

                        create_image_caption(os.path.join(COCO_IMAGE_PATH, filename), os.path.join(RESULT_IMAGE_W_CAPTION, str(i) + "_" + filename), lst)

                # print(percentile_count)
                # print(phil)
                break

        id += 1
    #
    # with open('result/survey_result.pkl', 'wb') as f:
    #     pickle.dump(percentile_count, f)
