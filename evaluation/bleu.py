from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from multiprocessing import Pool

class Bleu():

    def __init__(self):
        pass

    def compute_wrapper_1(self, x):
        return self.compute_score(x["reference"], x["hypothesis"], SmoothingFunction().method1)

    def compute_wrapper_2(self, x):
        return self.compute_score(x["reference"], x["hypothesis"], SmoothingFunction().method2)

    def compute_wrapper_3(self, x):
        return self.compute_score(x["reference"], x["hypothesis"], SmoothingFunction().method3)

    def compute_wrapper_4(self, x):
        return self.compute_score(x["reference"], x["hypothesis"], SmoothingFunction().method4)

    def compute(self, dataset):
        '''
            structure of datatset variable
            [
                {
                    "reference": [ [], [], [] ],
                    "hypothesis": [ ],
                    "skip": True/False
                }
            ]
        '''
        with Pool(30) as p:
            s1 = list(p.map(self.compute_wrapper_1, dataset))
            s2 = list(p.map(self.compute_wrapper_2, dataset))
            s3 = list(p.map(self.compute_wrapper_3, dataset))
            s4 = list(p.map(self.compute_wrapper_4, dataset))
        return (sum(s1)/len(s1)), (sum(s2)/len(s2)), (sum(s3)/len(s3)), (sum(s4)/len(s4))

    def compute_score(self, reference, hypothesis, smoothing_function = SmoothingFunction().method0):
        return sentence_bleu(reference, hypothesis, smoothing_function = smoothing_function)

if __name__ == "__main__":
    b = Bleu()
    s = b.compute([{"reference" : [ ["this", "really", "work"], ["does", "this", "work"] ], "hypothesis": ['it', 'does', 'really', 'work']}])
    print(s)
