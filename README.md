# PoS Based CNN-RNN (PCR) Image Captioning Sub-model Generation
This project is a simple CNN-RNN based image caption

**Part of Speech**
1. Noun
2. Pronoun
3. Verb
4. Adjective
5. Adverb
6. Conjunction
7. Preposition
8. Interjection


# How To Use
**1. Clone git**
```
$ git clone https://github.com/philgookang/pos_cnn.git
$ cd pos_cnn
```

**2. Install required library**
```
$ pip install -r requirements.txt
```

**3. Download dataset**
```
$ wget http://pisa.coco.philgookang.com/data.zip
$ unzip data.zip
$ rm data.zip
```

**4. Prepare corpus**

**4.1 Generate corpus**
```
$ python preprocess.py
```
**4.2 Download generated corpus**
```
$ cd pos_cnn/result/
$ wget http://pisa.coco.philgookang.com/corpus.pkl
```

A download link from dropbox is [available here](https://www.dropbox.com/s/e8fy41mthl4lpjq/corpus.pkl?dl=0)


**5. Train PoS models**

**5.1 Train them yourself**

Add the grammer model you want to train to the list ```pretrain.py```
```
    # back stats
    cudnn.benchmark = True

    # loop through list
    for pos in ["noun", "verb", "adjective", "conjunction", "preposition"]:

        # set book target
        book.set_target(pos)
```
```
$ python train.py
```
**5.2 Download pretrained models**
```
$ cd pos_cnn/result/
$ wget http://pisa.coco.philgookang.com/cnn_noun.pt
$ wget http://pisa.coco.philgookang.com/cnn_verb.pt
$ wget http://pisa.coco.philgookang.com/cnn_adjective.pt
$ wget http://pisa.coco.philgookang.com/cnn_conjunction.pt
$ wget http://pisa.coco.philgookang.com/cnn_preposition.pt
```
Dropbox download link:
* Noun: [link](https://www.dropbox.com/s/x8ifmjrlfxgtfxz/cnn_noun.pt?dl=0)
* Verb: [link](https://www.dropbox.com/s/o7s5ak12upn4lt7/cnn_verb.pt?dl=0)
* Adjective: [link](https://www.dropbox.com/s/da9fopl3436yrh4/cnn_adjective.pt?dl=0)
* Conjunction: [link](https://www.dropbox.com/s/ts1k5iivgz8sjm7/cnn_conjunction.pt?dl=0)
* Preposition: [link](https://www.dropbox.com/s/p4bxwcaaaysnb4i/cnn_preposition.pt?dl=0)


**6. Evaluate PoS models**
```
$ python test.py
```
