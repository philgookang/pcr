# Detecting Parts of Speech from Image for Caption Generator
This project is the original source code for the paper (to be inputted).

# How To Use
**1. Clone git**
```
$ git clone https://github.com/philgookang/pcr.git
$ cd pcr
```

**2. Install required library**
```
$ pip install -r requirements.txt
```

**3. Download dataset**  
When you download our dataset, you only download the captions for train, validation, and test. For the actual image, you need to download them at the official website. Also, all of our captions are saved by Pickle. You can only open them in python!
```
$ wget http://pcr.philgookang.com/data.zip
$ unzip data.zip
$ rm data.zip
```

**3.1. Prepare dataset**  
You only need to prepare your dataset if your are using your own custom dataset. Also, only run the code below for the dataset your are using. If you have downloaded our dataset skip this part.
```
$ python ready_dataset_mscoco.py
$ python ready_dataset_flickr30k.py
$ python ready_dataset_flickr8k.py
```


**4. Download pretrained model**
```
$ wget http://pisa.snu.ac.kr/pcr/model.zip
$ unzip model.zip
$ mv model/ ./result/
$ rm model.zip
```
If the link does not work, you can download the pretrained model at this dropbox.
