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
You only download the captions for train, validation, and test. For the actual image, you need to download them at the official website. 
```
$ wget http://pcr.philgookang.com/data.zip
$ unzip data.zip
$ rm data.zip
```

**3. Download pretrained model**
```
$ wget http://pisa.snu.ac.kr/pcr/model.zip
$ unzip model.zip
$ mv model/ ./result/
$ rm model.zip
```
If the link does not work, you can download the pretrained model at this dropbox.
