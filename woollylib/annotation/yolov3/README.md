## YoloV3-Annotation

The goal is to create text files in darknet yolo format that will be fed to training to the CNN. Below is explained pretty much how to use the functionalities.

**Disclaimer:** Credits to ManivannanMurugavel, you can find his repo here [Yolo-Annotation-Tool-New](https://github.com/ManivannanMurugavel/Yolo-Annotation-Tool-New-). Current code here is somewhat a correction of his repo.

Create a folder names `Images` and put all image files in this folder make sure images are `.jp*g` format

Before starting to annotate change image size to default size images.

```
python3 resize.py
```

Install tkinter

```
sudo apt-get install python3-tk
pip install tk
```

Create a new `classes.txt` file inside `current` folder with all classes names

```
hardhat
vest
mask
boots
```


Now run following for loading and annotating images

```
python3 main.py
```

once we have annotated images we will create test and train splits

```
-a for relative path to Images
-p for percent of split for test Images
```

```
python3 process.py -a Images -p 1
```



