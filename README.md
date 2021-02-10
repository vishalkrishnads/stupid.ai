# stupid.ai
This repository mainly serves as a reference for myself from other PC's. So, you might be better off not even minding this 😂
But if you can't get your head around this repo, then let me tell you two things:
* Firstly, you need to get a life
* Secondly, you can pass in any image dataset with labels to this template to train and predict from a live image feed, which is the only use of this code 😁

### Setting up the environment

1. Install and set up [Python 3.8.6](https://www.python.org/downloads/release/python-386/)
2. Create a virtual environment
      ```
      python -m venv --system-site-packages .\venv
      ```
3. Install all the required modules
      ```
      pip install opencv-python tensorflow scikit-learn
      ```
4. Verify that `tensorflow` is working by testing it out in the Python console
      ```
      $ python
      >>> import tensorflow
      ```
   which should hopefully return something like
      ```
      2021-02-10 22:20:28.734312: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library cudart64_101.dll
      ```
5. If you see the words, `Successfully opened dynamic library`, then it means exactly what it says. You're good to go. If you face any problems regarding installing `tensorflow` anywhere down the line, feel free to search on the internet 😉. OMG, please don't open an issue for this here.

6. Clone this repo to your machine
      ```
      $ git clone https://github.com/vishalkrishnads/stupid.ai.git
      ```
### Configuring the dataset

You can configure the dataset into images and their label keys as folder names. For example, if you're training an AI for counting your fingers live, put all the images of you showing 1 with your finger in a subdirectory named 1, all images of 2 in a subdirectory named 2 and so on. Do note that you have to give integer values (whose value is greater than 1) as names to your subdirectories. Any string value will throw an error and quit immediately. Finally, put all these subdirectories into a single directory, name it whatever you want, and that's your dataset.

A general format for your dataset can be
```
dataset
   |
   |--1
   |  |--all image files corresponding to label key 1
   |
   |--2
   |  |--all image files corresponding to label key 2
   |
   |--and so on
```
where `dataset` becomes the name of your dataset when training

### Using stupid.ai

You can train your AI by running the `train.py` and passing in the name of your dataset as argument.

```
$ python train.py <your-dataset>
$ python train.py mydataset # where mydataset is your dataset directory
```

You can make your neural network predict on a live image feed by running `recognise.py` and passing in the saved model's filename as argument

```
$ python recognise.py <your-model>
$ python recognise.py model.h5 # where model.h5 is your saved model
```

### Changing the network

The lines 81-94 in `train.py` model the neural network. You can change the network as per your needs. The configs for the image files and neural network are set as variables in lines 11-15 by default as
```
EPOCHS = 40
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 6
TEST_SIZE = 0.4
```

### Contributing

Really, did you think that I'm asking you to contribute to this template? 😆 That's it, thank you for wasting your time here.
