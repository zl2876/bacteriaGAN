# bacteriaGAN
To attempt to build a relatively large-scale dataset of high-quality, labeled images, this project employs the pix2pix conditional GAN model by Isola et al (2017)  to artificially synthesize images by training on the smaller currently available dataset. For assessment, a image classification model to evaluate if bacterial images can be differentiated accurately when additionally trained on synthesized images is included. 

## Prerequisites
- Linux or macOS
- Python 3
- NVIDIA GPU + CUDA CuDNN

## Getting Started

### Accessing data & preparation 
The Digital Images of Bacteria Species dataset (DIBaS), collected by Jagiellonian University in Krakow, Poland, contains 33 bacteria species with around 20 images for each. All of the samples were stained using the Gramm’s method, and evaluated using a 100 times objective. Each original image file is in TIF format, and the size is 2048 x 1532 pixels when converted to PNG format. The dataset has been uploaded to Google Drive for use on this project only, and can be accessed and downloaded from this link using a LionMail account. 

DIBaS Original Dataset for classification[https://drive.google.com/drive/folders/125ukQizEPnNS4KhASyOyhWi-Rse9uynu?usp=sharing]
DIBaS Sorted Dataset for pix2pix[https://drive.google.com/drive/folders/1sRcxDldxM6WQz1JjvO_IqI375obUv6wC?usp=sharing]

- **Add shortcut to MyDrive or any subfolder within MyDrive.**

## cGAN `dibas_pix2pix.ipynb`

To train and test the pix2pix on our dataset, run `dibas_pix2pix.ipynb` using Google Colab. The contents of the notebook are as follows.

- **Clone pix2pix repo**
```
!git clone https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
import os
os.chdir('pytorch-CycleGAN-and-pix2pix/')
!pip install -r requirements.txt
```
- **Load dataset**

Requires authentication of Google account with access to data files.
```
from google.colab import drive
drive.mount('/content/gdrive', force_remount = True)
```
- **Prepare dataset**

Pix2pix's training requires paired data. A python script to generate training data in the form of pairs of images {A,B}, where A and B are two different depictions of the same underlying scene, is included in pix2pix repo. To use this script, create folder /path/to/data with subdirectories A and B. A and B should each have their own subdirectories train, val, test, etc. In /path/to/data/A/train, put training images in style A. In /path/to/data/B/train, put the corresponding images in style B. Repeat same for other data splits (val, test, etc). Corresponding images in a pair {A,B} must be the same size and have the same filename, e.g., /path/to/data/A/train/1.jpg is considered to correspond to /path/to/data/B/train/1.jpg. 

Our provided DIBaS Sorted Dataset has already been sorted as follows:
[https://docs.google.com/spreadsheets/d/1jfPMpKVbrTJhCY1nUnlw3GetuzDloQFEjHqC0JkJvPg/edit?usp=sharing]

Once the data is formatted, call:
```
!python ./datasets/combine_A_and_B.py  --fold_A /path/to/data/A --fold_B /path/to/data/B --fold_AB /path/to/data 
```
where if accessed using Google Drive with shortcut of dataset on My Drive would be 
```
!python ./datasets/combine_A_and_B.py --fold_A /content/gdrive/"My Drive"/finalproject/A 
--fold_B /content/gdrive/"My Drive"/finalproject/B --fold_AB /content/gdrive/"My Drive"/finalproject/AB
```
- **Training**

`python train.py --dataroot /content/gdrive/"My Drive"/finalproject/AB --name dibas_pix2pix --model pix2pix --direction BtoA`

Change the `--dataroot` and `--name` to your own dataset's path and model's name. Use `--gpu_ids 0,1,..` to train on multiple GPUs and `--batch_size` to change the batch size. Add `--direction BtoA` if you want to train a model to transfrom from class B to A.

- **Testing**

`python test.py --dataroot /content/gdrive/"My Drive"/finalproject/AB --direction BtoA --model pix2pix --name dibas_pix2pix`

Change the `--dataroot`, `--name`, and `--direction` to be consistent with your trained model's configuration and how you want to transform images.
Outputs will be saved in as a zip file in `/content/gdrive/"My Drive"/finalproject/pix2pix_results.zip`. 

- **Assessment**

We have included code to calculate the Inception score for evaluation of generated images using the implementation from the Salimans et al. (2016) paper. This will take the path to the folder of generated images, resize all of the images within the folder, and returns a single value Inception Score. 


## Image classifier `dibas_classifier.ipynb`

To train and test the pix2pix on our dataset, run `dibas_classifier.ipynb` on Google Colab using a GPU. The main contents of the notebook are as follows.

- **Load & prepare original dataset**

We will use torchvision and torch.utils.data packages for loading the data. Training images will be resized to 256x256 and randomly flipped horziontally (data augmentation) and normalized. Valudation images will be resized to 256x256 and normalized. 

To load the dataset (with shortcut on your "My Drive"), we use `torchvision.datasets.ImageFolder` with root as the path to the DIBaS Original Dataset, then split the data using sklearn.model_selection.train_tests_split with test_size = 0.25 so that train-to-val ratio is 8:2 when generated images are added to the training set.

- **Load & prepare pix2pix generated dataset**

Unzip `/content/gdrive/"My Drive"/finalproject/pix2pix_results.zip`, or zip file of dibas_pix2pix.ipynb test results. 

As outputs from dibas_pix2pix.ipynb will not be sorted into the format required for `torchvision.datasets.ImageFolder`, we have created a `moveFile` method to sort images into subfolders for their strain by their file names. Pass the variable `path = '/content/gdrive/"My Drive"/finalproject/pix2pix_results/'`, or the location of the folder with pix2pix generated image outputs (unzipped) in your workspace. 

Then, load the data using ImageFolder and concatenate with previously loaded training data. 

For the data loaders, set dataset names appropriately for train and val loaders and adjust training batch size to length of `data_train_wfake`.

- **Finetuning the convnet**

Load a pretrained model and reset final fully connected layer. Here, we use MADGRAD from Facebook AI, which has been pip installed in the first cell from [https://github.com/facebookresearch/madgrad]. A lower weight decay than normal may be applied for better results, often 0.

- **Training**

Adjust epoch number and train model.
```
model_ft, loss, acc = train_model(model, criterion, optim, num_epochs=epoch)
```
- **Evaluation**

When executed, will return plot graphs of training/validation accuracy and loss, per-class accuracy as an array, confusion matrix, classsification report, and ten samples with true/predicted labels. 

