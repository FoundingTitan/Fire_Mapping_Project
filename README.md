# Fire_Mapping_Project

## Setting up the environment

1. Clone the repo.

```
git clone https://github.com/FoundingTitan/Fire_Mapping_Project.git
```

2. Copy the contents and paste it into the root of Google Drive.

```
/content/drive/MyDrive/Fire_Mapping_Project
```

3. Open a colab session at https://colab.research.google.com/

Make sure the session is set to `GPU`

Runtime > Change runtime Type > Hardware accelerator (from **None** to **GPU**)

4. Mount your google drive

```
from google.colab import drive
drive.mount('/content/drive')
```

4. In the first cell run

```
%cd /content/drive/MyDrive/Fire_Mapping_Project
```

## Training

```
!python train.py
```

* Additional arguments can be supplied

Use of additional arguments

Example form of `--arg option`
```
!python -W ignore train.py --net attn_unet --epochs 200 > output_attn_unet.txt
```

* Specify model type `--net`
  * `unet`
  * `attn_unet`

* Specify number of epochs `--epochs`
  * Default `100`
  * Any `int` value

* Batch-size `--batch_size`
  *  Default `8`
  *  Any `int` value

* cutoff `--cutoff`
  * Default `0.3`
  * Any `float` value

* learning rate `--lr`
  * Default `0.001`
  * Any `float` value

* Print loss values every log_interval epochs `--log_interval`
  * Default `1`
  * Any `int` value

* Transform data during training mode `--transform_mode`
  * Default Basic `basic`
  * Transform `transform`

* Transform type `--transform_types`
  * Default Crop `crop`
  * Horizontal Flip `hflip`
  * Vertical Flip `vflip`

* Set training seed `--seed`
  * Default `10`
  * Any `int` value. 

## Alternative fast setup

* After putting the project in the root directory in the google drive + mounting.
* Go to Google colab and go File > Upload notebook > (change to the Upload tab) > Choose File.
* Choose the `Main_train.ipynb` or `Main_Train_AUC.ipynb`.
* Make sure the Session runtime is set to GPU (see above).
* Run all cells.

## Adding augmented images

* Edit aug.py and add desired transforms at the top of the code.  This utilizes the Albumentations package
* Determine file names during the saving stage located at the bottom of the code
* Run aug.py
* Copy and paste generated images located at augmented_images onto train_images
* Do the same for generated masks, from augmented_masks onto train_masks

## Fire Prediction

* Open Wild_Fire.py with text editor to select image, image path and initial conditions
* Run Wild_Fire.py when satified.  
