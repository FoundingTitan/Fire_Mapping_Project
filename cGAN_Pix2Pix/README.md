# cGAN - Pix2Pix

Implementation based on Pix2Pix paper *Image-to-Image Translation with Conditional Adversarial Networks by Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, Alexei A. Efros*, modifying code from [aladdinpersson](https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/GANs/Pix2Pix/README.md).

## Recreate the environment

1. Create a new folder `evaluation` (to the root of this `cGAN_Pix2Pix` project folder)

2. Unzip the data file [`data`](https://drive.google.com/drive/folders/1srBgeb3p1m4xT6-VwvvzOPWwZdp09jVg?usp=sharing) (to the root of this `cGAN_Pix2Pix` project folder)

The file structure should be:

```
/data
+- train
|  +- x
|  +- y
+- val
|  +- x
|  +- y
```

Where `x` holds the lines-scan images, and `y` holds the masks.

3. Copy and paste the contents (including the two folders above) into the root of your google drive.

## Configuration

* In the `dataset.py`, some code is commented out. They are methods to be used when training with black/white images. Default used for color images.

If the training images are already in RGB format, you can comment out the `(from 'CMYK' to 'RGB')` block of code.

For B/W images, uncomment the `(Use for bw images)` block, and comment out the `(from 'CMYK' to 'RGB')` and the `(Use for color images)` blocks.

* In the `config.py` there are several self-explanatory options to change. Most notably, the `NUM_EPOCHS` can be set here. At least `1000` is recommended.

## Training

1. Open a new colab session. Change the runtime to `GPU` (See the first `README.md`) of the firemapping project.

2. Edit the config.py file to match the setup you want to use. Then run `train.py`

3. Mount your Google Drive.

4. In the first cell in the colab notebook run:

```
!pip install -U albumentations
%cd /content/drive/MyDrive/Fire_Mapping_Project/cGAN_Pix2Pix
!python train.py
```

### Evaluation

At the end of training, the metrics with each EPOCH will be saved to a `loss_tracker.csv` file.

### Loading a saved model.

1. Copy the model `gen.pth.tar` into the root directory of this project.

`Fire_Mapping_Project/cGAN_Pix2Pix/gen.pth.tar`

2. In `config.py`, change `SAVE_MODEL=False` and `LOAD_MODEL=True`

3. Empty the `evaluation` folder.

4. Place desired input masks for the generator in the `val/y` folder

5. In `train.py` comment out

```
L1, G_loss, D_loss, G_fake_loss, D_fake_loss, D_real_loss = train_fn(
        disc, gen, train_loader, opt_disc, opt_gen, L1_LOSS, BCE, g_scaler, d_scaler,
    )

    loss_tracker = loss_tracker.append({'EPOCH':epoch,
                                            'L1_loss':L1,
                                            'G_loss':G_loss,
                                            'D_loss':D_loss,
                                            'G_fake_loss':G_fake_loss,
                                            'D_fake_loss':D_fake_loss,
                                            'D_real_loss':D_real_loss},
                                           ignore_index=True)

        if config.SAVE_MODEL and epoch % 5 == 0:
            save_checkpoint(gen, opt_gen, filename=config.CHECKPOINT_GEN)
            save_checkpoint(disc, opt_disc, filename=config.CHECKPOINT_DISC)

...

loss_tracker.to_csv('loss_tracker.csv', index=False)
```
4. Generated/syntheic line-scan images will be found in the `evaluation` folder.

Note that you can augment the images in `val/y` by changing the `transform_only_input` in the `config.py` file.

For example:

```
transform_only_input = A.Compose(
  [
    A.HorizontalFlip(p=0.8),
    A.VerticalFlip(p=0.8),
    A.RandomRotate90(p=0.8),
    A.Transpose(p=0.8),
    A.Normalize(mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5],
                max_pixel_value=255.0),
    ToTensorV2(),
  ]
  )
```

## Pix2Pix paper

```
@misc{isola2018imagetoimage,
      title={Image-to-Image Translation with Conditional Adversarial Networks},
      author={Phillip Isola and Jun-Yan Zhu and Tinghui Zhou and Alexei A. Efros},
      year={2018},
      eprint={1611.07004},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
