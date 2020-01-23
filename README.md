# EDSR_reproduce

A reproduction of EDSR, NAS-EDSR using keras, tensorflow

## Structure

```
EDSR_reproduce
├── DIV2K_data, VIDEO_data.py 
├── preprocess,postprocess.py 
├── input_output.py 
├── EDSR.py 
└── test.py 
```

* `DIV2K_data, VIDEO_data.py` defined as a class, for preeparing the images from DIV2K dataset, and a minute long video.
The image is cropped and randomly flipped & rotated

* `preprocess,postprocess.py` is the training part & evaluation part of the made models

* `input_output.py` for single resolving & loading & plotting images

* `EDSR.py` actual deep learning code

* `test.py` testing & checking different models EDSR, NAS-EDSR, Fine-tuned version

#### LR image
<img width="800" alt="image" src="https://user-images.githubusercontent.com/50355670/72964747-0ac6a700-3dfe-11ea-89b7-363ea347feeb.png">

#### SR image (NAS-EDSR)
<img width="800" alt="image" src="https://user-images.githubusercontent.com/50355670/72964726-026e6c00-3dfe-11ea-9028-e802e4ed711d.png" title="SRdsadsdsasddsa">

#### HR image
<img width="800" alt="image" src="https://user-images.githubusercontent.com/50355670/72964760-10bc8800-3dfe-11ea-9c32-56f9e411d9b4.png">

#### EDSR (x4) version
PSNR value : 24.425, SSIM value : 0.8134

#### NAS - EDSR (x4) version
PSNR value : 27.852, SSIM value : 0.8763

