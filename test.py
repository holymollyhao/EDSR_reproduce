from input_output import *
from edsr import *
from extract_frame import *
import imageio
from prepare import *
import matplotlib.pyplot as plt



train = DIV2K(scale=4, downgrade='bicubic', subset='train')
train_ds = train.dataset(batch_size=16, random_transform=True)

edsr_model = load_model()

vidtoimage("./dataset/240p_s0_d60.webm", "./dataset/outputimages240")
image_srall(edsr_model, "./dataset/outputimages240", "./dataset/outputimages_sr240")
imagetovid("./dataset/outputimages_sr240", "./dataset", "output.webm", 30)
