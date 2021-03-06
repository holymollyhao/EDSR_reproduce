import os
import tensorflow as tf
import cv2
import math
import glob

from tensorflow.python.data.experimental import AUTOTUNE

class VIDEO:
    def __init__(self,
                 scale=4,
                 subset='train',
                 downgrade='bicubic',
                 images_dir='dataset/images',
                 caches_dir='dataset/caches'):

        _scales = [2, 3, 4, 8]

        if scale in _scales:
            self.scale = scale
        else:
            raise ValueError(f'scale must be in ${_scales}')

        if subset == 'train':
            self.image_ids = range(0, 60)
        elif subset == 'valid':
            self.image_ids = range(60, 80)
        else:
            raise ValueError("subset must be 'train' or 'valid'")

        _downgrades_a = ['bicubic']
        _downgrades_b = ['bilin']

        if scale == 8 and downgrade != 'bicubic':
            raise ValueError(f'scale 8 only allowed for bicubic downgrade')

        if downgrade in _downgrades_b and scale != 4:
            raise ValueError(f'{downgrade} downgrade requires scale 4')

        if downgrade == 'bicubic' and scale == 8:
            self.downgrade = 'x8'
        elif downgrade in _downgrades_b:
            self.downgrade = downgrade
            bilin_resize(self.hr_images_dir, self._lr_images_dir)
        else:
            self.downgrade = downgrade

        self.subset = subset
        self.images_dir = images_dir
        self.caches_dir = caches_dir

        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(caches_dir, exist_ok=True)

    def __len__(self):
        return len(self.image_ids)

    def dataset(self, batch_size=16, repeat_count=None, random_transform=True):
        print(self._hr_images_dir)
        ds = tf.data.Dataset.zip((self.lr_dataset(), self.hr_dataset()))
        if random_transform:
            ds = ds.map(lambda lr, hr: random_crop(lr, hr, scale=self.scale), num_parallel_calls=AUTOTUNE)
            ds = ds.map(random_rotate, num_parallel_calls=AUTOTUNE)
            ds = ds.map(random_flip, num_parallel_calls=AUTOTUNE)
        if(batch_size != None):
            ds = ds.batch(batch_size)
        ds = ds.repeat(repeat_count)
        ds = ds.prefetch(buffer_size=AUTOTUNE)
        return ds

    def hr_dataset(self):
        if not os.path.exists(self._hr_images_dir()):
            vidtoimage("./dataset/960p_s0_d60.webm", self._hr_images_dir(), fps = 30)

        ds = self._images_dataset(self._hr_image_files()).cache(self._hr_cache_file())

        if not os.path.exists(self._hr_cache_index()):
            self._populate_cache(ds, self._hr_cache_file())

        return ds

    def lr_dataset(self):
        if not os.path.exists(self._lr_images_dir()):
            vidtoimage("./dataset/240p_s0_d60.webm", self._lr_images_dir(), fps = 30, scale = self.scale)

        ds = self._images_dataset(self._lr_image_files()).cache(self._lr_cache_file())

        if not os.path.exists(self._lr_cache_index()):
            self._populate_cache(ds, self._lr_cache_file())

        return ds

    def _hr_cache_file(self):
        return os.path.join(self.caches_dir, f'VIDEO_{self.subset}_HR.cache')

    def _lr_cache_file(self):
        return os.path.join(self.caches_dir, f'VIDEO_{self.subset}_LR_{self.downgrade}_X{self.scale}.cache')

    def _hr_cache_index(self):
        return f'{self._hr_cache_file()}.index'

    def _lr_cache_index(self):
        return f'{self._lr_cache_file()}.index'

    def _hr_image_files(self):
        images_dir = self._hr_images_dir()
        return [os.path.join(images_dir, f'{image_id:04}.png') for image_id in self.image_ids]

    def _lr_image_files(self):
        images_dir = self._lr_images_dir()
        return [os.path.join(images_dir, self._lr_image_file(image_id)) for image_id in self.image_ids]

    def _lr_image_file(self, image_id):
        return f'{image_id:04}x{self.scale}.png'

    def _hr_images_dir(self): 
        return os.path.join(self.images_dir, f'VIDEO_{self.subset}_HR')

    def _lr_images_dir(self):
        return os.path.join(self.images_dir, f'VIDEO_{self.subset}_LR_{self.downgrade}')

    
    def _hr_images_archive(self):
        return f'VIDEO_{self.subset}_HR.zip'

    def _lr_images_archive(self):
        return f'VIDEO_{self.subset}_LR_{self.downgrade}.zip'
        

    @staticmethod
    def _images_dataset(image_files):
        ds = tf.data.Dataset.from_tensor_slices(image_files)
        ds = ds.map(tf.io.read_file)
        ds = ds.map(lambda x: tf.image.decode_png(x, channels=3), num_parallel_calls=AUTOTUNE)
        return ds

    @staticmethod
    def _populate_cache(ds, cache_file):
        print(f'Caching decoded images in {cache_file} ...')
        for _ in ds: pass
        print(f'Cached decoded images in {cache_file}.')


# -----------------------------------------------------------
#  Transformations
# -----------------------------------------------------------


def random_crop(lr_img, hr_img, hr_crop_size=256, scale=2):
    lr_crop_size = hr_crop_size // scale
    lr_img_shape = tf.shape(lr_img)[:2]

    lr_w = tf.random.uniform(shape=(), maxval=lr_img_shape[1] - lr_crop_size + 1, dtype=tf.int32)
    lr_h = tf.random.uniform(shape=(), maxval=lr_img_shape[0] - lr_crop_size + 1, dtype=tf.int32)

    hr_w = lr_w * scale
    hr_h = lr_h * scale

    lr_img_cropped = lr_img[lr_h:lr_h + lr_crop_size, lr_w:lr_w + lr_crop_size]
    hr_img_cropped = hr_img[hr_h:hr_h + hr_crop_size, hr_w:hr_w + hr_crop_size]

    return lr_img_cropped, hr_img_cropped


def random_flip(lr_img, hr_img):
    rn = tf.random.uniform(shape=(), maxval=1)
    return tf.cond(rn < 0.5,
                   lambda: (lr_img, hr_img),
                   lambda: (tf.image.flip_left_right(lr_img),
                            tf.image.flip_left_right(hr_img)))


def random_rotate(lr_img, hr_img):
    rn = tf.random.uniform(shape=(), maxval=4, dtype=tf.int32)
    return tf.image.rot90(lr_img, rn), tf.image.rot90(hr_img, rn)


# -----------------------------------------------------------
#  IO
# -----------------------------------------------------------

def vidtoimage(videopath, imgpath, fps = 1, scale = None):
    vidcap = cv2.VideoCapture(videopath)
    count = 0
    print(str(imgpath))
    print(os.path.exists(imgpath))
    if( not os.path.exists(imgpath)):   
        os.makedirs(imgpath)
        while vidcap.isOpened():
            success, image = vidcap.read()
            if success:
                if(count%fps == 0):
                    if scale:
                        cv2.imwrite(os.path.join(imgpath, '%04d'+'x'+str(scale)+'.png') % (count//fps), image)
                        print("vidtoimage #"+str(count))
                    else:
                        cv2.imwrite(os.path.join(imgpath, '%04d.png') % (count//fps), image)
                        print("vidtoimage #"+str(count))
                count += 1
            else:
                break
        cv2.destroyAllWindows()
        vidcap.release()

def bilin_resize(hr_images_dir, outputpath):
    print(os.path.join(hr_images_dir, "%4d.png" )%(0))
    if not os.path.exists(os.path.join(hr_images_dir, "%4d.png" )%(0)):
        os.makedirs(outputpath, exist_ok = True)
        for i in range(60):
            img = cv2.imread(os.path.join(hr_images_dir, "%04d.png" )%(i))
            print(img)
            res = cv2.resize(img, dsize=(426, 240), interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(os.path.join(outputpath, "%04dx4.png")%(i), res)
            