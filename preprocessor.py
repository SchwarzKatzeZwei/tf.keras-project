import random

import numpy as np
from keras_preprocessing.image.utils import img_to_array
from PIL import Image, ImageDraw, ImageFilter


def scale_to_width(img, width):
    height = round(img.height * width / img.width)
    return img.resize((width, height))


def draw_mask_color_circle(img, color=(255, 0, 0), fill=160, blur=10, size=(10, 25), p=0.5):
    if np.random.rand() > p:
        return img

    _img = img.copy()
    red_img = Image.new('RGB', (128, 128), color)
    mask = Image.new("L", _img.size, 0)
    draw = ImageDraw.Draw(mask)
    x = random.randint(30, _img.width - 50)
    y = random.randint(30, _img.height - 50)
    w = random.randint(size[0], size[1])
    h = random.randint(size[0], size[1])
    if color == (255, 255, 255):  # 白はちょっと大きくする
        x += 15
        y += 15
        w += 15
        h += 15

    if isinstance(fill, list):
        fill = random.randint(fill[0], fill[1])

    draw.ellipse((x, y, x + w, y + h), fill=fill)
    mask_blur = mask.filter(ImageFilter.GaussianBlur(blur))
    # mask_blur.save("draw.jpg")
    im = Image.composite(red_img, _img, mask_blur)
    # im.save("draw2.jpg")
    return im


def random_erasing(image_origin, p=0.5, s=(0.01, 0.06), r=(0.5, 1.5)):
    # マスクするかしないか
    if np.random.rand() > p:
        return image_origin

    image = np.copy(image_origin)

    # マスクする画素値をランダムで決める
    mask_value = np.random.randint(0, 256)
    # mask_value = np.random.choice([1, 256])
    # mask_value = random.choice([0, 256])

    h, w, _ = image.shape
    # マスクのサイズを元画像のs(0.02~0.4)倍の範囲からランダムに決める
    mask_area = np.random.randint(h * w * s[0], h * w * s[1])

    # マスクのアスペクト比をr(0.3~3)の範囲からランダムに決める
    mask_aspect_ratio = np.random.rand() * r[1] + r[0]

    # マスクのサイズとアスペクト比からマスクの高さと幅を決める
    # 算出した高さと幅(のどちらか)が元画像より大きくなることがあるので修正する
    mask_height = int(np.sqrt(mask_area / mask_aspect_ratio))
    if mask_height > h - 1:
        mask_height = h - 1
    mask_width = int(mask_aspect_ratio * mask_height)
    if mask_width > w - 1:
        mask_width = w - 1

    top = np.random.randint(0, h - mask_height)
    left = np.random.randint(0, w - mask_width)
    bottom = top + mask_height
    right = left + mask_width
    image[top:bottom, left:right, :].fill(mask_value)
    return image


def preprocessor_gen(gen):
    for data, labels in gen:
        temp_img_array_list = []
        for idx, d in enumerate(data):
            img = Image.fromarray(np.uint8(d * 255))

            if labels.argmax() == 0:  # NGなら
                num = random.choice([0, 1, 2])
                # num = 1
                if num == 0:  # 赤
                    img = draw_mask_color_circle(img, color=(255, 0, 0), size=(30, 40), fill=64, blur=3, p=1.0)
                elif num == 1:  # 黒
                    img = draw_mask_color_circle(img, color=(0, 0, 0), fill=[64, 128], blur=2, p=1.0)
                else:  # 白
                    img = draw_mask_color_circle(img, color=(255, 255, 255), fill=255, blur=1, p=1.0)

            temp_img_array = img_to_array(img)

            # if labels.argmax() == 0 and np.random.rand() > 1.0:  # NGなら
            #     for _ in range(100):
            #         # temp_img_array = random_erasing(temp_img_array, s=(0.001, 0.0015), p=1.0)
            #         temp_img_array = random_erasing(temp_img_array, s=(0.01, 0.015), p=1.0)

            aimg = Image.fromarray(np.uint8(temp_img_array))
            aimg.save(f"tmp/{labels.argmax()}_{idx}.jpg")

            temp_img_array = temp_img_array / 255.
            temp_img_array_list.append(temp_img_array)
        data = np.array(temp_img_array_list)

        yield data, labels


def preprocessor_gen2(gen):
    for data, labels in gen:
        temp_img_array_list = []
        for idx, d in enumerate(data):
            img = Image.fromarray(np.uint8(d * 255))
            temp_img_array = img_to_array(img)
            temp_img_array = temp_img_array / 255.
            temp_img_array_list.append(temp_img_array)
        data = np.array(temp_img_array_list)
        yield data, labels
