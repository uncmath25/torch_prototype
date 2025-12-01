import numpy as np
from PIL import Image
import torchvision.transforms as transforms

from data_loader import DataLoader


class Utils:
    @staticmethod
    def _show_image(idx):
        # 60000, 28, 28
        images = DataLoader.get_images(is_train=True)
        for row in images[idx]:
            print(row)
        Image.fromarray(images[idx]).show()
        trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        # 28, 60000, 28
        images = trans(images)
        image = (images[:, idx, :].T.numpy()*255).astype(np.uint8)
        for row in image:
            print(row)
        Image.fromarray(image).show()
