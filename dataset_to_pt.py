import torch
from torchvision import transforms
from PIL import Image
import os, glob

trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.Resize([224, 224])
])

def to_pt(path):
    image_paths = [*glob.glob(os.path.join(path, "*/*.JPG")), *glob.glob(os.path.join(path, "*/*.jpg"))]
    # image_paths = glob.glob(os.path.join(path, "*.JPG"))
    print(len(image_paths))

    for i, image_path in enumerate(image_paths):
        print(i, image_path)
        image = Image.open(image_path)
        image = trans(image)

        path_arr = image_path.split(os.path.sep)
        fileName = path_arr[-1]
        fileName = '.'.join([*fileName.split('.')[:-1], 'pt'])
        dirPath = os.path.join(f'{path}_pt', (os.path.sep).join(path_arr[-2:-1]))

        if not os.path.exists(dirPath):
            os.makedirs(dirPath)
        torch.save(image, os.path.join(dirPath, fileName))

if __name__ == '__main__':
    print('[Train]')
    to_pt( './dataset/train/families')
    print('[Test]')
    to_pt( './dataset/test/families')