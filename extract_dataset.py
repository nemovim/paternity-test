import glob
import os
from PIL import Image
import numpy as np
import cv2
import shutil

def copyImg(src, dst):
    shutil.copy2(src, dst)

def createDirectory(path):
    if not os.path.exists(path):
        os.makedirs(path)

def getImgPathArr(path):
    imgPathArr = [*glob.glob(os.path.join(path, f"**/2.Individuals/*.JPG"), recursive=True), *glob.glob(os.path.join(path, f"**/2.Individuals/*.jpg"), recursive=True)]
    imgPathArr = list(filter(lambda image_path: image_path.split('_')[-2] == '0' and 'CAM' not in image_path.split('_')[-1], imgPathArr))

    return imgPathArr

def loadImg(path):
    try:
        img = Image.open(path+"_0_01.JPG")
    except:
        img = Image.open(path+"_0_01.jpg")
    return np.array(img.resize((224, 224)))

def compareImg(img1, img2):
    return np.mean(cv2.absdiff(img1, img2))

def getSameImgPathSet(_imgPathArr):
    imgPathArr = list(map(lambda path: path.split(os.path.sep), _imgPathArr))
    imgNameArr = list(map(lambda path: path[-1].split('_'), imgPathArr))
    sameImgPathSet = set()

    for i in range(len(imgPathArr)):
        for j in range(i+1, len(imgPathArr)):
            if imgNameArr[i][2:5] == imgNameArr[j][2:5] and imgNameArr[i][0] != imgNameArr[j][0]:
                path1 = os.path.join((os.path.sep).join(imgPathArr[i][:-1]), '_'.join(imgNameArr[i][:-2]))
                path2 = os.path.join((os.path.sep).join(imgPathArr[j][:-1]), '_'.join(imgNameArr[j][:-2]))
                sameImgPathSet.add((path1, path2))
    
    return sameImgPathSet

def getExactSameImgPathPairArr(sameImgPathSet, threshold=40):
    sameImgPathPairArr = list(sameImgPathSet)
    exactSamePathPairArr = []
    for i in range(len(sameImgPathPairArr)):
        path1 = sameImgPathPairArr[i][0]
        path2 = sameImgPathPairArr[i][1]
        diff = compareImg(loadImg(path1), loadImg(path2))
        if diff < threshold:
            exactSamePathPairArr.append(sameImgPathPairArr[i])
    return exactSamePathPairArr

def extractImg(_path, threshold=40):

    dst = os.path.join(_path, 'extracted')
    createDirectory(dst)

    pathArr = getImgPathArr(_path)
    sameImgPathPairArr = getExactSameImgPathPairArr(getSameImgPathSet(pathArr), threshold)

    sameFamilyPairArr = list(map(lambda pair: list(map(lambda path: path.split(os.path.sep)[-1].split('_')[0], pair)), sameImgPathPairArr))

    sameFamilySet = set()

    for familyPair in sameFamilyPairArr:
            sameFamilySet.add(familyPair[1])
    
    print("Below groups will be excluded.")
    print(sameFamilySet)


    for path in pathArr:
        isSame = False
        for family in sameFamilySet:
            if family in path:
                isSame = True
        if not isSame:
            copyImg(path, dst)

if __name__ == '__main__':
    extractImg('./dataset/train')
    extractImg('./dataset/test')
