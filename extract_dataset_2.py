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
                if (path1, path2) in sameImgPathSet or (path2, path1) in sameImgPathSet:
                    continue
                sameImgPathSet.add((path1, path2))
    
    return sameImgPathSet

def getPossiblePathPairArr(imgPathArr, sameImgPathSet):
    sameImgPathPairArr = list(sameImgPathSet)
    possiblePathPairArr = []
    for i, sameImgPathPair in enumerate(sameImgPathPairArr):
        print(f'Checking... {i}/{len(sameImgPathPairArr)}')
        rel = sameImgPathPair[0].split('_')[-2]
        if rel == 'GM' or rel == 'GF':
            continue
        family1 = sameImgPathPair[0].split(os.path.sep)[-1].split('_')[0]
        family2 = sameImgPathPair[1].split(os.path.sep)[-1].split('_')[0]
        family1RelSet = set(map(lambda path: path.split('_')[-4], filter(lambda path: family1 in path, imgPathArr)))
        family2RelSet = set(map(lambda path: path.split('_')[-4], filter(lambda path: family2 in path, imgPathArr)))

        if ('GF' in family1RelSet and 'GM' in family2RelSet) or ('GM' in family1RelSet and 'GF' in family2RelSet):
            if rel == 'M' or rel == 'F':
                possiblePathPairArr.append((*sameImgPathPair, rel))
        elif ('F' in family1RelSet and 'M' in family2RelSet) or ('M' in family1RelSet and 'F' in family2RelSet):
            if rel == 'S' or rel == 'D' or rel == 'S2' or rel == 'D2' or rel == 'S3' or rel == 'D3' or rel == 'S4' or rel == 'D4':
            # if rel == 'S' or rel == 'D':
                possiblePathPairArr.append((*sameImgPathPair, rel))
    
    return possiblePathPairArr

def getExactPossiblePathPairArr(possiblePathPairArr, threshold=40):
    exactSamePathPairArr = []
    for i in range(len(possiblePathPairArr)):
        print(f'Matching... {i}/{len(possiblePathPairArr)}')
        path1 = possiblePathPairArr[i][0]
        path2 = possiblePathPairArr[i][1]
        diff = compareImg(loadImg(path1), loadImg(path2))
        if diff < threshold:
            exactSamePathPairArr.append(possiblePathPairArr[i])
    return exactSamePathPairArr

def extractImg(_path, threshold=40):


    pathArr = getImgPathArr(_path)
    possiblePathPairArr = getExactPossiblePathPairArr(getPossiblePathPairArr(pathArr, getSameImgPathSet(pathArr)), threshold)

    possibleFamilyPairArr = list(map(lambda pair: list(map(lambda path: path.split(os.path.sep)[-1].split('_')[0], pair)), possiblePathPairArr))

    print('Below gropus will be combined')
    print(possibleFamilyPairArr)

    for i, possibleFamilyPair in enumerate(possibleFamilyPairArr):
        family1 = possibleFamilyPair[0]
        family2 = possibleFamilyPair[1]
        sameRel = possibleFamilyPair[2]

        dst = os.path.join(_path, os.path.join('families', str(i)))
        createDirectory(dst)

        for path in pathArr:
            if family1 not in path and family2 not in path:
                continue
            rel = path.split('_')[-4]
            cnt = path.split('_')[-1]

            if sameRel == 'M' or sameRel == 'F':
                if family1 in path:
                    if rel == sameRel:
                        copyImg(path, os.path.join(dst, f'C_{cnt}'))
                    elif rel == 'GM':
                        copyImg(path, os.path.join(dst, f'M_{cnt}'))
                    elif rel == 'GF':
                        copyImg(path, os.path.join(dst, f'F_{cnt}'))
                elif family2 in path:
                    if rel == 'GM':
                        copyImg(path, os.path.join(dst, f'M_{cnt}'))
                    elif rel == 'GF':
                        copyImg(path, os.path.join(dst, f'F_{cnt}'))
            else:
                if family1 in path:
                    if rel == sameRel:
                        copyImg(path, os.path.join(dst, f'C_{cnt}'))
                    elif rel == 'M' or rel == 'F':
                        copyImg(path, os.path.join(dst, f'{rel}_{cnt}'))
                if family2 in path and (rel == 'M' or rel == 'F'):
                    copyImg(path, os.path.join(dst, f'{rel}_{cnt}'))

if __name__ == '__main__':
    print('[Train]')
    extractImg('./dataset/train')
    print('[Test]')
    extractImg('./dataset/test')
