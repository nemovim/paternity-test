# Dataset
Make "dataset" directory at the root.
Download dataset in that directory from [AIHub](https://www.aihub.or.kr/aihubdata/data/list.do?currMenu=115&topMenu=100&searchKeyword=%EA%B0%80%EC%A1%B1%20%EA%B4%80%EA%B3%84%EA%B0%80%20%EC%95%8C%EB%A0%A4%EC%A7%84%20%EC%96%BC%EA%B5%B4%20%EC%9D%B4%EB%AF%B8%EC%A7%80%20%EB%8D%B0%EC%9D%B4%ED%84%B0).
Unzip all zip files.
Make "train" and "test" directories in the "dataset" directory.
Move all directories and files in `1.Training\원천데이터 to train` and `2.Validation\원천데이터 to test`.
Remove whitespace in the filename to avoid exception for below five files. (between '...F_45_' and '0_0...')
* TS0083\A(친가)\2.Individuals\F0083_IND_F_45_ 0_01.JPG
* TS0083\A(친가)\2.Individuals\F0083_IND_F_45_ 0_02.JPG
* TS0083\A(친가)\2.Individuals\F0083_IND_F_45_ 0_03.JPG
* TS0083\A(친가)\2.Individuals\F0083_IND_F_45_ 0_04.JPG
* TS0083\A(친가)\2.Individuals\F0083_IND_F_45_ 0_05.JPG

# Usage
For CMD with Windows.

1. Clone the git repository
```
git clone https://github.com/nemovim/paternity-test.git
cd paternity-test/
```

2. Construct a virtual environment and activate it
```
python -m venv .
cd Scripts/
activate
```

3. Install necessary modules
```
cd ../
pip install -r requirements.txt
```

4. Extract meaningful dataset
```
python extract_dataset.py
```

5. Start training
```
python train.py --train_path=./dataset/train/extracted --val_path=./dataset/test/extracted -o ./out
```