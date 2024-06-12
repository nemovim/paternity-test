import torch
from torch.utils.data import DataLoader
import pandas as pd
from tripleNet.tripleNetx import *
from libs.dataset_fam import Dataset
import time
import os

def createDirectory(path):
    if not os.path.exists(path):
        os.makedirs(path)

# Train function
def train(model, dataloader, test_dataloader, criterion, optimizer, num_epochs=10):
    data = {'epoch' : list(range(num_epochs)), 'acc1' : [], 'acc2' : [], 'loss' : []}

    best_loss = 999999
    
    model.train()

    print('[Start Training]')

    for epoch in range(num_epochs):        

        running_loss = 0.0
        correct1, correct2 = 0, 0
        total = 0

        loopCnt = len(dataloader)

        t = time.time()

        for i, ((img1, img2, img3), (label1, label2), classes) in enumerate(dataloader):


            img1, img2, img3 = img1.to(device), img2.to(device), img3.to(device)
            label1, label2 = label1.to(device), label2.to(device)
            
            optimizer.zero_grad()
            output = model(img1, img2, img3)
            loss = criterion(output, label1, label2)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * img1.size(0)
            
            # Accuracy calculation
            prob1, prob2 = output[:, 0], output[:, 1]
            predicted1 = (prob1 > 0.5).float()
            predicted2 = (prob2 > 0.5).float()

            correct1 += (predicted1 == label1.resize(img1.size(0))).sum().item()
            correct2 += (predicted2 == label2.resize(img1.size(0))).sum().item()
            total += label1.size(0)

            print(f"Batch [{i+1}/{loopCnt}] | Epoch [{epoch+1}/{num_epochs}] | dt: {time.time()-t}s")

            t = time.time()
        
        epoch_loss = running_loss / len(dataloader.dataset)
        data['loss'].append(epoch_loss)
        
        accuracy1 = correct1 / total
        data['acc1'].append(accuracy1)
        
        accuracy2 = correct2 / total
        data['acc2'].append(accuracy2)
        
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss}, Accuracy for Mother: {accuracy1 * 100}%, Accuracy for Father: {accuracy2 * 100}%')

        test(model, test_dataloader, epoch, num_epochs)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict()
                },
                f"./out_triple/best.pth"
            )            

        if epoch%5 == 0:
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict()
                },
                f"./out_triple/epoch_{epoch+1}.pth"
            )            
        
    df = pd.DataFrame(data)
    df.to_csv('./out_triple/train.csv', index=False)

def test(model, dataloader, epoch, num_epochs):
    data = {'epoch' : list(range(len(dataloader))), 'acc1' : [], 'acc2' : [], 'loss' : []}
    
    model.eval()
    running_loss = 0.0
    correct1, correct2 = 0, 0
    total = 0
    cnt = 1

    t = time.time()

    loopCnt = len(dataloader)
    
    with torch.no_grad():
        for i, ((img1, img2, img3), (label1, label2), classes) in enumerate(dataloader):
            img1, img2, img3 = img1.to(device), img2.to(device), img3.to(device)
            label1, label2 = label1.to(device), label2.to(device)
            
            output = model(img1, img2, img3)
            loss = criterion(output, label1, label2)
            running_loss += loss.item() * img1.size(0)
            
            prob1, prob2 = output[:, 0], output[:, 1]
            predicted1 = (prob1 > 0.5).float()
            predicted2 = (prob2 > 0.5).float()
            
            correct1 += (predicted1 == label1).sum().item()
            correct2 += (predicted2 == label2).sum().item()
            total += label1.size(0)
            
            data['acc1'].append(correct1 / total)
            data['acc2'].append(correct2 / total)
            data['loss'].append(running_loss / cnt)
            cnt += 1

            print(f"Batch [{i+1}/{loopCnt}] | Epoch [{epoch+1}/{num_epochs}] | dt: {time.time()-t}s")
    
    epoch_loss = running_loss / len(dataloader.dataset)
    accuracy1 = correct1 / total
    accuracy2 = correct2 / total
    print(f'Loss : {epoch_loss}, Accuracy for Mother: {accuracy1 * 100}%, Accuracy for Father: {accuracy2 * 100}%')
    
    df = pd.DataFrame(data)
    df.to_csv('./out_triple/test.csv', index=False)
    
if __name__ == '__main__':
    
    
    print('cuda: ', torch.cuda.is_available())
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    createDirectory('./out_triple')

    train_dataset = Dataset('./dataset/test/families', shuffle_pairs=True, augment=True)
    test_dataset = Dataset('./dataset/test/families', shuffle_pairs=True, augment=True)
    train_dataloader = DataLoader(train_dataset, batch_size=32)
    test_dataloader = DataLoader(test_dataset, batch_size=32)
    
    model = TripleSiameseNetwork(layer='conv').to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = ContrastiveLoss(margin=1.0)
    
    train(model, train_dataloader, test_dataloader, criterion, optimizer, num_epochs=10)
