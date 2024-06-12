import os
import argparse
import time

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from siamese import SiameseNetwork
from libs.dataset import Dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--train_path',
        type=str,
        help="Path to directory containing training dataset.",
        default="./dataset/train/extracted"
    )
    parser.add_argument(
        '--val_path',
        type=str,
        help="Path to directory containing validation dataset.",
        default="./dataset/test/extracted"
    )
    parser.add_argument(
        '-o',
        '--out_path',
        type=str,
        help="Path for outputting model weights and tensorboard summary.",
        default="./out"
    )
    parser.add_argument(
        '-b',
        '--backbone',
        type=str,
        help="Network backbone from torchvision.models to be used in the siamese network.",
        default="resnet18"
    )
    parser.add_argument(
        '-lr',
        '--learning_rate',
        type=float,
        help="Learning Rate",
        default=1e-4
    )
    parser.add_argument(
        '-bs',
        '--batch_size',
        type=int,
        help="Batch size to train",
        default=32
    )
    parser.add_argument(
        '-e',
        '--epochs',
        type=int,
        help="Number of epochs to train",
        default=1000
    )
    parser.add_argument(
        '-s',
        '--save_after',
        type=int,
        help="Model checkpoint is saved after each specified number of epochs.",
        default=5
    )
    parser.add_argument(
        '-l',
        '--load_model',
        type=int,
        help="Load pre-trained model of specified number of epochs",
        default=0
    )

    args = parser.parse_args()

    os.makedirs(args.out_path, exist_ok=True)

    # Set device to CUDA if a CUDA device is available, else CPU
    print('cuda: ', torch.cuda.is_available())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_dataset   = Dataset(args.train_path, shuffle_pairs=True, augment=True)
    val_dataset     = Dataset(args.val_path, shuffle_pairs=False, augment=False)
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, drop_last=True, num_workers=6)
    val_dataloader   = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=6)

    model = SiameseNetwork(backbone=args.backbone)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = torch.nn.BCELoss()

    writer = SummaryWriter(os.path.join(args.out_path, "summary"))

    initial_epoch = args.load_model

    if args.load_model != 0:
        pth = torch.load(os.path.join(args.out_path, f"./epoch_{args.load_model}.pth"))
        model.load_state_dict(pth['model_state_dict'])
        optimizer.load_state_dict(pth['optimizer_state_dict'])

    best_val = 10000000000

    for epoch in range(initial_epoch, args.epochs):
        print("Epoch [{} / {}]".format(epoch+1, args.epochs))
        model.train()

        t = time.time()

        losses = []
        correct = 0
        total = 0

        loopCnt = len(train_dataloader)
        # Training Loop Start
        for i, ((img1, img2), y, (class1, class2), (path1, path2)) in enumerate(train_dataloader):

            img1, img2, y = map(lambda x: x.to(device), [img1, img2, y])

            prob = model(img1, img2)
            loss = criterion(prob, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            correct += torch.count_nonzero(y == (prob > 0.5)).item()
            total += len(y)

            print(f"Batch [{i+1}/{loopCnt}] | Epoch [{epoch+1}/{args.epochs}] | dt: {time.time()-t}s")

            t = time.time()

            if i+1 >= loopCnt:
                break

        writer.add_scalar('train_loss', sum(losses)/len(losses), epoch)
        writer.add_scalar('train_acc', correct / total, epoch)

        print("\tTraining: Loss={:.2f}\t Accuracy={:.2f}\t".format(sum(losses)/len(losses), correct / total))
        # Training Loop End

        # Evaluation Loop Start
        model.eval()

        losses = []
        correct = 0
        total = 0

        for i, ((img1, img2), y, (class1, class2), (path1, path2)) in enumerate(val_dataloader):
            img1, img2, y = map(lambda x: x.to(device), [img1, img2, y])

            prob = model(img1, img2)
            loss = criterion(prob, y)

            losses.append(loss.item())
            correct += torch.count_nonzero(y == (prob > 0.5)).item()
            total += len(y)

            if i+1 >= len(val_dataloader):
                break

        val_loss = sum(losses)/max(1, len(losses))
        writer.add_scalar('val_loss', val_loss, epoch)
        writer.add_scalar('val_acc', correct / total, epoch)

        print("\tValidation: Loss={:.2f}\t Accuracy={:.2f}\t".format(val_loss, correct / total))
        # Evaluation Loop End

        # Update "best.pth" model if val_loss in current epoch is lower than the best validation loss
        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "backbone": args.backbone,
                    "optimizer_state_dict": optimizer.state_dict()
                },
                os.path.join(args.out_path, "best.pth")
            )            

        # Save model based on the frequency defined by "args.save_after"
        if (epoch + 1) % args.save_after == 0:
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "backbone": args.backbone,
                    "optimizer_state_dict": optimizer.state_dict()
                },
                os.path.join(args.out_path, "epoch_{}.pth".format(epoch + 1))
            )
