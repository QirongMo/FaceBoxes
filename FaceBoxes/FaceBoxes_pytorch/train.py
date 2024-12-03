
import os
import torch
from torch.utils.data import DataLoader

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
def train(model, train_loader, criterion, optimizer):
    model.train()
    total_loss = 0.
    for i,(images,loc_targets,conf_targets) in enumerate(train_loader):
        images = images.to(device)
        loc_targets = loc_targets.to(device)
        conf_targets = conf_targets.to(device)
        conf_preds, loc_preds = model(images)
        loss = criterion(loc_preds,loc_targets,conf_preds,conf_targets)
        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print ('\r  Iter [{}/{}] Loss: {:.4f}, average_loss: {:.4f}'.format(
            i+1, len(train_loader), loss.item(), total_loss / (i+1)), end='')
    print()

    return total_loss/len(train_loader)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--img_dir', type=str, default='。/FDDB/originalPics')
    parser.add_argument('--trainfile', type=str, default='。/FDDB/pffb.txt')
    parser.add_argument('--checkpoint_folder', type=str, default='./log')
    parser.add_argument('--save_freq', type=int, default=1)
    args = parser.parse_args(args=[])

    # step 1: model
    from networks import FaceBoxes
    model = FaceBoxes().to(device)

    # Step 2:  criterion, optimizer, scheduler
    from multibox_loss import MultiBoxLoss
    criterion = MultiBoxLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15,35,50,60,70], gamma=0.3)

    # step 3：data
    from dataset import MyDataset
    train_dataset = MyDataset(img_dir=args.img_dir, file=args.trainfile)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    #step 4： 创建记录器
    from visualdl import LogWriter
    log_writer = LogWriter(logdir=args.checkpoint_folder)

    # Step 5：run
    # print('load model...')
    # model.load_state_dict(torch.load('../working/faceboxes.pt'))
    for epoch in range(1, args.num_epochs + 1):
        print(f"Epoch {epoch}/{args.num_epochs}: ")
        train_loss = train(model, train_loader, criterion, optimizer)
        scheduler.step()
        log_writer.add_scalar(tag='loss', value= train_loss, step=epoch)
        if epoch % args.save_freq == 0 or epoch == args.num_epochs:
            print('saving model ...')
            torch.save(model.state_dict(), os.path.join(args.checkpoint_folder,'faceboxes.pt'))

if __name__ == '__main__':
    main()