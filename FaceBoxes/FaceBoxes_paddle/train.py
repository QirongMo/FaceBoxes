
import os
import paddle
from paddle.io import DataLoader
from model import FaceBoxes
from multibox_loss import MultiBoxLoss

def train(model, train_loader, criterion, optimizer):
    total_loss = 0.
    total_loc_loss = 0.
    total_conf_loss = 0.
    step_epoch = len(train_loader)
    for i,(img, tg_obj, tg_boxes) in enumerate(train_loader):
        model.train()
        # 梯度清零
        optimizer.clear_grad()
        model.clear_gradients()
        img = img.astype('float32').transpose((0,3,1,2))
        tg_obj = tg_obj.astype("int64")
        tg_boxes = tg_boxes.astype('float32')
        loc_preds, conf_preds = model(img)
        loc_loss, conf_loss, loss = criterion(loc_preds, tg_boxes, conf_preds, tg_obj)
        total_loss += loss.numpy()[0]
        total_loc_loss += loc_loss.numpy()[0]
        total_conf_loss += conf_loss.numpy()[0]
        # 反向传播
        loss.backward()
        # 更新参数
        optimizer.step()

        print ('\r  Iter [{}/{}] Loss: {:.4f}, average_loss: {:.4f}, avg_loc_loss: {:.4f}, avg_conf_loss: {:.4f}'.format( \
            i+1, step_epoch, loss.numpy()[0], total_loss / (i+1), \
            total_loc_loss / (i+1),   total_conf_loss / (i+1)), end='')
    print()
    return total_loss/step_epoch, total_loc_loss/step_epoch, total_conf_loss/step_epoch

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--img_dir', type=str, default='./FDDB/originalPics/')
    parser.add_argument('--trainfile', type=str, default='./FDDB/FDDB.txt')
    parser.add_argument('--checkpoint_folder', type=str, default='./log')
    parser.add_argument('--save_freq', type=int, default=1)
    args = parser.parse_args(args=[])

    # step 1: model
    model = FaceBoxes()
    model_dict = paddle.load('./log/FaceBoxes.pdparams')
    model.set_state_dict(model_dict)
    # Step 2:  criterion, optimizer, scheduler
    criterion = MultiBoxLoss()
    scheduler = paddle.optimizer.lr.ReduceOnPlateau(learning_rate=args.lr, factor=0.5, patience=3, verbose=True)
    # optimizer = paddle.optimizer.SGD(learning_rate=scheduler, parameters=model.parameters(), weight_decay=1e-4)
    optimizer = paddle.optimizer.Adam(learning_rate=scheduler,parameters=model.parameters(),weight_decay=1e-4)

    # step 3：data
    from FDDB import FDDB
    train_dataset = FDDB(img_dir=args.img_dir, file=args.trainfile)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)

    #step 4： 创建记录器
    from visualdl import LogWriter
    log_writer = LogWriter(logdir=args.checkpoint_folder)

    # Step 5：run
    for epoch in range(101, args.num_epochs + 1):
        print(f"Epoch {epoch}/{args.num_epochs}: ")
        total_loss, total_loc_loss, total_conf_loss = train(model, train_loader, criterion, optimizer)
        scheduler.step(total_loss)
        log_writer.add_scalar(tag='loss', value= total_loss, step=epoch)
        log_writer.add_scalar(tag='loc_loss', value= total_loc_loss, step=epoch)
        log_writer.add_scalar(tag='conf_loss', value= total_conf_loss, step=epoch)
        if epoch % args.save_freq == 0 or epoch == args.num_epochs:
            print('saving model ...')
            model_path = os.path.join(args.checkpoint_folder, "FaceBoxes")
            model.eval()
            model_dict = model.state_dict()
            paddle.save(model_dict, model_path+'.pdparams')

if __name__ == '__main__':
    main()