import torch
import argparse
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch import nn, optim
from torchvision.transforms import transforms
from unet import Unet
from dataset import LiverDataset
import cv2


# 是否使用cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# mask只需要转换为tensor
y_transforms = transforms.ToTensor()

def train_model(model, criterion, optimizer, dataload, num_epochs=12):
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        dt_size = len(dataload.dataset)
        epoch_loss = 0
        step = 0
        for x, y in dataload:
            step += 1
            inputs = x.to(device)
            labels = y.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            print("%d/%d,train_loss:%0.3f" % (step, (dt_size - 1) // dataload.batch_size + 1, loss.item()))
        print("epoch %d loss:%0.3f" % (epoch, epoch_loss/step))
    torch.save(model.state_dict(), 'inception_%d.pth' % epoch)
    return model

#训练模型
def train(args):
    model = Unet(3, 1).to(device)
    batch_size = args.batch_size
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters())
    liver_dataset = LiverDataset("data/train",transform=x_transforms,target_transform=y_transforms)
    dataloaders = DataLoader(liver_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    train_model(model, criterion, optimizer, dataloaders)

#显示模型的输出结果
def test(args):
    model = Unet(3, 1)
    model.load_state_dict(torch.load("weights_19.pth",map_location='cpu'))
    liver_dataset = LiverDataset("data/val", transform=x_transforms,target_transform=y_transforms)
    dataloaders = DataLoader(liver_dataset, batch_size=1)
    model.eval()
    plt.ion()
    with torch.no_grad():
        i=0
        for x, _ in dataloaders:
            i+=1
            y=model(x)
            img_y=torch.squeeze(y).numpy()
            img_y = img_y * 256
            thresh1, img_y = cv2.threshold(img_y, 127, 255, cv2.THRESH_BINARY)  //将图片二值化
            cv2.imshow('pred_img%03d'%(i-1),img_y)
            cv2.waitKey(1000)
            cv2.imwrite(r"data/pred/%03d_pred.png"%(i-1),img_y)
            cv2.destroyAllWindows()

if __name__ == '__main__':

    #参数解析
    # parse=argparse.ArgumentParser()
    parse = argparse.ArgumentParser()
    # parse.add_argument("action", type=str,default='train' ,help="train or test")
    parse.add_argument("--batch_size", type=int, default=1)         #batch_size=1，训练是每次传入1张图片
    parse.add_argument("--ckpt", type=str,  default=None, help="the path of model weight file")

    args = parse.parse_args()
    args.action = 'train'    #选择训练或者测试

    if args.action=="train":  #训练
        train(args)
    elif args.action=="test":#测试
        test(args)
