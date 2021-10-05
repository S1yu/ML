import torch
import torch.nn as nn
import torchvision


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet,self).__init__()
        self.conv=nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=6,kernel_size=5,padding=2),
            nn.Sigmoid(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Conv2d(6,out_channels=16,kernel_size=5,),
            nn.Sigmoid(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.fc=nn.Sequential(
            nn.Linear(in_features=16*5*5,out_features=120,),
            nn.Sigmoid(),
            nn.Linear(120,84),
            nn.Sigmoid(),
            nn.Linear(84, 10),
        )

    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0], -1))
        return output






def load_data_fashion_mnist(batch_size):

    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    mnist_train = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST', train=True, download=False,
                                                    transform=transform)
    mnist_test = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST', train=False, download=False,
                                                   transform=transform)

    train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=0)
    test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_iter, test_iter


def evaluate_accuracy(data_iter, net, device=None):
    if device is None and isinstance(net, torch.nn.Module):
        device = list(net.parameters())[0].device
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(net, torch.nn.Module):
                # set the model to evaluation mode (disable dropout)
                net.eval()
                # get the acc of this batch
                acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
                # change back to train mode
                net.train()

            n += y.shape[0]
    return acc_sum / n


def train(net, train_iter, test_iter, optimizer, num_epochs):
    net = net.to(torch.device("cpu"))
    loss = nn.CrossEntropyLoss()
    batch_count = 0
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, Y in train_iter:
            X = X.to(torch.device("cpu"))
            Y = Y.to(torch.device("cpu"))
            y_hat = net(X)
            l = loss(y_hat, Y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == Y).sum().cpu().item()
            n += Y.shape[0]
            batch_count += 1
        test_acc = evaluate_accuracy(test_iter, net)
        print(
            f'epoch {epoch + 1} : loss {train_l_sum / batch_count:.3f}, train acc {train_acc_sum / n:.3f}, test acc {test_acc:.3f}')


if __name__ == '__main__':
    bathc_size = 256
    lr, num_epochs = 0.01, 10
    net = LeNet()
    opt = torch.optim.SGD(net.parameters(), lr=lr)

    train_iter, test_iter = load_data_fashion_mnist(batch_size=bathc_size)
    # train
    train(net, train_iter, test_iter, opt, num_epochs)
