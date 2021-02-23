
import Dataset 
import Module
import config
import torch

device = torch.device('cuda:0')

train, test = Dataset.train_test_split(config.TRAIN_PATH)
train_loader = torch.utils.data.DataLoader(dataset=train, batch_size=config.BATCH_SIZE, num_workers=4, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test, batch_size=config.BATCH_SIZE)

moudle = Module.CNN().to(device)
sgd = torch.optim.SGD(params=moudle.parameters(), lr=0.01, momentum=0.15)
loss_fun = torch.nn.functional.cross_entropy


def OutPutACC():
    with torch.no_grad():
        A = 0
        for batch in test_loader:
            X, y = batch
            X = X.to(device)
            y = y.to(device)
            pre = torch.argmax(moudle(X), dim=1)
            A += (pre == y).sum().item()
        print(A / len(test))
        


if __name__ == '__main__':
    for epoch in range(40):
        for i, batch in enumerate(train_loader):
            X, y = batch
            X = X.to(device)
            y = y.to(device)

            pre = moudle(X)
            loss = loss_fun(pre, y)

            sgd.zero_grad()
            loss.backward()
            sgd.step()
            if i % 10 == 0:
                print(i)
                print(loss)
        print('Out!')
        OutPutACC()