from cm_imports import * 
from BasicCNN import BasicCNN
from BasicDataset import MNIST_Data

mnist_trainset = datasets.MNIST(root='../data', train=True, download=False, transform=ToTensor())
mnist_testset = datasets.MNIST(root='../test', train=False, download=False, transform=ToTensor())

#Calculate mean and standard deviation
calcLoader = DataLoader(
    mnist_trainset,
    batch_size=64,
    shuffle=False
)

mean = 0.0
std = 0.0
length = 0
for images, _ in calcLoader:
    mean += images.mean()
    std += images.std()
    length += 1
mean /= length
std /= length
print(f'mean: {mean}\nstd: {std}')

transform = v2.Compose([
    v2.Normalize(mean=[mean.item()], std=[std.item()])
])

MNIST_trainset = MNIST_Data(mnist_trainset, transform=transform)
MNIST_testset = MNIST_Data(mnist_testset, transform=transform)

MNIST_train = MNIST_Data(MNIST_trainset, transform)
test_loader = DataLoader(
    MNIST_train,
    batch_size=64,
    num_workers=0,
    shuffle=True,
)

MNIST_test = MNIST_Data(MNIST_testset, transform)
train_loader = DataLoader(
    MNIST_test,
    batch_size=64,
    num_workers=0,
    shuffle=False,
)

#Declare the model, optimizer and loss function
clf = BasicCNN()
opt = torch.optim.Adam(clf.parameters(), lr=1e-2)
loss_fn = nn.CrossEntropyLoss()

#Trainning Loop
for epoch in range(10):
    run_loss = 0.0
    count = 0
    for X, y in train_loader:
        yhat = clf(X)
        loss = loss_fn(yhat, y)

        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        for X, y in test_loader:
            yhat = clf(X)
            loss = loss_fn(yhat, y)
            run_loss += loss.item()
            count+= 1
    print(f'Epoch: {epoch}, Loss:{run_loss/count}')