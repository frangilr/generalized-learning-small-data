import os
import sys

from tqdm import tqdm
import numpy as np
from torch import nn
import torch.optim as optim
import torch
import torch.nn.functional as F
from torchvision.transforms import v2
import timm
from torch.utils.data import DataLoader
from dataset_shapenet import CustomDataset

_seeds = [0, 10, 20]
_seed = _seeds[0]

torch.manual_seed(_seed)
torch.cuda.manual_seed_all(_seed)

log_key = {0: "zero", 1: "one", 2:"two", 3:"three", 4:"four", 5:"five", 6:"six", 7:"seven"}
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=str(_device)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

rootdir = "/dataset/dir/"
log_dir = f"/log/dir/TinyVit/100epochs/{log_key[_device]}"

num_classes = 6

def save_model(model, save_path):
    torch.save(model.state_dict(), save_path)

@torch.no_grad()
def eval_model(model, data_loader, device):
    correct = 0
    total = 0
    model.eval()

    assert not torch.is_grad_enabled(), "grad is enabled during inference"

    for data in data_loader:
        images, labels = data['img'], data['obj_cat']
        images, labels = images.to(device), labels.to(device)

        output_logits = model(images)

        total += labels.size(0)
        _, predicted = torch.max(output_logits.data, 1)
        correct += (predicted == labels).sum().item()

    assert not torch.is_grad_enabled(), "grad is enabled during inference"
    return 100 * correct / total

def main(_dataset):

    #________________________________________________________
    # choose train and test sets
    #
    train_csv = "train_lumpy.csv" if _dataset == "lumpy" else "train_nonLumpy.csv"
    test_csv = "test_shapenet.csv"

    epochs = 100
    val_freq = 1
    batch_size = 100

    lr_ = .0001

    print("\nInfo: ")
    print("\tTrain dataset: ", train_csv[:-4])
    print("\tTest dataset: ", test_csv[:-4])
    print(f"\tBatch size: {batch_size}")
    print("\tlr: ", lr_)

    my_dir = train_csv[:-4] + "_" + str(lr_)
    my_path = os.path.join(log_dir, my_dir)
    if not os.path.exists(my_path):
        os.makedirs(my_path)


    trainset = CustomDataset(csv_file=train_csv,
                            root_dir=rootdir,
                            train=True,
                            computing_mean=True
                            )

    print("\nPreparing datasets...")

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=4)

    my_mean = torch.zeros((3,), dtype=torch.float32)
    my_var = torch.zeros((3,), dtype=torch.float32)
    total_train_images = len(train_loader.dataset)
    total_train_batches = len(train_loader)

    print("total_train_images ", total_train_images)
    print("total_train_batches ", total_train_batches)
    
    # compute mean and sd
    for i, sample in tqdm(enumerate(train_loader), total=total_train_batches):

        my_mean += torch.sum(torch.mean(sample['img'], dim=(2,3)),dim=0)
        my_var += torch.sum(torch.var(sample['img'], dim=(2,3)),dim=0)

    my_mean /= total_train_images
    my_var  /= total_train_images
    my_std = torch.sqrt(my_var)

    del trainset, train_loader

    my_transforms = v2.Normalize(mean=my_mean, std=my_std)

    trainset = CustomDataset(csv_file=train_csv,
                            root_dir=rootdir,
                            train=True,
                            computing_mean=False,
                            normalize=my_transforms
                            )
    testset = CustomDataset(csv_file=test_csv,
                            root_dir=rootdir,
                            train=False,          
                            computing_mean=False,
                            normalize=my_transforms
                            )

    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)              
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)


    # initialize model
    model = timm.create_model('tiny_vit_11m_224', pretrained=False, num_classes=num_classes)
    optimizer = optim.SGD(model.parameters(), lr = lr_, weight_decay=0.001, momentum=0.9) 
    scheduler =  optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.9)
    criterion = nn.CrossEntropyLoss()

    model.to(device)
    all_test = []

    print("\nBegin Training...")
    best_acc = -1
    for epoch in range(epochs+1):
        running_loss = 0.0
        correct = 0

        model.train()
        
        # single epoch training
        for data in train_loader:
            images, labels = data['img'], data['obj_cat']
            images, labels = images.to(device), labels.to(device)
            
            output_logits = model(images)     

            loss = criterion(output_logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            with torch.no_grad():
                outputs = F.softmax(output_logits.detach(), dim=-1)
                _, predicted = torch.max(outputs.data, 1)

                correct += (predicted == labels).sum().item()

        scheduler.step()

        if (epoch % 10 == 0):
            print('[epoch] %2d [loss] %.3f [accuracy] %.3f' % (epoch, running_loss / len(train_loader), 100 * correct / len(train_loader.dataset)))
        if (epoch % val_freq == 0):
            save_path = os.path.join(my_path, f"e{epoch}.pth")

            with torch.no_grad():
                test_acc = eval_model(model, test_loader, device)
                save_model(model=model, save_path=save_path)
                if test_acc > best_acc:
                    best_acc = test_acc

            all_test.append(test_acc)

    print("\nFinished training.")

    print()
    print("Test metrics:")
    print("Best acc: ", max(all_test))
    print("\tEpoch: ", all_test.index(max(all_test)))
    
    # for i in range(epochs+1):
    #     if i != all_test.index(max(all_test)):
    #         try:
    #             os.remove(os.path.join(my_path, f"e{i}.pth"))
    #         except:
    #             pass

    print("\nInfo: ")
    print("\tTrain dataset: ", train_csv[:-4])
    print("\tVal dataset: ", test_csv[:-4])
    print(f"\tBatch size: {batch_size}")
    print("\tlr: ", lr_)

    return max(all_test)

if __name__ == "__main__":
    
    assert len(sys.argv) == 2 and sys.argv[1] in ["lumpy", "nonLumpy"], "Correct usage: python script.py [lumpy or nonLumpy]"
    _dataset = sys.argv[1]

    accs = []
    for i in range(1):
        acc = main(_dataset=_dataset)
        accs.append(acc)
    
    print(accs)
    print(np.mean(accs), '+-', np.std(accs))