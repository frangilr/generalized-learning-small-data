import pandas as pd
from torch import nn
import torch.optim as optim
import torch
import torch.nn.functional as F
import os
from torchvision.transforms import v2
from torchvision.models import resnet18
from torch.utils.data import DataLoader
from dataset_petface import CustomDataset

dev = 0

_seeds = [0, 10, 20]
_seed = _seeds[dev]

log_key = {0: "zero", 1: "one", 2:"two", 3:"three"}
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=str(dev)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(_seed)
torch.cuda.manual_seed_all(_seed)

rootdir = "/your/path/datasets/PetFace/images"

num_classes = 6
lumpy = True

fold = "lumpy" if lumpy else "nonLumpy"

log_dir = f"/your/path/logs/{fold}/{log_key[dev]}"

def save_model(model, save_path):
    torch.save(model.state_dict(), save_path)

@torch.no_grad()
def eval_model(model, data_loader, device):
    # Evaluate the model on data from valloader
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

def main():
    
    #________________________________________________________
    # choose train and test sets
    # 
    train_csv = "train_lumpy_petface.csv" if lumpy else "train_nonLumpy_petface.csv"
    test_csv = "test_petface.csv"

    epochs = 100
    val_freq = 1
    batch_size = 100
    train_size = None
    val_size = None

    lr_ = .001

    print("\nInfo: ")
    print("\tTrain dataset: ", train_csv[:-4])
    print("\tSize restriction: ", train_size) # not used
    print("\tTest dataset: ", test_csv[:-4])
    print("\tSize restriction: ", val_size)
    print(f"\tBatch size: {batch_size}")
    print("\tlr: ", lr_)
    print("\tLumpy: ", lumpy)

    my_dir = train_csv[:-4] + "_" + str(lr_)
    my_path = os.path.join(log_dir, my_dir)
    os.makedirs(my_path, exist_ok=True)

    print("\nPreparing datasets...")

    my_transforms = v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


    trainset = CustomDataset(csv_file=train_csv,
                                root_dir=rootdir,
                                train=True,
                                computing_mean=False,
                                size=train_size,
                                normalize=my_transforms)
    testset = CustomDataset(csv_file=test_csv,
                                root_dir=rootdir,
                                train=False,          
                                computing_mean=False,
                                size=val_size,
                                normalize=my_transforms)
    
    
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    total_train_images = len(train_loader.dataset)
    total_train_batches = len(train_loader)

    print("total_train_images ", total_train_images)
    print("total_train_batches ", total_train_batches)

    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)


    # initialize model
    model = resnet18()
    # model.fc = nn.Linear(2048, num_classes) # for resnet50
    model.fc = nn.Linear(512, num_classes)
    optimizer = optim.SGD(model.parameters(), lr = lr_, weight_decay=0.001, momentum=0.9) 
    scheduler =  optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.9)
    criterion = nn.CrossEntropyLoss()

    model.to(device)
    all_test = []

    print("Begin Training...")
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

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            running_loss += loss.item()
            
            with torch.no_grad():
                outputs = F.softmax(output_logits.detach(), dim=-1)
                _, predicted = torch.max(outputs.data, 1)

                correct += (predicted == labels).sum().item()

        scheduler.step()

        if (epoch % 10 == 0):
            print('[train] epoch - %d loss: %.3f accuracy: %.3f' % (epoch, running_loss / len(train_loader), 100 * correct / len(train_loader.dataset)))
        if (epoch % val_freq == 0):
            save_path = os.path.join(my_path, f"ce_resnet18_{epoch}.pth")

            with torch.no_grad():
                test_acc = eval_model(model, test_loader, device)
                print(f"[test] acc: {test_acc}")
                if test_acc > best_acc:
                    save_model(model=model, save_path=save_path)
                    best_acc = test_acc

            all_test.append(test_acc)


    print("\nFinished training.")

    print()
    print("Test metrics:")
    print("Best acc: ", max(all_test))
    print("\tEpoch: ", all_test.index(max(all_test)))
    
    for i in range(epochs+1):
        if i != all_test.index(max(all_test)):
            try:
                os.remove(os.path.join(my_path, f"ce_resnet18_{i}.pth"))
            except:
                pass

    print("\nInfo: ")
    print("\tTrain dataset: ", train_csv[:-4])
    print("\tSize restriction: ", train_size)
    print("\tVal dataset: ", test_csv[:-4])
    print("\tSize restriction: ", val_size)
    print(f"\tBatch size: {batch_size}")
    print("\tlr: ", lr_)
    print("\tLumpy: ", lumpy)

if __name__ == "__main__":
    main()
