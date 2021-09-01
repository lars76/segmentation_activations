import torch
import random
import os
import argparse
import numpy as np
import glob
import albumentations as A
import pandas as pd
from sklearn.model_selection import KFold
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_preprocessing_fn
from torch.utils.data import Dataset as BaseDataset
from torch.nn import BCELoss, MSELoss, BCEWithLogitsLoss
from torch.nn import functional as F
import torch.optim as optim
from tqdm import tqdm

STOP = 10
LR_REDUCE = 5
BATCH_SIZE = 8
EPSILON = 1e-7

def seed_all(seed):
    torch.manual_seed(seed)
    # might not be needed
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

### Initial seed
seed_all(0)
###

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

@torch.jit.script
def softsign_with_logits(y_hat : torch.Tensor, y_true : torch.Tensor, epsilon : float) -> torch.Tensor:
    z = 1 + torch.abs(y_hat)
    output1 = torch.log(z + y_hat)
    output2 = torch.log(z - y_hat)

    return torch.mean(-torch.log(0.5) - y_true * output1 - (1 - y_true) * output2 + torch.log(z))

@torch.jit.script
def inv_square_with_logits(y_hat : torch.Tensor, y_true : torch.Tensor, epsilon : float) -> torch.Tensor:
    z = 1 + y_hat ** 2
    z_sqrt = torch.sqrt(z)
    output1 = torch.log(z + z_sqrt)
    output2 = torch.log(z - y_hat * z_sqrt)
    return torch.mean(-torch.log(0.5) - y_true * output1 - (1 - y_true) * output2 + torch.log(z))

@torch.jit.script
def bce_with_logits(y_hat : torch.Tensor, y_true : torch.Tensor, epsilon : float) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(y_hat, y_true)

@torch.jit.script
def arctan_activation(x : torch.Tensor, epsilon : float) -> torch.Tensor:
    return epsilon + (1 - 2 * epsilon) * (0.5 + torch.arctan(x)/torch.tensor(np.pi))

@torch.jit.script
def softsign_activation(x : torch.Tensor, epsilon : float) -> torch.Tensor:
    return (0.5 - epsilon) * F.softsign(x) + 0.5

@torch.jit.script
def sigmoid_activation(x : torch.Tensor, epsilon : float) -> torch.Tensor:
    return torch.sigmoid(x)

@torch.jit.script
def linear_activation(x : torch.Tensor, epsilon : float) -> torch.Tensor:
    return epsilon + (1 - 2 * epsilon) * (x - x.min())/(x.max() - x.min())

@torch.jit.script
def inv_square_root_activation(x : torch.Tensor, epsilon : float) -> torch.Tensor:
    return (0.5 - epsilon) * x * torch.rsqrt(1 + x ** 2) + 0.5

@torch.jit.script
def cdf_activation(x : torch.Tensor, epsilon : float) -> torch.Tensor:
    # https://github.com/IraKorshunova/pytorch/blob/master/torch/autograd/_functions/pointwise.py#L274
    # https://github.com/IraKorshunova/pytorch/blob/master/torch/lib/THC/THCNumerics.cuh#L441
    # https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__SINGLE.html#group__CUDA__MATH__SINGLE_1g3b8115ff34a107f4608152fd943dbf81
    return (0.5 - epsilon) * torch.erf(x/torch.sqrt(torch.tensor(2))) + 0.5

@torch.jit.script
def hardtanh_activation(x : torch.Tensor, epsilon : float) -> torch.Tensor:
    return F.hardtanh(x, epsilon, 1.0 - epsilon)

class DiceLoss():
    def __init__(self):
        pass
    def __call__(self, y_pred, y_true):
        numerator = (y_pred * y_true).sum()
        denominator = y_pred.sum() + y_true.sum()
        return 1 - (2 * numerator) / denominator

def metrics(y_true, y_pred, epoch):
    row = {"epoch":epoch, "nll":0, "avg_total_dice":0}

    y_pred = np.clip(y_pred, EPSILON, 1-EPSILON)
    div = 0
    nll = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    for class_ in range(y_pred.shape[1]):
        nll_class = nll[:, class_]
        y_true_class = y_true[:, class_]
        y_pred_class = y_pred[:, class_]

        row[f"nll_class_{class_}"] = np.sum(nll_class) / y_pred_class.shape[0]
        row["nll"] += row[f"nll_class_{class_}"]
        row[f"best_dice_class_{class_}"] = 0

        for th in [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3,
                   0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65,
                   0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]:
            tp_mask = (y_true_class == 1) & (y_pred_class >= th)
            fp_mask = (y_true_class == 0) & (y_pred_class >= th)
            tn_mask = (y_true_class == 0) & (y_pred_class < th)
            fn_mask = (y_true_class == 1) & (y_pred_class < th)

            tp = np.sum(nll_class[tp_mask]) / y_pred_class.shape[0]
            fp = np.sum(nll_class[fp_mask]) / y_pred_class.shape[0]
            tn = np.sum(nll_class[tn_mask]) / y_pred_class.shape[0]
            fn = np.sum(nll_class[fn_mask]) / y_pred_class.shape[0]

            pred_masked = (y_pred_class > th).astype(np.float32)
            numerator = (pred_masked * y_true_class).sum()
            denominator = pred_masked.sum() + y_true_class.sum()
            dice = (2 * numerator) / denominator
            row[f"best_dice_class_{class_}"] = max(dice, row[f"best_dice_class_{class_}"])

            numerator2 = ((1 - pred_masked) * (1 - y_true_class)).sum()
            denominator2 = (1 - pred_masked).sum() + (1 - y_true_class).sum()
            neg_dice = (2 * numerator2) / denominator2

            row[f"dice_{th}_class_{class_}"] = dice
            row[f"neg_dice_{th}_class_{class_}"] = neg_dice
            row[f"tp_{th}_class_{class_}"] = tp
            row[f"fp_{th}_class_{class_}"] = fp
            row[f"tn_{th}_class_{class_}"] = tn
            row[f"fn_{th}_class_{class_}"] = fn
            row["avg_total_dice"] += dice
            div += 1
    
    row["avg_total_dice"] /= div

    return row

def train(args):
    loss_func = globals()[args.loss]
    activation_func = globals()[args.activation]
    feature_extractor = args.encoder
    architecture = getattr(smp, args.model)

    data_path = args.path
    image_size = args.image_size
    classes = args.classes
    save_only_best = args.save_only_best
    channels = args.channels

    numerical_stable_loss = False
    if args.loss == "BCELoss":
        if args.activation == "sigmoid_activation":
            print("Using bce_with_logits (numerical stable)")
            numerical_stable_loss = True
            loss_func = bce_with_logits
        elif args.activation == "softsign_activation":
            print("Using softsign_with_logits (numerical stable)")
            loss_func = softsign_with_logits
            numerical_stable_loss = True
        elif args.activation == "inv_square_root_activation":
            print("Using inv_square_with_logits (numerical stable)")
            loss_func = inv_square_with_logits
            numerical_stable_loss = True
        else:
            print("Using BCELoss (not necessarily numerical stable)")
            loss_func = BCELoss()
    else:
        loss_func = loss_func()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    preprocess_input = get_preprocessing_fn(feature_extractor,
                                            pretrained='imagenet')
    loaders = create_loaders(preprocess_input, data_path, image_size, classes, BATCH_SIZE)

    base_path = f"{args.prefix}-{args.loss}-{args.activation}-{args.encoder}-{args.model}"

    criterion = loss_func
    dice_total_all_folds = []
    dice_all_folds = []
    nll_all_folds = []
    for fold, (train_loader, val_loader) in enumerate(loaders):
        path = os.path.join(base_path, f"fold-{fold}")
        if not os.path.exists(path):
            os.makedirs(path)

        print(f"\nFold: {fold}")

        seed_all(fold)

        model = architecture(feature_extractor, classes=classes,
                      encoder_weights="imagenet",
                      in_channels=channels)

        model = model.to(device)
        optimizer = optim.Adam(model.parameters())
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                         patience=LR_REDUCE,
                                                         verbose=True)
        best_dice_total = -1e9
        best_dice = -1e9
        best_nll = 1e9

        stop = 0
        epoch = 0
        progress = []
        while True:
            print(f"Epoch: {epoch}")
            model.train()
            for inputs, targets in tqdm(train_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                
                if numerical_stable_loss:
                    outputs = model(inputs)
                    loss = criterion(outputs, targets, EPSILON)
                else:
                    outputs = activation_func(model(inputs), EPSILON)
                    loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                if loss.item() < 0 or torch.isnan(loss):
                    print(f"ERROR Loss is {loss}")

            model.eval()

            val_target = np.zeros((len(val_loader.dataset) * image_size ** 2, classes))
            val_pred = np.zeros((len(val_loader.dataset) * image_size ** 2, classes))
            with torch.no_grad():
                j = 0
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    flattened = activation_func(model(inputs), EPSILON).view(-1, classes).cpu().numpy()
                    val_pred[j : j + flattened.shape[0]] = flattened
                    val_target[j : j + flattened.shape[0]] = targets.view(-1, classes).cpu().numpy()
                    j += flattened.shape[0]

            row = metrics(val_target, val_pred, epoch)

            avg_dice = 0
            for class_ in range(val_pred.shape[1]):
                if val_pred.shape[1] > 1:
                    print(f"Best dice class {class_}:", row[f"best_dice_class_{class_}"])
                avg_dice += row[f"best_dice_class_{class_}"]
            avg_dice /= val_pred.shape[1]
            print(f"Avg dice (best threshold): {avg_dice}")
            print(f"Avg dice (all thresholds): {row['avg_total_dice']}")
            print(f"NLL: {row['nll']}")

            progress.append(row)

            best_dice = max(avg_dice, best_dice)
            best_nll = min(row["nll"], best_nll)

            improvement = False
            if row["avg_total_dice"] > best_dice_total:
                improvement = True
                best_dice_total = row["avg_total_dice"]
                stop = 0
                print(f"---> New best total dice {best_dice_total}")
            else:
                stop += 1
            
            state = {
                "state_dict": model.state_dict()
            }
            try:
                state |= row
            except:
                print("Warning: Python < 3.9, no support for joining dicts by '|'")

            if save_only_best and improvement:
                p = os.path.join(path, 'best.pt')
                torch.save(state, p)
                print(f"Saved model to {p}")
            elif not save_only_best:
                p = os.path.join(path, f'epoch-{epoch}.pt')
                torch.save(state, p)
                print(f"Saved model to {p}")

            if stop > STOP:
                print("\n------\n")
                print(f"Fold best avg total dice: {best_dice_total}")
                print(f"Fold best dice: {best_dice}")
                print(f"Fold best nll: {best_nll}\n")
                p = os.path.join(path, f"fold-{fold}-log.csv")
                pd.DataFrame(progress).to_csv(p, index=False)
                print(f"Saved training log to {p}")
                dice_total_all_folds.append(best_dice_total)
                nll_all_folds.append(best_nll)
                dice_all_folds.append(best_dice)
                break

            epoch += 1

            scheduler.step(1 - row['avg_total_dice'])

    print(f"\nAvg dice (all folds): {dice_all_folds}")
    print(f"Avg total dice (all folds): {dice_total_all_folds}")
    print(f"Avg nll: {nll_all_folds}\n")

    print(f"Avg dice (all folds): {np.mean(dice_all_folds)} (+- {np.std(dice_all_folds)})")
    print(f"Avg total dice (all folds): {np.mean(dice_total_all_folds)} (+- {np.std(dice_total_all_folds)})")
    print(f"Avg nll: {np.mean(nll_all_folds)} (+- {np.std(nll_all_folds)})")

def create_loaders(preprocessing, data_path, image_size, classes, batch_size):
    # sort to make sure that files are always in the same order
    folders = np.array(sorted(glob.glob(os.path.join(data_path, "*"))))
    loaders = []
    for fold, (train_index, test_index) in enumerate(KFold(n_splits=5).split(range(len(folders)))):
        train_X, val_X = folders[train_index], folders[test_index]
        #print(f"Fold {fold}")
        #print(f"Patients (train): {train_X}")
        #print(f"Patients (val): {val_X}")
        # optionally enable augmentations
        augmentation = A.Compose([])#A.HorizontalFlip(p=0.5),
                                  #A.VerticalFlip(p=0.5),
                                  #A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=30, p=0.5),
                                  #A.RandomCrop(image_size, image_size, p=0.5),
                                  #A.Cutout(num_holes=2, max_h_size=32, max_w_size=32, fill_value=0, p=0.5)])
        train_dataset = Dataset(train_X, preprocessing=preprocessing, augmentation=augmentation,
                                classes=classes)

        g = torch.Generator()
        g.manual_seed(fold)

        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   pin_memory=torch.cuda.is_available(),
                                                   shuffle=True,
                                                   batch_size=batch_size,
                                                   num_workers=1,
                                                   worker_init_fn=seed_worker,
                                                   generator=g)

        val_dataset = Dataset(val_X, preprocessing=preprocessing, augmentation=A.Compose([]),
                              classes=classes)
        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                 pin_memory=torch.cuda.is_available(),
                                                 shuffle=False,
                                                 batch_size=batch_size,
                                                 num_workers=1,
                                                 worker_init_fn=seed_worker,
                                                 generator=g)
        loaders.append((train_loader, val_loader))

    return loaders

class Dataset(BaseDataset):
    def __init__(self, input_images, preprocessing, augmentation, classes):
        self.input_images = []
        for patient in input_images:
            self.input_images.extend(glob.glob(os.path.join(patient, "processed_*.npy")))
        self.augmentation = augmentation
        self.preprocess_fn = preprocessing
        self.classes = classes

    def __getitem__(self, i):
        files = self.input_images[i]
        loaded_npy = np.load(self.input_images[i])

        image = loaded_npy[...,:-1]

        if self.classes > 1:
            mask = np.zeros((image.shape[0], image.shape[1], self.classes))
            for i in range(1, self.classes+1):
                mask[...,i-1] = loaded_npy[...,-1] == i
        else:
            mask = loaded_npy[...,-1:]

        sample = self.augmentation(image=image, mask=mask)
        image, mask = sample['image'], sample['mask']

        if image.shape[-1] >= 3:
            image[...,:3] = self.preprocess_fn(image[...,:3])
            if image.shape[-1] == 6:
                image[...,3:] = self.preprocess_fn(image[...,3:])

        image = image.transpose(2, 0, 1)
        mask = mask.transpose(2, 0, 1)

        return image.astype('float32'), mask.astype('float32')
        
    def __len__(self):
        return len(self.input_images)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('activation', type=str, help="arctan_activation, softsign_activation, sigmoid_activation, linear_activation, inv_square_root_activation, cdf_activation, hardtanh_activation")
    parser.add_argument('model', type=str, help="Unet") ######## actually the decoder, FIXME
    parser.add_argument('encoder', type=str, help="resnet34", default="resnet34", nargs='?')
    parser.add_argument('loss', type=str, help="BCELoss, MSELoss, DiceLoss")
    parser.add_argument('path', type=str, help="data path")
    parser.add_argument('image_size', type=int, help="image size", nargs='?')
    parser.add_argument('classes', type=int, help="classes", nargs='?')
    parser.add_argument('save_only_best', type=bool, help="save_only_best", default=True, nargs='?')
    parser.add_argument('channels', type=int, help="channels", nargs='?')
    parser.add_argument('prefix', type=int, help="prefix when saving files", nargs='?')

    args = parser.parse_args()
    print("Parameters", args)

    if "ACAC" in args.path:
        if args.image_size is None:
            args.image_size = 128
        if args.classes is None:
            args.classes = 3
        if args.channels is None:
            args.channels = 1
        if args.prefix is None:
            args.prefix = "ACAC"
    elif "ISLES" in args.path:
        if args.image_size is None:
            args.image_size = 256
        if args.classes is None:
            args.classes = 1
        if args.channels is None:
            args.channels = 7
        if args.prefix is None:
            args.prefix = "ISLES"
    elif "Kvasir" in args.path:
        if args.image_size is None:
            args.image_size = 256
        if args.classes is None:
            args.classes = 1
        if args.channels is None:
            args.channels = 3
        if args.prefix is None:
            args.prefix = "Kvasir"
    elif "MSD" in args.path:
        if args.image_size is None:
            args.image_size = 256
        if args.classes is None:
            args.classes = 2
        if args.channels is None:
            args.channels = 2
        if args.prefix is None:
            args.prefix = "MSD"

    train(args)