import torch

import datasets.rop_4cls as rop_4cls
from datasets import rop_3cls_combine_1_2 as rop_3cls_combine_1_2
from datasets import rop_2cls as rop_2cls
from datasets import rop_2cls_balance as rop_2cls_balance
from datasets import validation_2cls as val_2cls

def getDataloaders(dataset_name, splits, batch_size, num_workers=3, transform=None,
                   target_transform=None):
    train_loader, val_loader, test_loader = None, None, None

    if dataset_name == 'rop_4cls':
        if 'train' in splits:
            train_set = rop_4cls.rop_4cls('train', transform=transform, target_transform=target_transform)
            train_loader = torch.torch.utils.data.DataLoader(dataset=train_set,
                                                             batch_size=batch_size,
                                                             shuffle=True,
                                                             num_workers=num_workers)
        if 'val' in splits or 'test' in splits:
            val_dataset = rop_4cls.rop_4cls('val', transform=transform, target_transform=target_transform)
            val_loader = torch.torch.utils.data.DataLoader(dataset=val_dataset,
                                                           batch_size=batch_size,
                                                           shuffle=True,
                                                           num_workers=num_workers)

    if dataset_name == 'rop_3cls_combine_1_2':
        if 'train' in splits:
            train_set = rop_3cls_combine_1_2.rop_3cls_combine_1_2('train', transform=transform, target_transform=target_transform)
            train_loader = torch.torch.utils.data.DataLoader(dataset=train_set,
                                                             batch_size=batch_size,
                                                             shuffle=True,
                                                             num_workers=num_workers)
        if 'val' in splits or 'test' in splits:
            val_dataset = rop_3cls_combine_1_2.rop_3cls_combine_1_2('val', transform=transform, target_transform=target_transform)
            val_loader = torch.torch.utils.data.DataLoader(dataset=val_dataset,
                                                           batch_size=batch_size,
                                                           shuffle=True,
                                                           num_workers=num_workers)

    if dataset_name == 'rop_2cls':
        if 'train' in splits:
            train_set = rop_2cls.rop_2cls('train', transform=transform, target_transform=target_transform)
            train_loader = torch.torch.utils.data.DataLoader(dataset=train_set,
                                                             batch_size=batch_size,
                                                             shuffle=True,
                                                             num_workers=num_workers)
        if 'val' in splits or 'test' in splits:
            val_dataset = rop_2cls.rop_2cls('val', transform=transform, target_transform=target_transform)
            val_loader = torch.torch.utils.data.DataLoader(dataset=val_dataset,
                                                           batch_size=batch_size,
                                                           shuffle=True,
                                                           num_workers=num_workers)

    if dataset_name == 'rop_2cls_balance':
        if 'train' in splits:
            train_set = rop_2cls_balance.rop_2cls_balance('train', transform=transform, target_transform=target_transform)
            train_loader = torch.torch.utils.data.DataLoader(dataset=train_set,
                                                             batch_size=batch_size,
                                                             shuffle=True,
                                                             num_workers=num_workers)
        if 'val' in splits or 'test' in splits:
            val_dataset = rop_2cls_balance.rop_2cls_balance('val', transform=transform, target_transform=target_transform)
            val_loader = torch.torch.utils.data.DataLoader(dataset=val_dataset,
                                                           batch_size=batch_size,
                                                           shuffle=True,
                                                           num_workers=num_workers)

    if dataset_name == 'val_2cls':
        if 'train' in splits:
            train_set = val_2cls.val_2cls('train', transform=transform, target_transform=target_transform)
            train_loader = torch.torch.utils.data.DataLoader(dataset=train_set,
                                                             batch_size=batch_size,
                                                             shuffle=True,
                                                             num_workers=num_workers)
        if 'val' in splits or 'test' in splits:
            val_dataset = val_2cls.val_2cls('val', transform=transform, target_transform=target_transform)
            val_loader = torch.torch.utils.data.DataLoader(dataset=val_dataset,
                                                           batch_size=batch_size,
                                                           shuffle=True,
                                                           num_workers=num_workers)

    return train_loader, val_loader, test_loader
