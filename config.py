# this is used for storing configurations of datasets & models

datasets = {
    'cifar10': {
        'num_classes': 10,
        'augmentation': False,
    },
    'cifar10+': {
        'num_classes': 10,
        'augmentation': True,
    },
    'cifar100': {
        'num_classes': 100,
        'augmentation': False,
    },
    'cifar100+': {
        'num_classes': 100,
        'augmentation': True,
    },
    'rop_4cls': {
        'num_classes': 4,
        'augmentation': True,
    },
    'rop_3cls_combine_1_2': {
        'num_classes': 3,
        'augmentation': True,
    },
    'rop_2cls': {
        'num_classes': 2,
        'augmentation': True,
    },
    'rop_2cls_balance': {
        'num_classes': 2,
        'augmentation': True,
    },
    'val_2cls': {
        'num_classes': 2,
        'augmentation': True,
    },
}
