import os
import torchvision.models as models

from models.densenet import *
from models.DualPathNet import *

def load_pretrained_diff_parameter(model, model_path):
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_path)
    diff = {k: v for k, v in model_dict.items() if \
            k in pretrained_dict and pretrained_dict[k].size() != v.size()}
    pretrained_dict.update(diff)
    model.load_state_dict(pretrained_dict)
    return model

def get_model(name, n_classes):
    model = _get_model_instance(name)

    if name == 'densenet':
        model = model(n_classes=n_classes,
                      is_batchnorm=True,
                      in_channels=3,
                      is_deconv=True)
    elif name == 'DualPathNet':
        model = model(
            num_init_features=128, k_r=200, groups=50,
            k_sec=(4, 8, 20, 3), inc_sec=(20, 64, 64, 128),
            num_classes=n_classes, test_time_pool=True)
        # model_path = '/home/ubuntu/skin_demo/RoP/models/kaggle/DualPathNet107_downsamping_enhance_DualPathNet107enhance_epoch12_loss0.9639.pkl'
        # model = load_pretrained_diff_parameter(model, model_path)
    else:
        model = model(n_classes=n_classes)

    return model


def _get_model_instance(name):
    try:
        return {
            'DualPathNet': DPN,
        }[name]
    except:
        print('Model {} not available'.format(name))
