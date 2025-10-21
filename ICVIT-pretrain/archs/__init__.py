from .resnet import build_resnet
from .hf_vit import build_vitencoder, DINOHead
from .vit import vit_small, vit_base
from .ic_vit import vit_small as icvit_small


def build_model(name, image_size=224, channel=3, n_classes=1, drop_path_rate=0.1, pretrained=False):
    if name == 'resnet':
        return build_resnet(channel=channel, n_classes=n_classes, pretrained=pretrained)
    elif name == 'vit':
        return vit_small(
            in_chans=channel, 
            num_classes=n_classes, 
            drop_path_rate=drop_path_rate)
    elif 'dino' in name:
        return build_vitencoder(
            img_size=image_size,
            channel=channel, 
            n_classes=n_classes, 
            drop_path_rate=drop_path_rate, 
            model_name=name)
    elif name == 'chvit':
        return build_vitencoder(
            img_size=image_size,
            channel=channel, 
            n_classes=n_classes, 
            drop_path_rate=drop_path_rate, 
            model_name='chvit')
    elif name == 'icvit':
        return icvit_small(
            img_size=[image_size],
            in_chans=channel, 
            num_classes=n_classes, 
            drop_path_rate=drop_path_rate)
    else:
        return None
