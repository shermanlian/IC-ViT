import os

import torch
import torch.nn as nn

from archs import build_model, DINOHead
import dino_utils

import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(args):

    model = build_model(
        args.network, 
        image_size=args.image_size, 
        channel=args.channel, 
        n_classes=0).to(device)
    embed_dim = model.embed_dim

    if os.path.isfile(args.checkpoint):
        print(args.checkpoint)
        checkpoint = torch.load(args.checkpoint, weights_only=False)

        model = dino_utils.MultiCropWrapper(
            model, DINOHead(embed_dim, 65536, use_bn=False, norm_last_layer=True)
        )
        model = nn.DataParallel(model)
        model.module.load_state_dict(checkpoint['teacher'])#student
        model = model.module.backbone

        os.makedirs('checkpoints', exist_ok=True)
        torch.save(model.state_dict(), f'checkpoints/{args.output}')
    
    print('Done!')



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='./logs/clean-code-test/checkpoints/checkpoint0200.pth', 
                            help='Pretrained warpped dino checkpoint')
    parser.add_argument('--output', default='checkpoint0200_converted.pth', 
                            help='Nowarpped dino checkpoint path')

    parser.add_argument('--network', default='icvit', help='network name')
    parser.add_argument('--image-size', type=int, default=32)
    parser.add_argument('--channel', type=int, default=1)

    args = parser.parse_args()

    main(args)
