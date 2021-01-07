from pathlib import Path
import random

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

import ssd.adain.net 
from ssd.adain.function import adaptive_instance_normalization

def style_transfer(vgg, decoder, content, style, alpha=1.0):
    assert (0.0 <= alpha <= 1.0)
    content_f = vgg(content)
    style_f = vgg(style)
    if interpolation_weights:
        _, C, H, W = content_f.size()
        feat = torch.FloatTensor(1, C, H, W).zero_().to(device)
        base_feat = adaptive_instance_normalization(content_f, style_f)
        for i, w in enumerate(interpolation_weights):
            feat = feat + w * base_feat[i:i + 1]
        content_f = content_f[0:1]
    else:
        feat = adaptive_instance_normalization(content_f, style_f)
    feat = feat * alpha + content_f * (1 - alpha)
    return decoder(feat)


def apply_style_transfer(vgg_path, decoder_path, content_images, style_images, p):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = '/content/drive/MyDrive/DA_detection/AdaIN_images/'
    content_paths = [Path(content_images)]
    style_paths = [Path(style_images)]
    random.seed()
    alpha = 1.0

    decoder = net.decoder
    vgg = net.vgg

    decoder.eval()
    vgg.eval()

    decoder.load_state_dict(torch.load(decoder_path))
    vgg.load_state_dict(torch.load(vgg_path))
    vgg = nn.Sequential(*list(vgg.children())[:31])

    vgg.to(device)
    decoder.to(device)

    # process one content and one style
    for content_path in content_paths:
        for style_path in style_paths:
            if random.random > p:
                content = transforms.ToTensor(Image.open(str(content_path)))
                style = transforms.ToTensor(Image.open(str(style_path)))

                content = content.to(device).unsqueeze(0)
                style = style.to(device).unsqueeze(0)
                with torch.no_grad():
                    output = style_transfer(vgg, decoder, content, style, alpha)

                output = output.cpu()

                # testing (optional)
                output_name = output_dir / '{:s}_stylized_{:s}'.format(content_path.stem, style_path.stem)
                save_image(output, str(output_name))

                break
            else:
                content = transforms.ToTensor(Image.open(str(content_path)))
                output = content

                # testing (optional)
                output_name = output_dir / '{:s}_not_stylized'.format(content_path.stem)
                save_image(output, str(output_name))
