import torch
import argparse
import torch.backends.cudnn as cudnn
import os

from architecture.MST_plus import MST_plus
from utils import save_matv73
import glob
import cv2
import numpy as np
import itertools

os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description="SSR")
parser.add_argument("--outf", type=str, default=None, help='path log files')
parser.add_argument('--pretrained_model_path', type=str, default=None)
parser.add_argument('--ensemble_mode', type=str, default='mean')
opt = parser.parse_args()

pretrained_model_path = opt.pretrained_model_path
method = pretrained_model_path.split('/')[-2]


model = MST_plus().cuda()

def main():
    cudnn.benchmark = True
    print("\nbuilding models_baseline ...")
    print('Parameters number is ', sum(param.numel() for param in model.parameters()))
    if not os.path.exists(opt.outf):
        os.makedirs(opt.outf)

    if pretrained_model_path is not None:
        print(f'load model from {pretrained_model_path}')
        checkpoint = torch.load(pretrained_model_path)
        model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()},
                              strict=True)
    test_path = './Valid_RGB/'
    test(model, test_path, opt.outf)
    os.system(f'python prep_submission.py -i {opt.outf} -o {opt.outf}/submission -k')

def test(model, test_path, save_path):
    img_path_name = glob.glob(os.path.join(test_path, '*.jpg'))
    img_path_name.sort()
    var_name = 'cube'
    for i in range(len(img_path_name)):
        rgb = cv2.imread(img_path_name[i])
        if 'bgr' not in pretrained_model_path:
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        if 'norm' in pretrained_model_path:
            rgb = np.float32(rgb)
            rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())
        else:
            rgb = np.float32(rgb) / 255.0
        rgb = np.expand_dims(np.transpose(rgb, [2, 0, 1]), axis=0).copy()
        rgb = torch.from_numpy(rgb).float().cuda()
        with torch.no_grad():
            result = forward_ensemble(rgb, model, opt.ensemble_mode)
        result = result.cpu().numpy() * 1.0
        result = np.transpose(np.squeeze(result), [1, 2, 0])
        result = np.minimum(result, 1.0)
        result = np.maximum(result, 0)
        print(img_path_name[i].split('/')[-1])
        mat_name = img_path_name[i].split('/')[-1][:-4] + '.mat'
        mat_dir = os.path.join(save_path, mat_name)
        save_matv73(mat_dir, var_name, result)

def forward_ensemble(x, forward_func, ensemble_mode = 'mean'):
    def _transform(data, xflip, yflip, transpose, reverse=False):
        if not reverse:  # forward transform
            if xflip:
                data = torch.flip(data, [3])
            if yflip:
                data = torch.flip(data, [2])
            if transpose:
                data = torch.transpose(data, 2, 3)
        else:  # reverse transform
            if transpose:
                data = torch.transpose(data, 2, 3)
            if yflip:
                data = torch.flip(data, [2])
            if xflip:
                data = torch.flip(data, [3])
        return data

    outputs = []
    opts = itertools.product((False, True), (False, True), (False, True))
    for xflip, yflip, transpose in opts:
        data = x.clone()
        data = _transform(data, xflip, yflip, transpose)
        data = forward_func(data)
        outputs.append(
            _transform(data, xflip, yflip, transpose, reverse=True))
    if ensemble_mode == 'mean':
        return torch.stack(outputs, 0).mean(0)
    elif ensemble_mode == 'median':
        return torch.stack(outputs, 0).median(0)[0]


if __name__ == '__main__':
    main()