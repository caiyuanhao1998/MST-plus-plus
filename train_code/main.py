import torch
import torch.nn as nn
import argparse
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import os
from hsi_dataset import TrainDataset, ValidDataset
from architecture.MST_plus import MST_plus
from utils import AverageMeter, initialize_logger, save_checkpoint,  \
    Loss_valid, time2file_name, batch_PSNR, Loss_train
import datetime

parser = argparse.ArgumentParser(description="SSR")
parser.add_argument("--method", type=str, default='mst_plus_1stg')
parser.add_argument("--batch_size", type=int, default=20, help="batch size")
parser.add_argument("--end_epoch", type=int, default=300, help="number of epochs")
parser.add_argument("--init_lr", type=float, default=4e-4, help="initial learning rate")
parser.add_argument("--outf", type=str, default='./exp/', help='path log files')
parser.add_argument("--data_root", type=str, default='./ARAD_1K/')
parser.add_argument("--patch_size", type=int, default=128, help="patch size")
parser.add_argument("--stride", type=int, default=8, help="stride")
parser.add_argument("--gpu_id", type=str, default='0', help='path log files')
parser.add_argument('-norm', help="rescale the RGB into [0,1]", action='store_true', default=False)
opt = parser.parse_args()
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

# load dataset
print("\nloading dataset ...")
train_data = TrainDataset(data_root=opt.data_root, crop_size=opt.patch_size, bgr2rgb=True, arg=True, stride=opt.stride, norm=opt.norm)
print(f"Iteration per epoch: {len(train_data)}")
val_data = ValidDataset(data_root=opt.data_root, bgr2rgb=True, norm=opt.norm)
print("Validation set samples: ", len(val_data))

# Parameters, Loss and Optimizer
per_epoch_iteration = 1000
total_iteration = per_epoch_iteration*opt.end_epoch
criterion_train = Loss_train()
criterion_valid = Loss_valid()

# model
method = opt.method
print("\nbuilding models_baseline ...")
model = MST_plus().cuda()

print('Parameters number is ', sum(param.numel() for param in model.parameters()))

date_time = str(datetime.datetime.now())
date_time = time2file_name(date_time)
if opt.norm:
    opt.outf = opt.outf + '/' +f'{method}_lr{opt.init_lr}_p{opt.patch_size}_s{opt.stride}_norm/' + date_time
else:
    opt.outf = opt.outf + '/' +f'{method}_lr{opt.init_lr}_p{opt.patch_size}_s{opt.stride}/' + date_time
if not os.path.exists(opt.outf):
    os.makedirs(opt.outf)

if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
if torch.cuda.is_available():
    model.cuda()
    criterion_train.cuda()
    criterion_valid.cuda()

optimizer = optim.Adam(model.parameters(), lr=opt.init_lr, betas=(0.9, 0.999))
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_iteration, eta_min=1e-6)

# visualzation
log_dir = os.path.join(opt.outf, 'train.log')
logger = initialize_logger(log_dir)

# Resume
resume_file = ''
if resume_file:
    if os.path.isfile(resume_file):
        print("=> loading checkpoint '{}'".format(resume_file))
        checkpoint = torch.load(resume_file)
        start_epoch = checkpoint['epoch']
        iteration = checkpoint['iter']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

def main():
    cudnn.benchmark = True
    iteration = 0
    record_val_loss = 1000
    while iteration<total_iteration:
        model.train()
        losses = AverageMeter()
        train_loader = DataLoader(dataset=train_data, batch_size=opt.batch_size, shuffle=True, num_workers=2,
                                  pin_memory=True, drop_last=True)
        val_loader = DataLoader(dataset=val_data, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
        for i, (images, labels) in enumerate(train_loader):
            labels = labels.cuda()
            images = images.cuda()
            images = Variable(images)
            labels = Variable(labels)
            lr = optimizer.param_groups[0]['lr']
            optimizer.zero_grad()
            output = model(images)
            loss = criterion_train(output, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            losses.update(loss.data)
            iteration = iteration+1
            if iteration % 20 == 0:
                print('[iter:%d/%d],lr=%.9f,train_losses.avg=%.9f'
                      % (iteration, total_iteration, lr, losses.avg))
            if iteration % 1000 == 0:
                val_loss, psnr = validate(val_loader, model, criterion_valid)
                print(f'Valid loss:{val_loss}, PNSR:{psnr}')
                # Save model
                if torch.abs(val_loss - record_val_loss) < 0.01 or val_loss < record_val_loss or iteration % 5000 == 0:
                    print(f'Saving to {opt.outf}')
                    save_checkpoint(opt.outf, (iteration // 1000), iteration, model, optimizer)
                    if val_loss < record_val_loss:
                        record_val_loss = val_loss
                # print loss
                print(" Iter[%06d], Epoch[%06d], learning rate : %.9f, Train Loss: %.9f, Test Loss: %.9f, "
                      "Test PSNR: %.9f " % (iteration, iteration//1000, lr, losses.avg, val_loss, psnr))
                logger.info("Iter[%06d],  Epoch[%06d], learning rate : %.9f, Train Loss: %.9f, Test Loss: %.9f, "
                      "Test PSNR: %.9f " % (iteration, iteration//1000, lr, losses.avg, val_loss, psnr))
    return 0

# Validate
def validate(val_loader, model, criterion):
    model.eval()
    losses = AverageMeter()
    psnrs = AverageMeter()
    for i, (input, target) in enumerate(val_loader):
        input = input.cuda()
        target = target.cuda()
        with torch.no_grad():
            # compute output
            output = model(input)
            loss = criterion(output[:, :, 128:-128, 128:-128], target[:, :, 128:-128, 128:-128])
            psnr = batch_PSNR(output[:, :, 128:-128, 128:-128], target[:, :, 128:-128, 128:-128])
        # record loss
        losses.update(loss.data)
        psnrs.update(psnr.data)
    return losses.avg, psnrs.avg

if __name__ == '__main__':
    main()
    print(torch.__version__)