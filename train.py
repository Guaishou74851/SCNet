import torch, os, glob, cv2, random
import torch.nn.functional as F
import scipy.io as sio
import numpy as np
from torch.utils.data import Dataset, DataLoader
from argparse import ArgumentParser
from model import Net
from utils import *
from skimage.metrics import structural_similarity as ssim
from time import time
from tqdm import tqdm

parser = ArgumentParser(description="SCNet")
parser.add_argument("--start_epoch", type=int, default=0)
parser.add_argument("--end_epoch", type=int, default=600)
parser.add_argument("--module_num", type=int, default=20)
parser.add_argument("--learning_rate", type=float, default=1e-4)
parser.add_argument("--block_size", type=int, default=33)
parser.add_argument("--model_dir", type=str, default="weight")
parser.add_argument("--data_dir", type=str, default="data")
parser.add_argument("--log_dir", type=str, default="log")
parser.add_argument("--save_interval", type=int, default=100)
parser.add_argument("--testset_name", type=str, default="Set11")
parser.add_argument("--gpu_list", type=str, default="0")
parser.add_argument("--num_feature", type=int, default=32)
parser.add_argument("--max_ratio", type=float, default=0.1)

args = parser.parse_args()

start_epoch, end_epoch = args.start_epoch, args.end_epoch
learning_rate = args.learning_rate
K = args.module_num
B = args.block_size
C = args.num_feature

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_list
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

print("device =", device)

# fixed seed for reproduction
seed = 2024
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

batch_size = 64
N = B * B
cs_ratio_list = [0.01, 0.04, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5]
max_ratio = args.max_ratio
max_M = int(np.ceil(max_ratio * N))

Phi = np.load(os.path.join(args.data_dir, "1089x1089_Gaussian.npy"))
Phi = torch.from_numpy(Phi).to(device)

A = lambda z: F.conv2d(z, Phi, stride=B)
AT = lambda z: F.conv_transpose2d(z, Phi, stride=B)

data = sio.loadmat(os.path.join(args.data_dir, "Training_Data.mat"))["labels"]
data = torch.from_numpy(data).reshape(88912, 1, B, B)

model = Net(K, C).to(device)

class MyDataset(Dataset):
    def __getitem__(self, index):
        return data[index]

    def __len__(self):
        return 88912

dataloader = DataLoader(dataset=MyDataset(), batch_size=batch_size, num_workers=8, drop_last=True)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[500], gamma=0.25, last_epoch=start_epoch-1)

model_dir = "./%s/%f/layer_%d_block_%d_f_%d" % (args.model_dir, args.max_ratio, K, B, C)
log_dir = "./%s/%f" % (args.log_dir, args.max_ratio)
log_path = "./%s/%f/layer_%d_block_%d_f_%d.txt" % (args.log_dir, args.max_ratio, K, B, C)
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

# test set info
test_image_paths = glob.glob(os.path.join(args.data_dir, args.testset_name, "*"))

def test(cs_ratio):
    with torch.no_grad():
        PSNR_list, SSIM_list = [], []
        for i in range(len(test_image_paths)):
            test_image = cv2.imread(test_image_paths[i], 1)  # read test data from image file
            test_image_ycrcb = cv2.cvtColor(test_image, cv2.COLOR_BGR2YCrCb)
            img, old_h, old_w, img_pad, new_h, new_w = my_zero_pad(test_image_ycrcb[:,:,0], block_size=B)
            img_pad = img_pad.reshape(1, 1, new_h, new_w) / 255.0  # normalization
            x_input = torch.from_numpy(img_pad).to(device).float()
            q = torch.tensor([[cs_ratio * N]], device=device).ceil()
            cs_ratio = q / N
            mask = (torch.arange(N,device=device).view(1,N) < q).view(1,N,1,1)
            cur_A = lambda z: A(z) * mask
            y = cur_A(x_input)
            x_output = model(y, cur_A, AT, cs_ratio).squeeze()[:old_h, :old_w]
            x_output = x_output.clamp(min=0.0,max=1.0).cpu().numpy() * 255.0
            PSNR_list.append(psnr(x_output, img))
            SSIM_list.append(ssim(x_output, img, data_range=255))
    return np.mean(PSNR_list), np.mean(SSIM_list)

print("start training...")
for epoch_i in range(start_epoch + 1, end_epoch + 1):
    start_time = time()
    loss_avg, iter_num = 0.0, 0
    loss_measure_avg = 0.0
    loss_image_avg = 0.0
    for x in tqdm(dataloader):
        x = x.to(device)
        q = torch.randint(low=1, high=max_M+1, size=(batch_size,1), device=device)
        cs_ratio = q / N
        cs_ratio_comp = max_ratio - cs_ratio
        mask = (torch.arange(N,device=device).view(1,N).expand(batch_size,N) < q).view(batch_size,N,1,1)
        mask = torch.cat([mask[:,:max_M][:,torch.randperm(max_M,device=device)], mask[:,max_M:]], dim=1)
        mask_max = (torch.arange(N,device=device).view(1,N).expand(batch_size,N) < max_M).view(batch_size,N,1,1)
        mask_comp = mask.logical_not().logical_and(mask_max)
        cur_A = lambda z: A(z) * mask
        cur_A_comp = lambda z: A(z) * mask_comp
        y = A(x)
        x_out1 = model(y*mask, cur_A, AT, cs_ratio)
        x_out2 = model(y*mask_comp, cur_A_comp, AT, cs_ratio_comp)
        loss1 = ((A(x_out1) - y) * mask_comp).abs().mean()
        loss2 = ((A(x_out2) - y) * mask).abs().mean()

        x_out1 = H(x_out1, random.randint(0, 7))
        x_out2 = H(x_out2, random.randint(0, 7))

        q_new = torch.randint(low=1, high=N+1, size=(batch_size,1), device=device)
        cs_ratio_new = q_new / N
        mask_new = (torch.arange(N,device=device).view(1,N).expand(batch_size,N) < q_new).view(batch_size,N,1,1)
        mask_new = mask_new[:,torch.randperm(N,device=device)]
        cur_A_new = lambda z: A(z) * mask_new
        x_out1_new = model(cur_A_new(x_out1), cur_A_new, AT, cs_ratio_new)
        x_out2_new = model(cur_A_new(x_out2), cur_A_new, AT, cs_ratio_new)
        loss1_new = (x_out1_new - x_out1).abs().mean()
        loss2_new = (x_out2_new - x_out2).abs().mean()

        loss_measure = loss1 + loss2
        loss_image = loss1_new + loss2_new
        loss = loss_measure + 0.1 * loss_image

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        iter_num += 1
        loss_avg += loss.item()
        loss_measure_avg += loss_measure.item()
        loss_image_avg += loss_image.item()

    scheduler.step()

    loss_avg /= iter_num
    loss_measure_avg /= iter_num
    loss_image_avg /= iter_num
    log_data = "[%d/%d] Average loss: %f, loss_measure: %f, loss_image: %f, time cost: %.2fs, cur lr is %f." % (epoch_i, end_epoch, loss_avg, loss_measure_avg, loss_image_avg, time() - start_time, scheduler.get_last_lr()[0])
    print(log_data)
    with open(log_path, "a") as log_file:
        log_file.write(log_data + "\n")

    if epoch_i % args.save_interval == 0:
        torch.save(model.state_dict(), "./%s/net_params_%d.pkl" % (model_dir, epoch_i))  # save only the parameters

    if epoch_i == 1 or epoch_i % 10 == 0:
        for cs_ratio in cs_ratio_list:
            cur_psnr, cur_ssim = test(cs_ratio)
            log_data = "CS Ratio is %.2f, PSNR is %.2f, SSIM is %.4f." % (cs_ratio, cur_psnr, cur_ssim)
            print(log_data)
            with open(log_path, "a") as log_file:
                log_file.write(log_data + "\n")