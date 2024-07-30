import torch, os, glob, cv2, random
import torch.nn.functional as F
import numpy as np
from argparse import ArgumentParser
from model import Net
from utils import *
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

parser = ArgumentParser()
parser.add_argument("--epoch", type=int, default=600)
parser.add_argument("--module_num", type=int, default=20)
parser.add_argument("--block_size", type=int, default=33)
parser.add_argument("--model_dir", type=str, default="weight")
parser.add_argument("--data_dir", type=str, default="data")
parser.add_argument("--testset_name", type=str, default="Set11")
parser.add_argument("--result_dir", type=str, default="test_out")
parser.add_argument("--gpu_list", type=str, default="0")
parser.add_argument("--num_feature", type=int, default=32)
parser.add_argument("--max_ratio", type=float, default=0.1)

args = parser.parse_args()
epoch = args.epoch
K = args.module_num
B = args.block_size
C = args.num_feature

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_list
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# fixed seed for reproduction
seed = 2024
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

N = B * B
cs_ratio_list = [0.1,0.3,0.5]
max_ratio = args.max_ratio
max_M = int(np.ceil(max_ratio * N))

Phi = np.load(os.path.join(args.data_dir, "1089x1089_Gaussian.npy"))
Phi = torch.from_numpy(Phi).to(device)

A = lambda z: F.conv2d(z, Phi, stride=B)
AT = lambda z: F.conv_transpose2d(z, Phi, stride=B)

model = Net(K, C).to(device)
model_dir = "./%s/%f/layer_%d_block_%d_f_%d" % (args.model_dir, max_ratio, K, B, C)
model.load_state_dict(torch.load("%s/net_params_%d.pkl" % (model_dir, epoch)))

result_dir = "./%s/%s/%f/layer_%d_block_%d_f_%d" % (args.result_dir, args.testset_name, max_ratio, K, B, C)
os.makedirs(result_dir, exist_ok=True)

# test set iCo
test_image_paths = glob.glob(os.path.join(args.data_dir, args.testset_name, "*"))
test_image_num = len(test_image_paths)

def test(cs_ratio):
    with torch.no_grad():
        PSNR_list, SSIM_list = [], []
        for i, path in enumerate(test_image_paths):
            test_image = cv2.imread(path, 1)  # read test data from image file
            test_image_ycrcb = cv2.cvtColor(test_image, cv2.COLOR_BGR2YCrCb)
            img, old_h, old_w, img_pad, new_h, new_w = my_zero_pad(test_image_ycrcb[:,:,0], block_size=B)
            img_pad = img_pad.reshape(1, 1, new_h, new_w) / 255.0  # normalization
            x = torch.from_numpy(img_pad).to(device).float()
            q = torch.tensor([[cs_ratio * N]], device=device).ceil()
            cs_ratio = q / N
            mask = (torch.arange(N,device=device).view(1,N) < q).view(1,N,1,1)
            cur_A = lambda z: A(z) * mask
            y = cur_A(x)
            x_out = model(y, cur_A, AT, cs_ratio).squeeze()[:old_h, :old_w]
            x_out = (x_out.clamp(min=0.0, max=1.0) * 255.0).cpu().numpy()
            PSNR = psnr(x_out, img)
            SSIM = ssim(x_out, img, data_range=255)
            # print("[%d/%d] %s, PSNR: %.2f, SSIM: %.4f" % (i, test_image_num, path, PSNR, SSIM))
            test_image_ycrcb[:,:,0] = x_out
            test_image = cv2.cvtColor(test_image_ycrcb, cv2.COLOR_YCrCb2BGR).astype(np.uint8)
            result_path = os.path.join(result_dir, os.path.basename(path))
            cv2.imwrite("%s_ratio_%.2f_PSNR_%.2f_SSIM_%.4f.png" % (result_path, cs_ratio, PSNR, SSIM), test_image)
            PSNR_list.append(PSNR)
            SSIM_list.append(SSIM)
    return np.mean(PSNR_list), np.mean(SSIM_list)

for cs_ratio in cs_ratio_list:
    avg_psnr, avg_ssim = test(cs_ratio)
    print("Test Set is %s, Max Ratio is %f, CS Ratio is %.2f, avg PSNR is %.2f, avg SSIM is %.4f." % (args.testset_name, max_ratio, cs_ratio, avg_psnr, avg_ssim))