import os
import torch
from torchvision import transforms
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_recall_curve, auc
from PIL import Image
import numpy as np
import csv
import argparse
from tqdm import tqdm
from skimage import measure
from model.paladin_model import Paladin


parser = argparse.ArgumentParser("Paladin", add_help=True)
# paths
parser.add_argument("--few_shot", type=lambda x: (str(x).lower() == 'true'), default=True)  # Correct boolean parsing
parser.add_argument("--k_shot", type=int, default=1)
parser.add_argument("--round", type=int, default=3)

parser.add_argument("--local_rank", type=int, default=0, help="Local rank for distributed training")
parser.add_argument("--paladin_ckpt_path", type=str, help="Path to the Paladin model checkpoint")
parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset root directory")


parser.add_argument("--method", type=str, default="paladin", choices=["paladin", "paladintopk"], help="Specify the model method")


command_args = parser.parse_args()

torch.cuda.set_device(command_args.local_rank)

FEW_SHOT = command_args.few_shot 

# init the model
args = {
    'model': 'model_paladin',
    'paladin_ckpt_path': command_args.paladin_ckpt_path,
    'stage': 2,
    'max_tgt_len': 128,
    'lora_r': 32,
    'lora_alpha': 32,
    'lora_dropout': 0.1,
}

method = command_args.method.lower()


model = Paladin(**args)

anomaly_ckpt = torch.load(args['paladin_ckpt_path'], map_location=torch.device('cpu'))
model.load_state_dict(anomaly_ckpt, strict=False)
model = model.eval().cuda()

def predict(
    image_path,
    normal_img_path,
    max_length,
    top_p,
    temperature,
    history,
    modality_cache,
):
    input_dict = {
        'prompt': '',
        'image_paths': [image_path] if image_path else [],
        'audio_paths': [],
        'video_paths': [],
        'thermal_paths': [],
        'normal_img_paths': normal_img_path if normal_img_path else [],
        'top_p': top_p,
        'temperature': temperature,
        'max_tgt_len': max_length,
        'modality_embeds': modality_cache
    }

    # Set topk based on method name
    topk = method == 'paladintopk'

    # Pass topk flag to model.generate
    return model.generate(input_dict, topk=topk)

root_dir = command_args.dataset_path

mask_transform = transforms.Compose([
    transforms.Resize((448, 448), interpolation=transforms.InterpolationMode.NEAREST),
    transforms.ToTensor()
])

datas_csv_path = os.path.join(command_args.dataset_path, 'split_csv', '1cls.csv')



CLASS_NAMES = ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1', 'macaroni2','pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum']

file_paths = {}
normal_img_path = {}

for class_name in CLASS_NAMES:
    file_paths[class_name] = []
    normal_img_path[class_name] = []

with open(datas_csv_path, 'r') as file:
    reader = csv.reader(file)

    for row in reader:
        if row[1] == 'test' and row[0] in CLASS_NAMES:
            file_paths[row[0]].append(os.path.join(root_dir, row[3]))
        if row[0] in CLASS_NAMES and len(normal_img_path[row[0]]) < command_args.round * 4 + command_args.k_shot and row[1] == 'train':
            normal_img_path[row[0]].append(os.path.join(root_dir, row[3]))


if FEW_SHOT:
    for i in CLASS_NAMES:
        normal_img_path[i] = normal_img_path[i][command_args.round * 4:]


def cal_pro_score(masks, amaps, max_step=200, expect_fpr=0.3):
    # ref: https://github.com/gudovskiy/cflow-ad/blob/master/train.py
    binary_amaps = np.zeros_like(amaps, dtype=bool)
    min_th, max_th = amaps.min(), amaps.max()
    delta = (max_th - min_th) / max_step
    pros, fprs, ths = [], [], []
    for th in np.arange(min_th, max_th, delta):
        binary_amaps[amaps <= th], binary_amaps[amaps > th] = 0, 1
        pro = []
        for binary_amap, mask in zip(binary_amaps, masks):
            for region in measure.regionprops(measure.label(mask)):
                tp_pixels = binary_amap[region.coords[:, 0], region.coords[:, 1]].sum()
                pro.append(tp_pixels / region.area)
        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum()
        pros.append(np.array(pro).mean())
        fprs.append(fpr)
        ths.append(th)
    pros, fprs, ths = np.array(pros), np.array(fprs), np.array(ths)
    idxes = fprs < expect_fpr
    fprs = fprs[idxes]
    fprs = (fprs - fprs.min()) / (fprs.max() - fprs.min())
    pro_auc = auc(fprs, pros[idxes])
    return pro_auc


# Lists to store metrics for each class
i_auroc_list = []
p_auroc_list = []
i_f1_list = []
p_f1_list = []
p_pro_list = []
i_ap_list = []


for c_name in CLASS_NAMES:

    p_pred = []
    p_label = []
    i_pred = []
    i_label = []
    for file_path in tqdm(file_paths[c_name], disable=True):
        if FEW_SHOT:
            anomaly_map, anomaly_score = predict(file_path, normal_img_path[c_name], 512, 0.01, 1.0, [], [])
        else:
            anomaly_map, anomaly_score = predict(file_path, None, 512, 0.01, 1.0, [], [])
        
        is_normal = 'Normal' in file_path.split('/')[-2]

        if is_normal:
            img_mask = Image.fromarray(np.zeros((448, 448)), mode='L')
        else:
            mask_path = file_path.replace('Images', 'Masks')
            mask_path = mask_path.replace('.JPG', '.png')
            img_mask = Image.open(mask_path).convert('L')

        img_mask = mask_transform(img_mask)
        threshold = img_mask.max() / 100
        img_mask[img_mask > threshold], img_mask[img_mask <= threshold] = 1, 0
        img_mask = img_mask.squeeze().reshape(448, 448).cpu().numpy()
        
        if isinstance(anomaly_map, torch.Tensor):
            if anomaly_map.requires_grad:
                anomaly_map = anomaly_map.reshape(448, 448).detach().cpu().numpy()
            else:
                anomaly_map = anomaly_map.reshape(448, 448).cpu().numpy()
        else:
            anomaly_map = anomaly_map.reshape(448, 448)
        



        
        
        p_label.append(img_mask)
        p_pred.append(anomaly_map)

        i_label.append(1 if not is_normal else 0)
        
        if anomaly_score is not None:
            i_pred.append(float(anomaly_score) if isinstance(anomaly_score, torch.Tensor) else anomaly_score)
        else:
            i_pred.append(anomaly_map.max())

    p_pred = np.array(p_pred)
    p_label = np.array(p_label)

    i_pred = np.array(i_pred)
    i_label = np.array(i_label)

    # Pixel-level AUROC
    p_auroc = round(roc_auc_score(p_label.ravel(), p_pred.ravel()) * 100, 2)
    p_auroc_list.append(p_auroc)

    # Image-level AUROC
    i_auroc = round(roc_auc_score(i_label.ravel(), i_pred.ravel()) * 100, 2)
    i_auroc_list.append(i_auroc)

    # Pixel-level F1
    precisions, recalls, thresholds = precision_recall_curve(p_label.ravel(), p_pred.ravel())
    f1_scores = (2 * precisions * recalls) / (precisions + recalls)
    p_f1 = round(np.max(f1_scores[np.isfinite(f1_scores)]) * 100, 2)
    p_f1_list.append(p_f1)

    # Image-level F1
    precisions, recalls, thresholds = precision_recall_curve(i_label.ravel(), i_pred.ravel())
    f1_scores = (2 * precisions * recalls) / (precisions + recalls)
    i_f1 = round(np.max(f1_scores[np.isfinite(f1_scores)]) * 100, 2)
    i_f1_list.append(i_f1)

    # Pixel-level PRO
    p_pro = round(cal_pro_score(p_label, p_pred) * 100, 2)
    p_pro_list.append(p_pro)

    # Image Level AP
    i_ap = round(average_precision_score(i_label.ravel(), i_pred.ravel()) * 100, 2)
    i_ap_list.append(i_ap)


    print(f"{c_name} - i_AUROC: {i_auroc}",flush=True)
    print(f"{c_name} - p_AUROC: {p_auroc}",flush=True)
    print(f"{c_name} - i_F1: {i_f1}",flush=True)
    print(f"{c_name} - p_F1: {p_f1}",flush=True)
    print(f"{c_name} - p_PRO: {p_pro}",flush=True)
    print(f"{c_name} - i_AP: {i_ap}",flush=True)



# Calculate means
print(f"Mean i_AUROC: {torch.tensor(i_auroc_list).mean().item():.2f}")
print(f"Mean p_AUROC: {torch.tensor(p_auroc_list).mean().item():.2f}")
print(f"Mean i_F1: {torch.tensor(i_f1_list).mean().item():.2f}")
print(f"Mean p_F1: {torch.tensor(p_f1_list).mean().item():.2f}")
print(f"Mean p_PRO: {torch.tensor(p_pro_list).mean().item():.2f}")
print(f"Mean i_AP: {torch.tensor(i_ap_list).mean().item():.2f}")
