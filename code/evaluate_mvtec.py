import os
import torch
from torchvision import transforms
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_recall_curve, auc
from PIL import Image
import numpy as np
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

CLASS_NAMES = ['bottle', 'cable', 'capsule', 'carpet', 'grid','hazelnut', 'leather', 'metal_nut', 'pill', 'screw','tile', 'toothbrush', 'transistor', 'wood', 'zipper']


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
    

    
    normal_img_paths = [
        os.path.join(command_args.dataset_path, c_name, "train", "good",
                     f"{(command_args.round * 4 + i):03}.png")
        for i in range(command_args.k_shot)
    ]
    normal_img_paths = normal_img_paths[:command_args.k_shot]    
    
    p_pred = []
    p_label = []
    i_pred = []
    i_label = []
    for root, dirs, files in os.walk(root_dir):
        for file in tqdm(files, disable=True):
            file_path = os.path.join(root, file)
            if "test" in file_path and 'png' in file and c_name in file_path:
                
                anomaly_map, anomaly_score = predict(
                    file_path,
                    normal_img_paths if FEW_SHOT else [],
                    512,
                    0.1,
                    1.0,
                    [],
                    []
                )

                is_normal = 'good' in file_path.split('/')[-2]
                status_str = os.path.basename(os.path.dirname(file_path))
                img_stem = os.path.splitext(os.path.basename(file_path))[0]  # e.g. '035'

                if is_normal:
                    img_mask = Image.fromarray(np.zeros((448, 448)), mode='L')
                else:
                    mask_path = file_path.replace('test', 'ground_truth')
                    mask_path = mask_path.replace('.png', '_mask.png')
                    img_mask = Image.open(mask_path).convert('L')

                img_mask = mask_transform(img_mask)
                img_mask[img_mask > 0.1], img_mask[img_mask <= 0.1] = 1, 0
                img_mask = img_mask.squeeze().reshape(448, 448).cpu().numpy()
                if isinstance(anomaly_map, torch.Tensor):
                    if anomaly_map.requires_grad:
                        anomaly_map = anomaly_map.reshape(448, 448).detach().cpu().numpy()
                    else:
                        anomaly_map = anomaly_map.reshape(448, 448).cpu().numpy()
                else:
                    anomaly_map = anomaly_map.reshape(448, 448)


                # OUT_ROOT = "./paladin_results/mvtec"  
                # for sub in ["original", "anomaly_maps", "gt_masks"]:
                #     os.makedirs(os.path.join(OUT_ROOT, sub), exist_ok=True)

                # import matplotlib.pyplot as plt

                # # ---------- (a) save original ----------
                
                # orig_img = Image.open(file_path).convert("RGB").resize((448, 448))
                # orig_fname = f"{c_name}_{status_str}_{img_stem}.png"
                # orig_img.save(os.path.join(OUT_ROOT, "original", orig_fname))

                # # ---------- (b) save anomaly heat-map ----------

                # amap = anomaly_map.astype(np.float32)          # make sure it's float
                # amin, amax = amap.min(), amap.max()
                # if amax > amin:                                # avoid divide-by-zero
                #     amap = (amap - amin) / (amax - amin)
                # else:
                #     amap = np.zeros_like(amap)                 # all-one-color map if flat

                # fig, ax = plt.subplots(figsize=(2.24, 2.24), dpi=100)   # 448×448 px output
                # ax.imshow(orig_img)                                     # background layer

                # # Overlay the heat-map; adjust vmax if your model's scores exceed 1
                # ax.imshow(amap, cmap='jet', vmin=0, vmax=1, alpha=0.5)

                # ax.axis('off')
                # plt.tight_layout(pad=0)

                # amap_fname = f"{c_name}_{status_str}_{img_stem}_k{command_args.k_shot}.png"
                # plt.savefig(os.path.join(OUT_ROOT, "anomaly_maps", amap_fname),
                #             bbox_inches="tight", pad_inches=0)
                # plt.close()

                # # ---------- (c) save ground-truth mask ----------
                
                # mask_img = Image.fromarray((img_mask * 255).astype(np.uint8))
                # mask_fname = f"{c_name}_{status_str}_{img_stem}_mask.png"
                # mask_img.save(os.path.join(OUT_ROOT, "gt_masks", mask_fname))



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


    print(f"{c_name} - i_AUROC: {i_auroc}", flush=True)
    print(f"{c_name} - p_AUROC: {p_auroc}", flush=True)
    print(f"{c_name} - i_F1: {i_f1}", flush=True)
    print(f"{c_name} - p_F1: {p_f1}", flush=True)
    print(f"{c_name} - p_PRO: {p_pro}", flush=True)
    print(f"{c_name} - i_AP: {i_ap}", flush=True)



# Calculate means
print(f"Mean i_AUROC: {torch.tensor(i_auroc_list).mean().item():.2f}")
print(f"Mean p_AUROC: {torch.tensor(p_auroc_list).mean().item():.2f}")
print(f"Mean i_F1: {torch.tensor(i_f1_list).mean().item():.2f}")
print(f"Mean p_F1: {torch.tensor(p_f1_list).mean().item():.2f}")
print(f"Mean p_PRO: {torch.tensor(p_pro_list).mean().item():.2f}")
print(f"Mean i_AP: {torch.tensor(i_ap_list).mean().item():.2f}")
