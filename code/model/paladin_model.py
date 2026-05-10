from __future__ import annotations
from torch import Tensor
import os
import sys
from pathlib import Path
import cv2
import matplotlib.pyplot as plt

from header import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Optional, List

from utils.losses import FocalLoss, BinaryDiceLoss

from .projection_layers import LinearLayerClip
from .image_preprocessing import load_and_transform_vision_data
from . import open_clip_local as open_clip
from .open_clip_local.transformer import text_global_pool
from .open_clip_local import get_tokenizer
from PIL import Image
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DINO_REPO_DIR = PROJECT_ROOT / "dinov3"
sys.path.append(str(DINO_REPO_DIR))

########################################################
# 1) DEFAULT PROMPTS
########################################################

prompt_normal_default = [
    '{}',
    'flawless {}',
    'perfect {}',
    'unblemished {}',
    '{} without flaw',
    '{} without defect',
    '{} without damage'
]

prompt_abnormal_default = [
    'damaged {}',
    'broken {}',
    '{} with flaw',
    '{} with defect',
    '{} with damage'
]

prompt_templates_default = [
    'a photo of a {}.',
    'a photo of the {}.'
]

########################################################
# 2) LIST OF ALL OBJECTS (TRAINING CLASSES)
########################################################

objs = [
    'bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather',
    'metal nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor',
    'wood', 'zipper', 'object', 'candle', 'cashew', 'chewinggum', 'fryum',
    'macaroni', 'pcb', 'pipe fryum', 'macaroni1', 'macaroni2', 'pcb1',
    'pcb2', 'pcb3', 'pcb4', 'capsules'
]

########################################################
# 3) MVTec-AD FINE-GRAINED ANOMALY DETAILS
########################################################

mvtec_anomaly_detail_gpt = {
    "carpet":     "discoloration in a specific area,irregular patch or section with a different texture,frayed edges or unraveling fibers,burn mark or scorching",
    "grid":       "crooked,cracks,excessive gaps,discoloration,deformation,missing,inconsistent spacing between grid elements,corrosion,visible signs,chipping",
    "leather":    "scratches,discoloration,creases,uneven texture,tears,brittleness,damage,seams,heat damage,mold",
    "tile":       "chipped,irregularities,discoloration,efflorescence,warping,missing,depressions,lippage,fungus,damage",
    "wood":       "knots,warping,cracks along the grain,mold growth on the surface,staining from water damage,wood rot,woodworm holes,rough patches,protruding knots",
    "bottle":     "cracked large,cracked small,dented large,dented small,leaking,discolored,deformed,missing cap,excessive condensation,unusual odor",
    "cable":      "twisted,knotted cable strands,detached connectors,excessive stretching,dents,corrosion,scorching along the cable,exposed conductive material",
    "capsule":    "irregular shape,discoloration coloring,crinkled,uneven seam,condensation inside the capsule,foreign particles,unusually soft or hard",
    "hazelnut":   "fungal growth,Unusual discoloration,rotten or foul odor emanating,insect infestation,wetness,misshapen shell,unusually thin,contaminants,unusual texture",
    "metal nut":  "cracks,irregular threading,corrosion,missing,distortion,signs of discoloration,excessive wear on contact surfaces,inconsistent texture",
    "pill":       "irregular shape,crumbling texture,excessive powder,Uneven coating,presence of air bubbles,disintegration,abnormal specks",
    "screw":      "rust on the surface,bent,damaged threads,stripped threads,deformed top,coating damage,uneven grooves,inconsistent size",
    "toothbrush": "loose bristles,uneven bristle distribution,excessive shedding of bristles,staining on the bristles,abrasive texture,irregularities in the shape",
    "transistor": "burn marks,detached leads,signs of corrosion,irregularities in the shape,presence of cracks or fractures,signs of physical trauma,irregularities in the surface texture",
    "zipper":     "bent,frayed,misaligned,excessive stiffness,corroded,detaches,loose,warped",
}

########################################################
# 4) VISA-SPECIFIC FINE-GRAINED ANOMALY DETAILS
########################################################

visa_anomaly_detail_gpt = {
    "candle":      "cracks or fissures in the wax,Wax pooling unevenly around the wick,tunneling,incomplete wax melt pool,irregular or flickering flame,other,extra wax in candle,wax melded out of the candle",
    "capsules":    "uneven capsule size,capsule shell appears brittle,excessively soft,dents,condensation,irregular seams or joints,specks",
    "cashew":      "uneven coloring,fungal growth,presence of foreign objects,unusual texture,empty shells,signs of moisture,stuck together",
    "chewinggum":  "consistency,presence of foreign objects,uneven coloring,excessive hardness,similar colour spot",
    "fryum":       "irregular shape,unusual odor,uneven coloring,unusual texture,small scratches,different colour spot,fryum stuck together,other",
    "macaroni1":   "uneven shape ,small scratches,small cracks,uneven coloring,signs of insect infestation,uneven texture,Unusual consistency",
    "macaroni2":   "irregular shape,small scratches,presence of foreign particles,excessive moisture,Signs of infestation,small cracks,unusual texture",
    "pcb1":        "oxidation on the copper traces,separation of layers,presence of solder bridges,excessive solder residue,discoloration,Uneven solder joints,bowing of the board,missing vias",
    "pcb2":        "oxidation on the copper traces,separation of layers,presence of solder bridges,excessive solder residue,discoloration,Uneven solder joints,bowing of the board,missing vias",
    "pcb3":        "oxidation on the copper traces,separation of layers,presence of solder bridges,excessive solder residue,discoloration,Uneven solder joints,bowing of the board,missing vias",
    "pcb4":        "oxidation on the copper traces,separation of layers,presence of solder bridges,excessive solder residue,discoloration,Uneven solder joints,bowing of the board,missing vias",
    "pipe fryum":  "uneven shape,presence of foreign objects,different colour spot,unusual odor,empty interior,unusual texture,similar colour spot,stuck together",
}


# Convert comma-separated strings → lists
mvtec_anomaly_detail = {
    k: [s.strip() for s in v.split(',')]
    for k, v in mvtec_anomaly_detail_gpt.items()
}

visa_anomaly_detail = {
    k: [s.strip() for s in v.split(',')]
    for k, v in visa_anomaly_detail_gpt.items()
}

# Merge all sources (later entries override earlier ones)
anomaly_detail = {**mvtec_anomaly_detail, **visa_anomaly_detail}

########################################################
# 5) BUILD PROMPT SENTENCES DICTIONARY 
########################################################

prompt_sentences_clip_l14 = {}
for obj in objs:
    # normalize key (e.g. 'metal_nut' → 'metal nut')
    obj_key = obj.replace('_', ' ')
    
    # 6a) normal prompts (always default)
    normal_list = [p.format(obj_key) for p in prompt_normal_default]
    
    # 6b) abnormal prompts
    if obj_key in anomaly_detail:
        # use the fine-grained list
        abnormal_list = [
            f"{obj_key} with {detail}"
            for detail in anomaly_detail[obj_key]
        ]
    else:
        # fallback to the generic abnormal templates
        abnormal_list = [p.format(obj_key) for p in prompt_abnormal_default]
    
    # 6c) apply the generic templates (e.g. “a photo of a {}.”)
    prompt_group_list_clip_l14 = []

    for group in (normal_list, abnormal_list):
        sentences = [
            tmpl.format(sent)
            for sent in group
            for tmpl in prompt_templates_default
        ]

        transformed_clip_l14 = get_tokenizer("ViT-L-14")(sentences).to(torch.cuda.current_device())
         
        prompt_group_list_clip_l14.append(transformed_clip_l14)
    prompt_sentences_clip_l14[obj_key] = prompt_group_list_clip_l14


class TextPromptLearner(nn.Module):
    """
    Two independent learnable 2‑token prefixes for the OpenCLIP text tower:
        * extra_tokens_normal   – prepended to every *normal* prompt
        * extra_tokens_abnormal – prepended to every *abnormal* prompt
    The positional‑embedding prefix (length = n_ctx) is also trainable;
    the remaining rows stay frozen as a buffer.
    """
    def __init__(self, clip_model: nn.Module, n_ctx: int = 2):
        super().__init__()
        W     = clip_model.token_embedding.embedding_dim            # 768 for ViT‑L/14
        dtype = clip_model.token_embedding.weight.dtype             # fp16 / bf16

        # ────────────────────────────────────────────────────────────
        # 1) Learnable text‑prefix tokens (for normal / abnormal prompts)
        # ────────────────────────────────────────────────────────────
        self.extra_tokens_normal   = nn.Parameter(torch.randn(n_ctx, W, dtype=dtype) * 0.02)
        self.extra_tokens_abnormal = nn.Parameter(torch.randn(n_ctx, W, dtype=dtype) * 0.02)

        # ────────────────────────────────────────────────────────────
        # 2) Positional embedding: split into trainable prefix + frozen body
        # ────────────────────────────────────────────────────────────
        pe_old = clip_model.positional_embedding              # (77, W)

        # Trainable part (the new rows we prepend)
        self.pos_prefix = nn.Parameter(pe_old[:n_ctx].clone())        # (n_ctx, W)

        # Frozen part (original 77 rows).  Registered as a *buffer* so it
        # lives in the state_dict but never gets a gradient.
        self.register_buffer("pos_fixed", pe_old.clone(), persistent=True)  # (77, W)

        # Tell OpenCLIP the new sequence length so that it rebuilds attention masks
        clip_model.context_length = pe_old.size(0) + n_ctx
        clip_model.attn_mask      = None           # will be rebuilt on‑the‑fly

        # ────────────────────────────────────────────────────────────
        # 3) Freeze everything else in the text tower
        # ────────────────────────────────────────────────────────────
        for p in clip_model.transformer.parameters():     p.requires_grad = False
        for p in clip_model.token_embedding.parameters(): p.requires_grad = False
        clip_model.ln_final.requires_grad_(False)

        if clip_model.text_projection is not None:
            if isinstance(clip_model.text_projection, nn.Linear):
                clip_model.text_projection.requires_grad_(False)
            else:                                         # `Parameter`
                clip_model.text_projection.requires_grad = False

        self.clip_model = clip_model
        self.n_ctx      = n_ctx

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    
    def _embed_fixed(self, tokens: torch.LongTensor):
        """Token‑embedding lookup (frozen)."""
        return self.clip_model.token_embedding(tokens)

    def _encode(self, x: torch.Tensor, tokens: torch.LongTensor):
        """
        Add positional embeddings, run the frozen transformer,
        pool, then (if present) project.
        """
        dtype = self.clip_model.transformer.get_cast_dtype()

        # Assemble positional embedding on‑the‑fly
        pos_full = torch.cat([self.pos_prefix, self.pos_fixed], dim=0)     # (n_ctx+77, W)
        pos      = pos_full[:x.size(1), :].to(dtype)                       # match seq‑len
        x        = x + pos

        out = self.clip_model.transformer(x)
        out = out[0] if isinstance(out, tuple) else out
        out = self.clip_model.ln_final(out)

        pooled = text_global_pool(out, tokens, pool_type='argmax')

        proj = self.clip_model.text_projection
        if proj is None:
            return pooled
        if isinstance(proj, nn.Linear):
            return proj(pooled)        # Linear layer
        else:
            return pooled @ proj       # matrix multiply

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, tokenised: torch.LongTensor, abnormal: bool) -> torch.Tensor:
        """
        tokenised : (B, L) token ids (77 max from OpenCLIP)
        abnormal  : bool  → choose which learnable prefix to prepend
        Returns   : (B, 768) text features
        """
        B, L = tokenised.shape

        txt = self._embed_fixed(tokenised)                       # (B, L, W)

        prefix_param = self.extra_tokens_abnormal if abnormal else self.extra_tokens_normal
        prefix       = prefix_param.unsqueeze(0).expand(B, -1, -1)  # (B, n_ctx, W)

        x = torch.cat([prefix, txt], dim=1)                      # (B, L+n_ctx, W)
        return self._encode(x, tokenised)                        # (B, 768)


class Paladin(nn.Module):

    def __init__(self, **args):
        super().__init__()
        self.args = args

        # ──────────────────────────────────────────────────────────────
        # OpenCLIP text pathway  (ViT‑L/14 → 768‑d)
        #      – keep vision weights on CPU to save GPU memory
        # ──────────────────────────────────────────────────────────────
        self.clip_name       = args.get("clip_model_name", "ViT-L-14")
        self.clip_pretrained = args.get("clip_pretrained",  "openai")

        print(f"[Init] Loading OpenCLIP text encoder {self.clip_name} "
            f"({self.clip_pretrained}) …")
        self.image_size      = 448

        # 1. full CLIP model on **CPU**
        clip_full, _, _ = open_clip.create_model_and_transforms(
            self.clip_name,
            self.image_size,
            pretrained=self.clip_pretrained,
        )
        clip_full.eval().to("cpu")
        for p in clip_full.parameters():
            p.requires_grad = False          # keep it frozen

        # 2. move ONLY the text‑side tensors to GPU
        gpu_device = torch.cuda.current_device()
        for name, param in clip_full.named_parameters():
            if not name.startswith("visual."):      # skip ViT parameters
                param.data = param.data.to(gpu_device)
        for name, buf in clip_full.named_buffers():
            if not name.startswith("visual."):
                buf.data = buf.data.to(gpu_device)

        # 3. keep the whole object but use it only for encode_text()
        self.clip_text_encoder = clip_full          # .encode_text() is available
        self.txt_embed_dim = clip_full.text_projection.shape[1]   # 768 for ViT‑L/14

        # 4. (optional) drop the vision tower to free **CPU** RAM
        del clip_full.visual
        torch.cuda.empty_cache()
        print("[Init] Text pathway moved to GPU; vision tower remains on CPU.")


        # ──────────────────────────────────────────────────────────────
        # DINOV2 visual backbone  (ViT‑L/14)
        # ──────────────────────────────────────────────────────────────
        self.model_name       = args.get("dino_model_name", "dinov3_vitl16")
        self.pretrained = args.get("dino_pretrained",  True)
        # Which transformer blocks to return for patch features
        self.features_list   = args.get("features_list", [5, 11, 17, 23])
        # Resolve the checkpoint from the project root instead of a machine-specific path.
        weights_path = str(
            PROJECT_ROOT / "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
        )
        # path to your local cloned dinov3 repo
        REPO_DIR = str(DINO_REPO_DIR)

        print(f"[Init] Loading DINOv3 {self.model_name} …")
        self.visual_encoder = torch.hub.load(
            REPO_DIR,
            self.model_name,
            source="local",
            weights=weights_path
        ).eval().to(torch.cuda.current_device())
        
        for p in self.visual_encoder.parameters():
            p.requires_grad = False
        print("[Init] Visual encoder ready.")
        
        # ---- important CLIP dimensions --------------------------------
        #  * patch / hidden width  = 1024  (ViT width)
        #  * projected embed dim   =  768  (CLS after proj layer)
        self.patch_width   = self.visual_encoder.embed_dim
        
        self.visual_hidden_size = self.patch_width

        self.image_decoder = LinearLayerClip(self.patch_width, self.txt_embed_dim, 4)

        # ──────────────────────────────────────────────────────────────
        self.loss_focal = FocalLoss()
        self.loss_dice  = BinaryDiceLoss()

        # misc
        self.device      = torch.cuda.current_device()

        # ─────────────── learnable Tokens ────────────────
        self.text_prompt_learner = TextPromptLearner(self.clip_text_encoder,
                                                     n_ctx=2).to(self.device)
        self._debug_counter = 0

    def encode_text(self, obj: List[str]) -> Tensor:
        """
        Re-uses your original prompt-sentence dictionary.
        Returns (B,2,768)  (normal, abnormal)
        """
        global prompt_sentences_clip_l14
        
        B = len(obj)
        normal_tok, abnormal_tok = [], []
        for name in obj:
            nrm, abn = prompt_sentences_clip_l14[name.replace("_", " ")]
            normal_tok.append(nrm)
            abnormal_tok.append(abn)

        tok_n = torch.cat(normal_tok).to(self.device)
        tok_a = torch.cat(abnormal_tok).to(self.device)

        with torch.no_grad():
            emb_n = self.clip_text_encoder.encode_text(tok_n)  # (B*P,768)
            emb_a = self.clip_text_encoder.encode_text(tok_a)

        Pn, Pa = emb_n.shape[0] // B, emb_a.shape[0] // B
        emb_n = emb_n.view(B, Pn, -1).mean(dim=1, keepdim=True)
        emb_a = emb_a.view(B, Pa, -1).mean(dim=1, keepdim=True)
        return torch.cat([emb_n, emb_a], dim=1)      


    def encode_text_with_prompt_learnable(self, obj: list[str]) -> torch.Tensor:
        """
        obj : list[str] of class names (len == B)
        Returns
        -------
        (B, 2, 768) tensor  [normal_embed , abnormal_embed]
        """

        global prompt_sentences_clip_l14  # ← reused cache
        B = len(obj)

        # ------------------------------------------------------------------
        # 1) gather the *tokenised* prompts from the global dict
        #    prompt_sentences_clip_l14[name]  == [tokens_normal , tokens_abnormal]
        # ------------------------------------------------------------------
        normal_tokens   = []
        abnormal_tokens = []
        for name in obj:
            name = name.replace('_', ' ')
            tok_normal, tok_abnormal = prompt_sentences_clip_l14[name]       # tensors
            normal_tokens.append(tok_normal)        # shape (P, 77)
            abnormal_tokens.append(tok_abnormal)

        tokens_normal   = torch.cat(normal_tokens,   dim=0).to(self.device)  # (B*Pn, L)
        tokens_abnormal = torch.cat(abnormal_tokens, dim=0).to(self.device)  # (B*Pa, L)

        # ------------------------------------------------------------------
        # 2) encode with CLIP + learnable prefixes
        #    (do NOT wrap in torch.no_grad() – the prefixes need gradients)
        # ------------------------------------------------------------------
        emb_normal   = self.text_prompt_learner(tokens_normal,   abnormal=False)  # (B*Pn, 768)
        emb_abnormal = self.text_prompt_learner(tokens_abnormal, abnormal=True)   # (B*Pa, 768)

        # ------------------------------------------------------------------
        # 3) reshape → (B, #prompts, D) and mean‑pool over the prompt set
        # ------------------------------------------------------------------
        Pn = emb_normal.shape[0] // B
        Pa = emb_abnormal.shape[0] // B

        emb_normal   = emb_normal.view(B, Pn, -1).mean(dim=1, keepdim=True)   # (B,1,768)
        emb_abnormal = emb_abnormal.view(B, Pa, -1).mean(dim=1, keepdim=True) # (B,1,768)
        
        # ------------------------------------------------------------------
        # 4) concatenate normal + abnormal → (B, 2, 768)
        # ------------------------------------------------------------------
        return torch.cat([emb_normal, emb_abnormal], dim=1)

    @torch.no_grad()
    def _encode_image_tensor(
        self,
        batch: torch.Tensor,
        return_patches: bool = True
    ) -> tuple[torch.Tensor, Optional[list[torch.Tensor]]]:
        
        """
        Encodes a batch of images into global and patch-level features.

        Args:
            batch (torch.Tensor): Image batch of shape (B, C, H, W) in range [0, 1], on CUDA
            return_patches (bool): Whether to return patch-level features

        Returns:
            tuple:
                - cls_emb (torch.Tensor): Global CLS embedding of shape (B, D)
                - patches (list[torch.Tensor] | None): Patch embeddings per layer (B, N, D)
        """
        
        # keep fp16 / bf16 consistency with the backbone
        batch = batch.to(next(self.visual_encoder.parameters()).dtype)


        if return_patches:
            feats = self.visual_encoder.get_intermediate_layers(
                batch,
                n=self.features_list,          # can be an int or a sequence
                return_class_token=True        # (patch_tokens, cls_token)
            )                                   # → tuple[(B,1+N,D), (B,D)]
                                                #                     patches   CLS
            # unzip the list of tuples
            patches, cls_tokens = zip(*feats)   # each element shape as above
            cls_emb = cls_tokens[-1]                  # take CLS from the deepest layer

            return cls_emb, patches
        else:
            # the regular forward already returns the global representation
            cls_emb = self.visual_encoder(batch)      # (B, D)
        return cls_emb, None


    def get_patch_tokens_from_path(self, image_paths: list[str], *, return_cls: bool = False) -> Union[list[torch.Tensor], tuple[list[torch.Tensor], torch.Tensor]]:
        
        """
        Encodes images from file paths.

        Args:
            image_paths (list[str]): List of file paths
            return_cls (bool): Whether to also return CLS token embeddings

        Returns:
            list[torch.Tensor]: Patch tokens per layer
            or
            tuple: (patch_tokens, cls_raw) if return_cls is True
        """
        
        batch = load_and_transform_vision_data(image_paths, self.device)
        cls_raw, feats = self._encode_image_tensor(batch, return_patches=True)  # <-- no padding

        patch_tokens = self.image_decoder(feats)                # list[(B,N,768)]

        if return_cls:
            # cls_raw is still 768-d (or whatever the backbone natively outputs)
            return patch_tokens, cls_raw
        
        return patch_tokens

    def get_patch_tokens_from_tensor(self, image_tensors: Union[list[torch.Tensor], torch.Tensor]) -> list[torch.Tensor]:
        
        """
        Encodes images from already-loaded tensors.

        Args:
            image_tensors (list[torch.Tensor] or torch.Tensor): Single or list of (C, H, W) tensors

        Returns:
            list[torch.Tensor]: Patch tokens per layer
        """
        
        
        if not isinstance(image_tensors, list):
            image_tensors = [image_tensors]
            
        batch = torch.stack(image_tensors, 0).to(self.device)
        cls_emb, feats = self._encode_image_tensor(batch, return_patches=True)
        patch_tokens = self.image_decoder(feats)
        
        return patch_tokens

    def get_patch_tokens_for_fewshot(self, image_paths: list[str]) -> list[torch.Tensor]:
        
        
        """
        Extract patch-level features from image paths for one-shot matching.

        Args:
            image_paths (list[str]): Paths to input images

        Returns:
            list[torch.Tensor]: Patch features from each selected layer
        """        
        
        batch = load_and_transform_vision_data(image_paths, self.device)
        _, feats = self._encode_image_tensor(batch, return_patches=True)
        return feats                             # list[(B,N,1024)]

    def get_patch_tokens_for_fewshot_with_aug(self, image_paths: list[str]) -> list[torch.Tensor]:
        """
        Extract patch-level features from images using multiple rotation augmentations.

        Args:
            image_paths (list[str]): Paths to normal images

        Returns:
            list[torch.Tensor]: Augmented patch tokens with shape (B, len(angles)*N, D)
        """
        # Define rotation angles
        angles = [0, 45, 90, 135, 180, 225, 270, 315]

        # Load images as torch tensors: (B, C, H, W)
        imgs = load_and_transform_vision_data(image_paths, self.device)
        B, C, H, W = imgs.shape

        # Convert torch -> numpy for rotation (B, H, W, C)
        imgs_np = imgs.permute(0, 2, 3, 1).cpu().numpy()

        aug_list = []
        for angle in angles:
            rotated_batch = []
            for i in range(B):
                image = imgs_np[i]
                image_center = tuple(np.array(image.shape[1::-1]) / 2)
                rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
                result = cv2.warpAffine(
                    image,
                    rot_mat,
                    image.shape[1::-1],
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT_101,  # smoother borders
                )
                rotated_batch.append(result)
            rotated_batch = np.stack(rotated_batch)  # (B, H, W, C)
            aug_list.append(rotated_batch)

        # Stack augmentations: (len(angles), B, H, W, C)
        aug_np = np.stack(aug_list, axis=0)

        # Back to torch (len(angles)*B, C, H, W)
        aug = torch.from_numpy(aug_np).to(self.device).permute(0, 1, 4, 2, 3)
        aug = aug.reshape(len(angles) * B, C, H, W)

        # Encode with patches
        _, feats = self._encode_image_tensor(aug, return_patches=True)

        reshaped = []
        for f in feats:  # f: (len(angles)*B, N, D)
            f = f.reshape(B, len(angles), -1, self.visual_hidden_size)  # (B, A, N, D)
            reshaped.append(f.reshape(B, -1, self.visual_hidden_size))  # (B, A*N, D)

        return reshaped

    def forward(self, inputs: dict) -> tuple[torch.Tensor, float]:
        
        """
        Forward pass for pixel-level anomaly detection loss calculation.

        Args:
            inputs (dict): A dictionary containing:
                - 'images': List of image tensors
                - 'class_names': List of class name strings
                - 'masks': List of binary mask tensors (ground truth)

        Returns:
            tuple: (loss_pixel, dummy_value) where:
                - loss_pixel (torch.Tensor): Aggregated loss
                - dummy_value (float): Currently always 0.0
        """
        
        image_tensor = inputs['images']
        # self._save_debug_batch(
        #     images=image_tensor,
        #     masks=inputs.get('masks'),
        #     img_paths=inputs.get('img_paths') or inputs.get('image_paths'),
        #     class_names=inputs.get('class_names')
        # )
        patch_tokens = self.get_patch_tokens_from_tensor(image_tensor)
        class_name = inputs['class_names']
        loss_pixel = 0
        feats_text_tensor = self.encode_text_with_prompt_learnable(class_name)

        anomaly_maps = []
        for layer in range(len(patch_tokens)):
            patch_tokens[layer] = patch_tokens[layer] / patch_tokens[layer].norm(dim=-1, keepdim=True)
            feats_text_tensor = feats_text_tensor / feats_text_tensor.norm(dim=-1, keepdim=True)
            anomaly_map = 100.0 * patch_tokens[layer] @ feats_text_tensor.transpose(-2, -1)

            B, L, C = anomaly_map.shape
            H = int(np.sqrt(L))
            anomaly_map = F.interpolate(
                anomaly_map.permute(0, 2, 1).view(B, 2, H, H),
                size=self.image_size, mode='bilinear', align_corners=True
            )
            anomaly_map = torch.softmax(anomaly_map, dim=1)
            anomaly_maps.append(anomaly_map)

        gt = torch.stack(inputs['masks'], dim=0).to(self.device).squeeze()
        gt[gt > 0.3], gt[gt <= 0.3] = 1, 0

        for anomaly_map in anomaly_maps:
            f_loss = self.loss_focal(anomaly_map, gt)
            d_loss = self.loss_dice(anomaly_map[:, 1, :, :], gt)
            loss_pixel += f_loss + d_loss

        return loss_pixel, 0.0

    def _save_debug_batch(self,
                          images: list[torch.Tensor],
                          masks: Optional[list[torch.Tensor]],
                          img_paths: Optional[list[str]],
                          class_names: Optional[list[str]]) -> None:
        if not images or not masks:
            return
        save_root = self.args.get('save_path')
        if not save_root:
            return
        vis_dir = os.path.join(save_root, "batch_visualizations")
        os.makedirs(vis_dir, exist_ok=True)

        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        for idx, (img, mask) in enumerate(zip(images, masks)):
            img_cpu = img.detach().cpu()
            if img_cpu.dim() == 4:
                img_cpu = img_cpu[0]
            if img_cpu.size(0) == 3:
                img_cpu = img_cpu * std.to(img_cpu.dtype) + mean.to(img_cpu.dtype)
            img_np = img_cpu.clamp(0, 1).permute(1, 2, 0).numpy()

            mask_cpu = mask.detach().cpu()
            if mask_cpu.dim() > 2:
                mask_cpu = mask_cpu.squeeze(0)
            mask_cpu = (mask_cpu > 0.3).float()
            mask_np = mask_cpu.numpy()

            fig, axes = plt.subplots(1, 2, figsize=(8, 4))
            axes[0].imshow(img_np)
            axes[0].set_title("Image")
            axes[0].axis("off")

            axes[1].imshow(mask_np, cmap="gray")
            axes[1].set_title("Mask")
            axes[1].axis("off")

            if img_paths and idx < len(img_paths):
                base_name = os.path.splitext(os.path.basename(img_paths[idx]))[0]
            else:
                base_name = f"sample_{idx:04d}"

            cls_name = None
            if class_names and idx < len(class_names):
                cls_name = class_names[idx]
            if cls_name is None:
                cls_name = "unknown"
            cls_name = cls_name.replace(" ", "_")

            sample_type = "normal" if (idx % 2 == 0) else "anomaly"
            out_path = os.path.join(
                vis_dir,
                f"{cls_name}_{base_name}_{sample_type}.png"
            )
            fig.tight_layout()
            fig.savefig(out_path, dpi=150)
            plt.close(fig)

            self._debug_counter += 1

    def anomalymap_histogram(
        self,
        inputs: dict,
        anomaly_map,
        save_dir: str = "/mnt/lts5/scratch/basaran/PaLaDiN/code/anomalymap_histogram"
    ):
        """
        Save side-by-side visualization (448×448):
        1. Original image
        2. Original + anomaly overlay
        3. Histogram of anomaly map pixel values
        """
        import os, numpy as np, matplotlib.pyplot as plt
        from PIL import Image
        import torch

        os.makedirs(save_dir, exist_ok=True)

        # --- Load + resize image to 448 ---
        img_path = inputs["image_paths"][0]
        img_np = plt.imread(img_path)
        if img_np.dtype.kind == "f" and img_np.max() <= 1.0:
            img_np = (img_np * 255).astype(np.uint8)
        img_np = np.array(Image.fromarray(img_np).convert("RGB").resize((448, 448)))

        # --- Anomaly map (assume already 448×448) ---
        if isinstance(anomaly_map, torch.Tensor):
            am = anomaly_map.detach().cpu().numpy()
        else:
            am = np.array(anomaly_map)
        am = np.squeeze(am).astype(np.float32)
        amin, amax = am.min(), am.max()
        am = (am - amin) / (amax - amin) if amax > amin else np.zeros_like(am, dtype=np.float32)

        # --- Build save path ---
        parts = img_path.split(os.sep)
        dataset = parts[-5] 
        defect  = parts[-2] 
        stem    = os.path.splitext(parts[-1])[0]
        save_path = os.path.join(save_dir, f"{dataset}_{defect}_{stem}.png")

        # --- Plot ---
        plt.figure(figsize=(18, 6))

        # (1) Original
        ax1 = plt.subplot(1, 3, 1)
        ax1.imshow(img_np)
        ax1.set_title("Original")
        ax1.axis("off")

        # (2) Overlay
        ax2 = plt.subplot(1, 3, 2)
        ax2.imshow(img_np)
        ax2.imshow(am, cmap="jet", alpha=0.45)
        ax2.set_title("Overlay")
        ax2.axis("off")

        # (3) Histogram
        ax3 = plt.subplot(1, 3, 3)
        ax3.hist(am.ravel(), bins=50, color="red", alpha=0.75)
        ax3.set_title("Anomaly Map Histogram")
        ax3.set_xlabel("Pixel value")
        ax3.set_ylabel("Count")

        plt.tight_layout()
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()

        print(f"[Saved] {save_path}")


        
    def get_anomaly_map_anomal_score(self, inputs: dict, topk: bool = True) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        
        """
        Extracts pixel-level features combining image-text similarity and one-shot image comparison.

        Args:
            inputs (dict): Dictionary containing:
                - 'image_paths': List of image paths
                - 'normal_img_paths': List of corresponding normal image paths
            topk (bool): If True, compute a tail-average anomaly score from the final output

        Returns:
            tuple:
                - final_pixel_output (Tensor): Average of anomaly map and few-shot dissimilarity
                - fewshot_map (Tensor): 1 - similarity map
                - anomaly_score (Tensor | None): Tail-average anomaly score per image (B,)
        """      
        
        if inputs['image_paths']:
            
            parts = [p for p in os.path.normpath(inputs['image_paths'][0]).split(os.sep) if p]
            parts_lower = [p.lower() for p in parts]

            c_name = None
            if "data" in parts_lower:
                idx = parts_lower.index("data")
                if idx > 0:
                    c_name = parts[idx - 1]
            elif "mvtec" in parts_lower:
                idx = parts_lower.index("mvtec")
                if idx + 1 < len(parts):
                    c_name = parts[idx + 1]

            c_name = c_name.replace('_', ' ')
            if c_name not in prompt_sentences_clip_l14:
                c_name = 'object'

            feats_text_tensor = self.encode_text_with_prompt_learnable([c_name])
            
            patch_tokens, cls_raw = self.get_patch_tokens_from_path(inputs['image_paths'], return_cls=True)

            anomaly_maps = []
            for layer in range(len(patch_tokens)):
                patch_tokens[layer] = patch_tokens[layer] / patch_tokens[layer].norm(dim=-1, keepdim=True)
                feats_text_norm = feats_text_tensor / feats_text_tensor.norm(dim=-1, keepdim=True)
                
                anomaly_map = (20.0 * patch_tokens[layer] @ feats_text_norm.transpose(-2,-1))
                B, L, C = anomaly_map.shape
                H = int(np.sqrt(L))
                anomaly_map = F.interpolate(anomaly_map.permute(0, 2, 1).view(B, 2, H, H),
                                            size=self.image_size, mode='bilinear', align_corners=True)
                anomaly_map = torch.softmax(anomaly_map, dim=1)
                anomaly_maps.append(anomaly_map[:,1,:,:])

            zeroshot_map = torch.mean(torch.stack(anomaly_maps, dim=0), dim=0).unsqueeze(1)
            if inputs['normal_img_paths']:
                query_patch_tokens = self.get_patch_tokens_for_fewshot(inputs['image_paths'])
                normal_patch_tokens = self.get_patch_tokens_for_fewshot_with_aug(inputs['normal_img_paths'])

                sims = []

                # Optional knob to trade speed vs. memory (defaults to 2048 if not provided)
                chunk_size = int(self.args.get("similarity_chunk_size", 4096))

                for i in range(len(query_patch_tokens)):
                    # Shapes: (B, Nq, D) and (B, Nn, D). Typically B == 1 here.
                    q = query_patch_tokens[i]
                    n = normal_patch_tokens[i]

                    # Collapse batch if present; keep everything on the active CUDA device
                    q = q.reshape(-1, q.shape[-1])        # (Nq, D)
                    n = n.reshape(-1, n.shape[-1])        # (Nn, D)

                    # L2-normalize for cosine similarity
                    q = F.normalize(q, dim=1)
                    n = F.normalize(n, dim=1)

                    # Compute max similarity per query patch in chunks to control memory
                    # We avoid forming the full (Nq x Nn) matrix.
                    sim_max = None
                    for j in range(0, n.size(0), chunk_size):
                        n_chunk = n[j:j + chunk_size]                 # (chunk, D)
                        sim_chunk = torch.matmul(q, n_chunk.transpose(0, 1))  # (Nq, chunk)
                        # Max across this chunk
                        chunk_max = sim_chunk.max(dim=1).values       # (Nq,)
                        sim_max = chunk_max if sim_max is None else torch.maximum(sim_max, chunk_max)

                    sims.append(sim_max)  # (Nq,)

                sim = torch.mean(torch.stack(sims,dim=0), dim=0).reshape(1,1,28,28)
                sim_full = F.interpolate(sim,size=self.image_size, mode='bilinear', align_corners=True)

                fewshot_map = 1 - sim_full
                final_pixel_output = (zeroshot_map + fewshot_map) / 2
            else:
                fewshot_map = None
                final_pixel_output = zeroshot_map


            # ──────────────────────────────────────────────────────────────
            # Compute anomaly score using tail-average (ETaR / top‑k mean)
            # ─────────────────────────────────────────────────────────────
            if final_pixel_output is not None:
                flat = final_pixel_output.view(final_pixel_output.size(0), -1)
                if topk:
                    k = max(1, int(0.001 * flat.size(1)))
                    anomaly_score = torch.topk(flat, k, dim=1).values.mean(dim=1)
                else:
                    anomaly_score = flat.max(dim=1).values  # max pixel per image

        return final_pixel_output, anomaly_score

    def generate(self, inputs: dict, topk: bool = True) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        
        """
        Main entry point for generation from multimodal input (e.g., image, text, etc.).

        Args:
            inputs (dict): Dictionary containing:
                - 'image_paths': Optional[List[str]]
                - 'normal_img_paths': Optional[List[str]]
                - 'prompt': str
                - 'mode': str (unused here)
            topk (bool): If True, return tail-average anomaly score

        Returns:
            tuple:
                - torch.Tensor: Final anomaly map
                - Optional[torch.Tensor]: Tail-average anomaly score (B,) if topk=True
        """
        
        final_pixel_output, anomaly_score = self.get_anomaly_map_anomal_score(inputs, topk=topk)
        
        return final_pixel_output, anomaly_score
