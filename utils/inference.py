"""
inference.py
------------
Loads trained encoder + analysis heads and runs a full inference pass
on a single satellite image. Returns structured analysis results.

Usage (standalone test):
    python utils/inference.py --image path/to/satellite.jpg

Or import and call:
    from utils.inference import ArchaeologicalAnalyzer
    analyzer = ArchaeologicalAnalyzer()
    results  = analyzer.analyze(pil_image)
"""
import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import cv2
from PIL import Image
import matplotlib
matplotlib.use('Agg')   # headless backend
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from models.resnet_encoder import get_resnet_encoder
from models.analysis_heads import MultiTaskArchaeologist

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))


# ── Internal helper: encoder with intermediate feature outputs ────────────
import torch.nn as nn

class _EncoderWithFeatures(nn.Module):
    def __init__(self, base):
        super().__init__()
        resnet = base.model   # ResNetEncoder wraps torchvision ResNet18 under self.model
        self.stem   = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

    def forward(self, x):
        x  = self.stem(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return [x1, x2, x3, x4]


# ── Analyser class ─────────────────────────────────────────────────────────
class ArchaeologicalAnalyzer:
    """
    Loads checkpoints once, then accepts PIL images for analysis.
    """

    # Label map for segmentation classes
    SEG_LABELS   = {0: "Background", 1: "Archaeological Feature / Ruins", 2: "Vegetation"}
    SEG_COLORS   = {0: (30, 30, 80), 1: (220, 50, 50), 2: (50, 160, 50)}   # BGR for cv2 
    CATEGORY_COLORS = {0: np.array([30, 30, 80]),    # dark blue  – Background
                       1: np.array([220, 50, 50]),   # red        – Ruins
                       2: np.array([50, 160, 50])}   # green      – Vegetation

    def __init__(self,
                 byol_ckpt:     str = "models/checkpoints/ssl/byol_final.pth",
                 analysis_ckpt: str = "models/checkpoints/analysis/analysis_heads_final.pth",
                 img_size:      int = 224):

        self.img_size = img_size
        self.device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # ── Build encoder ──────────────────────────────────────────────────
        base = get_resnet_encoder(pretrained=False)
        if os.path.exists(byol_ckpt):
            state = torch.load(byol_ckpt, map_location='cpu')
            # BYOL checkpoint keys match ResNetEncoder directly (model.conv1.weight etc.)
            base.load_state_dict(state, strict=False)
            print(f"[Inference] Loaded encoder from {byol_ckpt}")
        else:
            print(f"[Inference] BYOL checkpoint not found at '{byol_ckpt}', using random weights.")

        self.encoder = _EncoderWithFeatures(base).to(self.device).eval()
        for p in self.encoder.parameters():
            p.requires_grad = False

        # ── Build analysis heads ───────────────────────────────────────────
        self.heads = MultiTaskArchaeologist().to(self.device).eval()
        if os.path.exists(analysis_ckpt):
            state = torch.load(analysis_ckpt, map_location='cpu')
            self.heads.load_state_dict(state, strict=False)
            print(f"[Inference] Loaded analysis heads from {analysis_ckpt}")
        else:
            print(f"[Inference] Analysis checkpoint not found at '{analysis_ckpt}', using random weights.")

        # ── Pre-processing pipeline ────────────────────────────────────────
        self.transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def analyze(self, pil_image: Image.Image) -> dict:
        orig_size = pil_image.size   # (W, H)
        tensor    = self.transform(pil_image).unsqueeze(0).to(self.device)

        # ── 1. Neural network pass ──────────────────────────────────────────
        features    = self.encoder(tensor)
        outputs     = self.heads(features)
        seg_logits  = outputs['segmentation']
        erosion_map = outputs['erosion']
        fault_map   = outputs['faults']

        seg_probs = F.softmax(seg_logits, dim=1)   # [1,3,H,W]
        total_px  = float(seg_probs.shape[2] * seg_probs.shape[3])

        nn_ruin      = float((seg_probs[0,1] > 0.38).float().sum()) / total_px
        nn_veg       = float((seg_probs[0,2] > 0.28).float().sum()) / total_px
        nn_ruin_mean = float(seg_probs[0,1].mean())
        nn_veg_mean  = float(seg_probs[0,2].mean())
        eros_np      = erosion_map[0,0].cpu().numpy()
        fault_np     = fault_map[0,0].cpu().numpy()
        nn_eros      = float((erosion_map[0,0] > 0.30).float().sum()) / total_px
        nn_fault     = float((fault_map[0,0]   > 0.22).float().sum()) / total_px

        # ── 2. Colour-space + channel setup ────────────────────────────────
        img_np  = np.array(pil_image.convert('RGB'))
        # Mild denoise before analysis (reduces false detections)
        img_smooth = cv2.GaussianBlur(img_np, (3, 3), 0)
        img_bgr = cv2.cvtColor(img_smooth, cv2.COLOR_RGB2BGR)
        img_hsv = cv2.cvtColor(img_bgr,    cv2.COLOR_BGR2HSV)
        img_lab = cv2.cvtColor(img_bgr,    cv2.COLOR_BGR2Lab)

        h_ch = img_hsv[:,:,0].astype(np.float32)
        s_ch = img_hsv[:,:,1].astype(np.float32)
        v_ch = img_hsv[:,:,2].astype(np.float32)
        r_f  = img_smooth[:,:,0].astype(np.float32)
        g_f  = img_smooth[:,:,1].astype(np.float32)
        b_f  = img_smooth[:,:,2].astype(np.float32)
        a_ch = img_lab[:,:,1].astype(np.float32)  # green(-) / red(+)
        bstar= img_lab[:,:,2].astype(np.float32)  # blue(-) / yellow(+)
        gray_u8 = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        gray_f  = gray_u8.astype(np.float32)
        total   = float(img_np.shape[0] * img_np.shape[1])

        def morph_clean(mask, k_open=5, k_close=9):
            m = cv2.morphologyEx(mask.astype(np.uint8),
                                 cv2.MORPH_OPEN,
                                 cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_open, k_open)))
            m = cv2.morphologyEx(m,
                                 cv2.MORPH_CLOSE,
                                 cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_close, k_close)))
            return m.astype(bool)

        def otsu_thresh(arr):
            """Return Otsu threshold on a float32 array normalised to uint8."""
            norm = cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            thr, _ = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return float(thr) / 255.0 * (arr.max() - arr.min()) + arr.min()

        # Shadow mask: very dark pixels are unreliable for any class
        shadow_mask = v_ch < 30  # HSV Value < 30/255 → shadow

        # ── 3. VEGETATION ──────────────────────────────────────────────────
        # ExG: Excess Green Index  2G-R-B  (positive = vegetation)
        rgb_sum = r_f + g_f + b_f + 1e-5
        r_n, g_n, b_n = r_f/rgb_sum, g_f/rgb_sum, b_f/rgb_sum
        ExG  = 2*g_n - r_n - b_n          # [-1, 2]  higher = greener
        ExR  = 1.4*r_n - g_n              # excess red (soil/urban indicator)
        ExGR = ExG - ExR                  # combined index: vegetation vs soil

        # CIVE (Colour Index of Vegetation Extraction)
        CIVE = 0.441*r_f - 0.811*g_f + 0.385*b_f + 18.78745

        # Lab a* channel: negative a* = green, positive = red
        veg_lab = a_ch < 128 - 8   # below neutral (128) = greenish

        # Thresholds via Otsu for ExGR
        exgr_thr = max(0.02, otsu_thresh(ExGR))
        veg_exgr  = ExGR > exgr_thr
        veg_cive  = CIVE < 0              # negative CIVE = vegetation
        veg_hsv   = (h_ch >= 30) & (h_ch <= 90) & (s_ch > 35) & (v_ch > 35)

        # Combine: majority vote across 4 independent signals
        veg_votes = veg_exgr.astype(np.uint8) + veg_cive.astype(np.uint8) + \
                    veg_lab.astype(np.uint8)   + veg_hsv.astype(np.uint8)
        veg_mask  = morph_clean(veg_votes >= 2, k_open=7, k_close=11)
        veg_mask  = veg_mask & ~shadow_mask
        cv_veg    = float(veg_mask.sum()) / total

        # ── 4. RUINS / STRUCTURES ──────────────────────────────────────────
        # Local Standard Deviation (texture richness)
        mean_f   = cv2.blur(gray_f, (9, 9))
        sq_mean  = cv2.blur(gray_f**2, (9, 9))
        local_sd = np.sqrt(np.clip(sq_mean - mean_f**2, 0, None))

        # Laplacian for sharpness (structured surfaces are sharp)
        lap = np.abs(cv2.Laplacian(gray_u8, cv2.CV_32F))
        lap_blur = cv2.blur(lap, (11, 11))

        # Edge density (Canny multi-threshold combined)
        edges1 = cv2.Canny(gray_u8, 20, 60)
        edges2 = cv2.Canny(gray_u8, 50, 150)
        edge_map = ((edges1.astype(np.uint8) | edges2.astype(np.uint8))
                       .astype(np.float32))
        edge_dens = cv2.blur(edge_map, (19, 19))

        # Material colour: stone/concrete/masonry
        stone_color  = (s_ch < 60) & (v_ch >= 55) & (v_ch < 215)          # gray/beige
        sandstone    = (h_ch >= 8) & (h_ch <=  30) & (s_ch < 90) & (v_ch >= 70)
        concrete_col = (s_ch < 25) & (v_ch >= 140)

        # Require material colour AND (high texture OR high edge density)
        sd_thr   = max(5.0, otsu_thresh(local_sd) * 0.6)
        edge_thr = max(0.04, otsu_thresh(edge_dens) * 0.6)
        struct   = (stone_color | sandstone | concrete_col) & \
                   ((local_sd > sd_thr) | (edge_dens > edge_thr) | (lap_blur > 8))
        struct_mask = morph_clean(struct, k_open=3, k_close=7)
        struct_mask = struct_mask & ~veg_mask & ~shadow_mask
        cv_ruin     = float(struct_mask.sum()) / total

        # ── 5. EROSION / BARE EARTH ────────────────────────────────────────
        # BSI alone mis-classifies gray concrete (roads) as bare earth.
        # Fix: require BOTH positive BSI AND a warm earthy hue (H=5-30)
        # so neutral grays are excluded entirely.
        BSI = (r_f + b_f - g_f) / (r_f + b_f + g_f + 1e-5)

        # Strict warm-hued soil colours: brown, red-brown, tan, orange
        warm_hue    = (h_ch >= 4) & (h_ch <= 30) & (s_ch > 18)   # warm tint
        sandy_warm  = (s_ch < 60) & (v_ch > 100) & (r_f > b_f + 8)   # warm pale sand
        # Positive BSI AND warm hue = actual bare earth, not road/concrete
        bsi_pos     = BSI > 0.04
        bare_mask   = morph_clean(
            (bsi_pos & warm_hue) | (warm_hue & (v_ch > 40)) | sandy_warm,
            k_open=5, k_close=9)
        bare_mask   = bare_mask & ~veg_mask & ~shadow_mask & ~struct_mask
        cv_eros     = float(bare_mask.sum()) / total

        # ── 6. WATER BODIES ────────────────────────────────────────────────
        # NDWI proxy: strongly negative = water
        NDWI_proxy = (g_f - b_f) / (g_f + b_f + 1e-5)
        # Hue: strict blue/cyan (H 92-130), HIGH saturation (>60)
        water_hsv  = (h_ch >= 92) & (h_ch <= 130) & (s_ch > 60) & (v_ch > 25) & (v_ch < 200)
        # Dark areas with STRONGLY blue-dominant channel
        water_dark = (b_f > r_f + 25) & (b_f > g_f + 15) & (v_ch < 100)
        # Lab b* well below neutral (128); <105 is clearly blue
        water_lab  = (bstar < 105) & (s_ch > 30) & (v_ch < 160)

        # Require 3 of 4 signals (strict majority) to mark as water
        water_votes = (NDWI_proxy < -0.12).astype(np.uint8) + \
                      water_hsv.astype(np.uint8) + \
                      water_dark.astype(np.uint8) + \
                      water_lab.astype(np.uint8)
        water_mask  = morph_clean(water_votes >= 3, k_open=7, k_close=13)
        water_mask  = water_mask & ~veg_mask & ~bare_mask & ~struct_mask
        cv_water    = float(water_mask.sum()) / total

        # ── 7. URBAN / BUILT-UP AREA ───────────────────────────────────────
        # Roads/buildings: low saturation + high edge density + not classified elsewhere
        urban_col    = (s_ch < 40) & (v_ch >= 70) & (v_ch <= 230)
        urban_struct_raw = morph_clean(
            urban_col & (edge_dens > max(0.05, edge_thr * 0.8)),
            k_open=3, k_close=5)
        urban_struct_raw = urban_struct_raw & ~struct_mask & ~veg_mask & ~water_mask
        cv_urban     = float(urban_struct_raw.sum()) / total

        # ── 8. FAULT / LINEAR FEATURES ─────────────────────────────────────
        # Probabilistic Hough Line Transform → count lines per area
        canny_fault = cv2.Canny(gray_u8, 40, 120, apertureSize=3)
        lines = cv2.HoughLinesP(canny_fault, rho=1, theta=np.pi/180,
                                 threshold=40, minLineLength=30, maxLineGap=10)
        fault_line_img = np.zeros_like(gray_u8, dtype=np.float32)
        if lines is not None:
            for ln in lines:
                x1,y1,x2,y2 = ln[0]
                cv2.line(fault_line_img, (x1,y1), (x2,y2), 1.0, 2)
        # Also use gradient magnitude for continuous fault map
        sobelx = cv2.Sobel(gray_u8, cv2.CV_32F, 1, 0, ksize=5)
        sobely = cv2.Sobel(gray_u8, cv2.CV_32F, 0, 1, ksize=5)
        grad_mag  = np.sqrt(sobelx**2 + sobely**2)
        grad_norm = grad_mag / (grad_mag.max() + 1e-6)
        # Hough density + gradient give complementary signals
        fault_cv_f = np.clip(0.6 * fault_line_img +
                              0.4 * (grad_norm > 0.25).astype(np.float32), 0, 1)
        fault_cv_f = cv2.blur(fault_cv_f, (5, 5))
        cv_fault   = float(fault_cv_f.mean()) * 0.8

        # ── 9. Blend NN + CV with calibrated weights ────────────────────────
        nn_w, cv_w = 0.30, 0.70

        ruin_probability = min(1.0, nn_w * max(nn_ruin, nn_ruin_mean*3.0) + cv_w * cv_ruin)
        raw_veg          = min(1.0, nn_w * max(nn_veg,  nn_veg_mean*3.0)  + cv_w * cv_veg)
        erosion_risk     = min(1.0, nn_w * nn_eros  + cv_w * cv_eros)
        fault_prob       = min(1.0, nn_w * nn_fault + cv_w * cv_fault)
        water_prob       = min(1.0, cv_water * 1.1)
        urban_prob       = min(1.0, cv_urban * 0.75)
        landslide_risk   = min(1.0, 0.40 * erosion_risk + 0.35 * fault_prob
                               + 0.15 * water_prob + 0.10 * ruin_probability)
        raw_bg           = max(0.0, 1.0 - ruin_probability - raw_veg
                               - water_prob - urban_prob - erosion_risk * 0.5)

        # ── 10. Build rich 6-class composite overlay ────────────────────────
        W, H = orig_size
        def rs(mask):
            return cv2.resize(mask.astype(np.uint8), (W, H),
                              interpolation=cv2.INTER_NEAREST).astype(bool)

        seg_rgb = np.zeros((H, W, 3), dtype=np.uint8)
        seg_rgb[:]                 = (15,  15,  40)   # deep background
        seg_rgb[rs(bare_mask)]     = (160, 100,  40)  # warm brown — bare/erosion
        seg_rgb[rs(urban_struct_raw)]=(190,190, 195)  # silver    — urban/roads
        seg_rgb[rs(water_mask)]    = ( 20,  90, 210)  # deep blue — water
        seg_rgb[rs(struct_mask)]   = (210,  45,  45)  # bright red — ruins
        seg_rgb[rs(veg_mask)]      = ( 30, 190,  55)  # bright green — vegetation

        seg_overlay = self._blend_overlay(pil_image, Image.fromarray(seg_rgb), alpha=0.48)

        # ── 11. Erosion heatmap ─────────────────────────────────────────────
        bare_f    = cv2.resize(bare_mask.astype(np.float32), (W, H))
        BSI_f     = cv2.resize(cv2.normalize(BSI, None, 0, 1, cv2.NORM_MINMAX), (W, H))
        eros_nn_f = cv2.resize(eros_np, (W, H))
        eros_comb = np.clip(0.45*bare_f + 0.30*BSI_f + 0.25*eros_nn_f, 0, 1)
        erosion_heatmap = self._apply_colormap(eros_comb, 'YlOrRd', orig_size)

        # ── 12. Fault map ───────────────────────────────────────────────────
        fault_cv_rs = cv2.resize(fault_cv_f, (W, H))
        fault_nn_rs = cv2.resize(fault_np,   (W, H))
        fault_comb  = np.clip(0.55*fault_cv_rs + 0.45*fault_nn_rs, 0, 1)
        fault_rgb   = self._apply_colormap(fault_comb, 'PuRd', orig_size)

        # ── 13. Summary ─────────────────────────────────────────────────────
        summary = self._build_summary(ruin_probability, erosion_risk, landslide_risk,
                                      fault_prob, raw_veg, water_prob, urban_prob)

        return {
            "ruin_probability":     ruin_probability,
            "erosion_risk":         erosion_risk,
            "landslide_risk":       landslide_risk,
            "fault_probability":    fault_prob,
            "water_probability":    water_prob,
            "urban_probability":    urban_prob,
            "segmentation_overlay": seg_overlay,
            "erosion_heatmap":      erosion_heatmap,
            "fault_mask":           fault_rgb,
            "risk_summary":         summary,
            "details": {
                "seg_class_probs": {
                    "Background":  raw_bg,
                    "Ruins/Walls": ruin_probability,
                    "Vegetation":  raw_veg,
                    "Water":       water_prob,
                    "Urban":       urban_prob,
                },
                "erosion_risk":   erosion_risk,
                "landslide_risk": landslide_risk,
                "fault_lines":    fault_prob,
            }
        }



    # ── Internal helpers ───────────────────────────────────────────────────
    @staticmethod
    def _blend_overlay(base: Image.Image, overlay: Image.Image, alpha: float) -> np.ndarray:
        base_arr    = np.array(base.convert('RGB')).astype(np.float32)
        overlay_arr = np.array(overlay).astype(np.float32)
        blended = (1 - alpha) * base_arr + alpha * overlay_arr
        return blended.clip(0, 255).astype(np.uint8)

    @staticmethod
    def _apply_colormap(arr: np.ndarray, cmap_name: str, target_size: tuple) -> np.ndarray:
        """Normalises arr to [0,1], applies a matplotlib colormap → RGB uint8."""
        vmin, vmax = arr.min(), arr.max()
        if vmax - vmin < 1e-6:
            arr_norm = np.zeros_like(arr)
        else:
            arr_norm = (arr - vmin) / (vmax - vmin)
        cmap    = cm.get_cmap(cmap_name)
        colored = (cmap(arr_norm)[:, :, :3] * 255).astype(np.uint8)   # [H,W,3]
        pil_cm  = Image.fromarray(colored).resize(target_size, Image.BILINEAR)
        return np.array(pil_cm)

    @staticmethod
    def _build_summary(ruin: float, erosion: float, landslide: float, fault: float,
                       veg: float = 0.0, water: float = 0.0, urban: float = 0.0) -> str:
        lines = ["🗺️  Archaeological & Hazard Analysis Report", "─" * 48]

        # Archaeological
        ruin_pct = ruin * 100
        if ruin_pct > 60:
            lines.append(f"🏛️  HIGH probability of archaeological features ({ruin_pct:.1f}%)")
            lines.append("    → Suggests ruins, walls, or ancient structures present.")
        elif ruin_pct > 30:
            lines.append(f"🏛️  MODERATE archaeological feature probability ({ruin_pct:.1f}%)")
            lines.append("    → Potential ruins or buried structures detected.")
        else:
            lines.append(f"🏛️  LOW archaeological feature probability ({ruin_pct:.1f}%)")

        # Vegetation
        veg_pct = veg * 100
        if veg_pct > 50:
            lines.append(f"🌿  HIGH vegetation cover ({veg_pct:.1f}%) — dense green canopy detected.")
        elif veg_pct > 20:
            lines.append(f"🌿  MODERATE vegetation cover ({veg_pct:.1f}%).")
        else:
            lines.append(f"🌿  SPARSE vegetation ({veg_pct:.1f}%) — mostly bare or built-up terrain.")

        # Erosion
        eros_pct = erosion * 100
        if eros_pct > 60:
            lines.append(f"🌊  HIGH erosion risk ({eros_pct:.1f}%) — bare/exposed terrain dominant.")
        elif eros_pct > 30:
            lines.append(f"🌊  MODERATE erosion risk ({eros_pct:.1f}%).")
        else:
            lines.append(f"🌊  LOW erosion risk ({eros_pct:.1f}%).")

        # Landslide
        ls_pct = landslide * 100
        if ls_pct > 60:
            lines.append(f"⛰️  HIGH landslide/subsidence risk ({ls_pct:.1f}%) — immediate caution advised.")
        elif ls_pct > 30:
            lines.append(f"⛰️  MODERATE landslide risk ({ls_pct:.1f}%).")
        else:
            lines.append(f"⛰️  LOW landslide risk ({ls_pct:.1f}%).")

        # Faults
        fault_pct = fault * 100
        if fault_pct > 50:
            lines.append(f"⚡  SIGNIFICANT land fault / discontinuity detected ({fault_pct:.1f}%).")
        elif fault_pct > 25:
            lines.append(f"⚡  MINOR fault lines possible ({fault_pct:.1f}%).")
        else:
            lines.append(f"⚡  No major land faults detected ({fault_pct:.1f}%).")

        # Water
        water_pct = water * 100
        if water_pct > 10:
            lines.append(f"💧  Water bodies detected — coverage {water_pct:.1f}%.")

        # Urban
        urban_pct = urban * 100
        if urban_pct > 10:
            lines.append(f"🏙️  Urban/built-up surfaces detected ({urban_pct:.1f}%) — roads or structures present.")

        return "\n".join(lines)


# ── CLI test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Archaeological Inference Test")
    parser.add_argument("--image", type=str, default=None,
                        help="Path to a satellite image (JPG/PNG)")
    args = parser.parse_args()

    analyzer = ArchaeologicalAnalyzer()

    if args.image and os.path.exists(args.image):
        img = Image.open(args.image).convert("RGB")
    else:
        # Use the first processed image as a demo
        processed_dir = "data/processed"
        samples = [f for f in os.listdir(processed_dir) if f.endswith('.jpg')] if os.path.isdir(processed_dir) else []
        if not samples:
            print("[Test] No processed images found. Run data/prepare_dataset.py first.")
            exit(1)
        img = Image.open(os.path.join(processed_dir, samples[0])).convert("RGB")
        print(f"[Test] Using sample image: {samples[0]}")

    results = analyzer.analyze(img)

    print("\n" + results["risk_summary"])
    print("\n── Raw scores ─────────────────────────────────────────────────")
    for k, v in results["details"].items():
        if isinstance(v, dict):
            for sub_k, sub_v in v.items():
                print(f"   {sub_k:<30}: {sub_v*100:.2f}%")
        else:
            print(f"   {k:<30}: {v*100:.2f}%")

    # Save overlay images to test_output/
    os.makedirs("test_output", exist_ok=True)
    Image.fromarray(results["segmentation_overlay"]).save("test_output/segmentation_overlay.jpg")
    Image.fromarray(results["erosion_heatmap"]).save("test_output/erosion_heatmap.jpg")
    Image.fromarray(results["fault_mask"]).save("test_output/fault_mask.jpg")
    print("\n[Test] Overlay images saved to test_output/")
