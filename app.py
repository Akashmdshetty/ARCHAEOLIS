"""
app.py  —  ARCHAEOLIS Flask Web Server
Serves index.html (landing) + portal (AI analysis) + /api/analyze endpoint.
Run: python app.py
"""

import os, sys, base64, io
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import yaml
import numpy as np
import cv2
from PIL import Image
from flask import Flask, request, jsonify, send_from_directory, render_template_string

from utils.inference import ArchaeologicalAnalyzer
from utils.visualization_utils import overlay_mask, draw_boxes, overlay_heatmap

# ── Init Flask ────────────────────────────────────────────────
app = Flask(__name__, static_folder='.', static_url_path='')

# ── Load model once ───────────────────────────────────────────
with open('configs/config.yaml') as f:
    config = yaml.safe_load(f)

byol_ckpt     = os.path.join(config['model']['checkpoint_dir'],         'byol_final.pth')
analysis_ckpt = os.path.join(config['analysis_heads']['checkpoint_dir'],'analysis_heads_final.pth')
analyzer = ArchaeologicalAnalyzer(byol_ckpt=byol_ckpt, analysis_ckpt=analysis_ckpt,
                                   img_size=config['dataset']['image_size'])
print("[ARCHAEOLIS] Model loaded.")

# ── Helper: img → base64 PNG ──────────────────────────────────
def to_b64(arr: np.ndarray) -> str:
    pil = Image.fromarray(arr.astype(np.uint8))
    buf = io.BytesIO()
    pil.save(buf, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()

# ─────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────

@app.route('/')
def landing():
    return send_from_directory('.', 'index.html')

@app.route('/portal')
def portal():
    return send_from_directory('.', 'portal.html')

@app.route('/results')
def results():
    return send_from_directory('.', 'results.html')

@app.route('/api/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file      = request.files['image']
    pil_image = Image.open(file.stream).convert('RGB')

    res = analyzer.analyze(pil_image)

    seg   = res['segmentation_overlay']
    eros  = res['erosion_heatmap']
    fault = res['fault_mask']

    # Build composite: seg overlay is the base, blend erosion + fault on top
    composite = seg.astype(np.float32).copy()
    h, w = composite.shape[:2]

    eros_rs = cv2.resize(eros, (w, h)) if eros.shape[:2] != (h, w) else eros
    eros_mask = (eros_rs.mean(axis=2) > 80).astype(np.float32)[..., None]
    composite = composite * (1 - 0.30 * eros_mask) + eros_rs.astype(np.float32) * 0.30 * eros_mask

    fault_rs = cv2.resize(fault, (w, h)) if fault.shape[:2] != (h, w) else fault
    fault_mask_arr = (fault_rs.mean(axis=2) > 80).astype(np.float32)[..., None]
    composite = composite * (1 - 0.25 * fault_mask_arr) + fault_rs.astype(np.float32) * 0.25 * fault_mask_arr

    composite = np.clip(composite, 0, 255).astype(np.uint8)

    labels = ["Ruins/Walls", "Erosion Zone", "Vegetation", "Fault Region", "Clear Land"]
    probs  = np.array([
        res['ruin_probability'], res['erosion_risk'],
        res['details']['seg_class_probs']['Vegetation'],
        res['fault_probability'], res['details']['seg_class_probs']['Background']
    ], dtype=np.float32)
    probs = probs / (probs.sum() + 1e-5)

    return jsonify({
        'composite':       to_b64(composite),
        'segmentation':    to_b64(seg),
        'erosion_heatmap': to_b64(eros),
        'ruin_prob':       float(res['ruin_probability']),
        'erosion_risk':    float(res['erosion_risk']),
        'landslide_risk':  float(res['landslide_risk']),
        'fault_prob':      float(res['fault_probability']),
        'veg_prob':        float(res['details']['seg_class_probs']['Vegetation']),
        'water_prob':      float(res.get('water_probability', 0)),
        'urban_prob':      float(res.get('urban_probability', 0)),
        'primary_feature': labels[int(np.argmax(probs))],
        'labels':          labels,
        'probs':           [float(p) for p in probs],
        'risk_summary':    res['risk_summary'],
    })

@app.route('/api/map_scan')
def map_scan():
    lat = float(request.args.get('lat', 55.47))
    lng = float(request.args.get('lng', 15.48))
    coord_seed = int((abs(lat) + abs(lng)) * 10000) % 10
    proc_dir   = 'data/processed'
    files = sorted([f for f in os.listdir(proc_dir) if f.endswith(('.jpg','.jpeg','.png'))])
    if not files:
        return jsonify({'error': 'No demo images in data/processed'}), 404
    img_path  = os.path.join(proc_dir, files[coord_seed % len(files)])
    pil_image = Image.open(img_path).convert('RGB')
    res       = analyzer.analyze(pil_image)

    seg   = res['segmentation_overlay']
    eros  = res['erosion_heatmap']
    fault = res['fault_mask']

    composite = seg.astype(np.float32).copy()
    h, w = composite.shape[:2]
    eros_rs = cv2.resize(eros, (w, h)) if eros.shape[:2] != (h, w) else eros
    eros_mask = (eros_rs.mean(axis=2) > 80).astype(np.float32)[..., None]
    composite = composite * (1 - 0.30 * eros_mask) + eros_rs.astype(np.float32) * 0.30 * eros_mask
    fault_rs = cv2.resize(fault, (w, h)) if fault.shape[:2] != (h, w) else fault
    fault_mask_arr = (fault_rs.mean(axis=2) > 80).astype(np.float32)[..., None]
    composite = composite * (1 - 0.25 * fault_mask_arr) + fault_rs.astype(np.float32) * 0.25 * fault_mask_arr
    composite = np.clip(composite, 0, 255).astype(np.uint8)

    labels = ["Ruins/Walls", "Erosion Zone", "Vegetation", "Fault Region", "Clear Land"]
    probs  = np.array([
        res['ruin_probability'], res['erosion_risk'],
        res['details']['seg_class_probs']['Vegetation'],
        res['fault_probability'], res['details']['seg_class_probs']['Background']
    ], dtype=np.float32)
    probs /= (probs.sum() + 1e-5)

    return jsonify({
        'composite':       to_b64(composite),
        'ruin_prob':       float(res['ruin_probability']),
        'erosion_risk':    float(res['erosion_risk']),
        'landslide_risk':  float(res['landslide_risk']),
        'fault_prob':      float(res['fault_probability']),
        'veg_prob':        float(res['details']['seg_class_probs']['Vegetation']),
        'water_prob':      float(res.get('water_probability', 0)),
        'urban_prob':      float(res.get('urban_probability', 0)),
        'primary_feature': labels[int(np.argmax(probs))],
        'labels':          labels,
        'probs':           [float(p) for p in probs],
        'risk_summary':    res['risk_summary'],
    })


if __name__ == '__main__':
    print("\n" + "="*55)
    print("  ARCHAEOLIS  |  AI-Driven Archaeological Site Mapping")
    print("  Landing  : http://localhost:5000")
    print("  Portal   : http://localhost:5000/portal")
    print("="*55 + "\n")
    app.run(debug=False, port=5000, host='0.0.0.0')
