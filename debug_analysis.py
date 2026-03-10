import sys, os
sys.path.insert(0, '.')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import yaml, glob
import numpy as np
from PIL import Image
from utils.inference import ArchaeologicalAnalyzer

with open('configs/config.yaml') as f:
    config = yaml.safe_load(f)

byol = os.path.join(config['model']['checkpoint_dir'], 'byol_final.pth')
anal = os.path.join(config['analysis_heads']['checkpoint_dir'], 'analysis_heads_final.pth')

analyzer = ArchaeologicalAnalyzer(byol_ckpt=byol, analysis_ckpt=anal,
                                   img_size=config['dataset']['image_size'])

imgs = glob.glob('data/processed/*.jpg')
img = Image.open(imgs[0]).convert('RGB')
print(f'Image: {imgs[0]}  size={img.size}')

res = analyzer.analyze(img)

print('\n=== SCALAR OUTPUTS ===')
print(f'  Ruin probability : {res["ruin_probability"]*100:.2f}%')
print(f'  Erosion risk     : {res["erosion_risk"]*100:.2f}%')
print(f'  Landslide risk   : {res["landslide_risk"]*100:.2f}%')
print(f'  Fault probability: {res["fault_probability"]*100:.2f}%')

print('\n=== SEG CLASS PROBS ===')
for k, v in res['details']['seg_class_probs'].items():
    print(f'  {k:<30}: {v*100:.2f}%')

seg = res['segmentation_overlay']
print(f'\nSeg overlay  shape={seg.shape}  dtype={seg.dtype}  min={seg.min()}  max={seg.max()}')

flat = seg.reshape(-1, 3)
unique, counts = np.unique(flat[::50], axis=0, return_counts=True)
top = sorted(zip([tuple(u) for u in unique], counts.tolist()), key=lambda x: -x[1])[:12]
print('Dominant colors in seg overlay:')
for c, n in top:
    print(f'  RGB{c}  count={n}')
