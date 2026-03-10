import sys, os, traceback
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.path.insert(0, '.')

try:
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
    with open('RESULT.txt', 'w', encoding='ascii', errors='replace') as f:
        f.write(f"Image: {imgs[0]}, size={img.size}\n")
        sys.stdout = f
        res = analyzer.analyze(img)
        sys.stdout = sys.__stdout__
        f.write(f"Ruin:     {res['ruin_probability']*100:.2f}%\n")
        f.write(f"Veg:      {res['details']['seg_class_probs']['Vegetation']*100:.2f}%\n")
        f.write(f"Erosion:  {res['erosion_risk']*100:.2f}%\n")
        f.write(f"Fault:    {res['fault_probability']*100:.2f}%\n")
        f.write(f"Water:    {res.get('water_probability',0)*100:.2f}%\n")
        f.write(f"Urban:    {res.get('urban_probability',0)*100:.2f}%\n")
        f.write(f"Landslide:{res['landslide_risk']*100:.2f}%\n")
        seg = res['segmentation_overlay']
        f.write(f"Seg overlay shape:{seg.shape} min:{seg.min()} max:{seg.max()}\n")

    print("Done. Check RESULT.txt")

except Exception:
    sys.stdout = sys.__stdout__
    traceback.print_exc()
    with open('RESULT.txt', 'w') as f:
        traceback.print_exc(file=f)
