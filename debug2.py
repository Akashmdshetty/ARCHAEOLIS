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
    if not imgs:
        print("ERROR: no processed images found")
        sys.exit(1)

    img = Image.open(imgs[0]).convert('RGB')
    print(f"Image: {imgs[0]}, size={img.size}")

    res = analyzer.analyze(img)

    print("\n=== RESULTS ===")
    print(f"  Ruin prob:      {res['ruin_probability']*100:.2f}%")
    print(f"  Vegetation:     {res['details']['seg_class_probs']['Vegetation']*100:.2f}%")
    print(f"  Erosion risk:   {res['erosion_risk']*100:.2f}%")
    print(f"  Fault prob:     {res['fault_probability']*100:.2f}%")
    print(f"  Water:          {res.get('water_probability',0)*100:.2f}%")
    print(f"  Urban:          {res.get('urban_probability',0)*100:.2f}%")
    print(f"  Landslide:      {res['landslide_risk']*100:.2f}%")
    print("\nSummary:", res['risk_summary'][:200])
except Exception:
    print("EXCEPTION:")
    traceback.print_exc()
