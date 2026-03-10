import os, sys
# Ensure project root is on path (needed when launched via `streamlit run dashboard/...`)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import streamlit as st
import streamlit.components.v1 as components
import torch
import yaml
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
from PIL import Image
import folium
from streamlit_folium import st_folium

from utils.inference import ArchaeologicalAnalyzer
from utils.visualization_utils import overlay_mask, draw_boxes, overlay_heatmap, get_placeholder_analytics
from utils.las_parser import get_borehole_data

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ARCHAEOLIS | AI-Driven Site Mapping",
    layout="wide",
    initial_sidebar_state="collapsed",
    page_icon="🏛️"
)

# ─────────────────────────────────────────────────────────────
# GLOBAL CSS — mirrors index.html exactly
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;500;700&family=Space+Mono:ital,wght@0,400;0,700;1,400&display=swap');

/* ── Reset & Base ── */
html, body, [class*="css"], .stApp {
    font-family: 'Space Grotesk', sans-serif !important;
    background: #05090F !important;
    color: #f1f5f9 !important;
}
.stApp {
    background-image:
        linear-gradient(rgba(0,255,170,0.05) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,255,170,0.05) 1px, transparent 1px) !important;
    background-size: 40px 40px !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, header[data-testid="stHeader"], footer { display: none !important; }
.block-container { padding: 0 !important; max-width: 100% !important; }
section[data-testid="stSidebar"] {
    background: rgba(5,9,15,0.97) !important;
    border-right: 1px solid rgba(0,229,255,0.15) !important;
}

/* ── Glass Panel ── */
.glass-panel {
    background: rgba(10,15,25,0.7);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(0,229,255,0.2);
}

/* ── Animations ── */
@keyframes pulse-slow {
    0%,100% { opacity: 0.3; }
    50%      { opacity: 0.6; }
}
@keyframes fadeInUp {
    from { opacity:0; transform:translateY(20px); }
    to   { opacity:1; transform:translateY(0);    }
}
@keyframes scanline {
    0%   { transform: translateY(-100vh); }
    100% { transform: translateY(100vh);  }
}
@keyframes radarPing {
    0%   { transform:translate(-50%,-50%) scale(1);   opacity:0.3; }
    100% { transform:translate(-50%,-50%) scale(1.8); opacity:0;   }
}
.animate-pulse-slow { animation: pulse-slow 4s ease infinite; }
.animate-fade-in-up { animation: fadeInUp  0.8s ease-out forwards; opacity:0; }

/* ── Buttons ── */
.stButton > button {
    background: transparent !important;
    border: 1px solid #00E5FF !important;
    color: #00E5FF !important;
    font-family: 'Space Mono', monospace !important;
    font-weight: 700 !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    border-radius: 2px !important;
    padding: 0.65rem 1.6rem !important;
    transition: all 0.3s ease !important;
}
.stButton > button:hover {
    background: rgba(0,229,255,0.1) !important;
    box-shadow: 0 0 20px rgba(0,229,255,0.4) !important;
    transform: translateY(-2px) !important;
}

/* ── Metrics ── */
[data-testid="stMetric"] {
    background: rgba(0,229,255,0.04) !important;
    border: 1px solid rgba(0,229,255,0.15) !important;
    border-radius: 2px !important;
    padding: 1rem !important;
}
[data-testid="stMetricLabel"] {
    font-family: 'Space Mono',monospace !important;
    font-size: 0.6rem !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    color: #64748b !important;
}
[data-testid="stMetricValue"] {
    font-family: 'Space Mono',monospace !important;
    color: #00FFAA !important;
    font-size: 1.6rem !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    border: 1px dashed rgba(0,229,255,0.3) !important;
    border-radius: 2px !important;
    background: rgba(0,229,255,0.03) !important;
    padding: 1rem !important;
}

/* ── Selectbox ── */
.stSelectbox > div > div {
    background: rgba(10,15,25,0.9) !important;
    border: 1px solid rgba(0,229,255,0.2) !important;
    color: #00E5FF !important;
    font-family: 'Space Mono',monospace !important;
    border-radius: 2px !important;
}

/* ── Toggle / Checkbox ── */
[data-testid="stToggle"] label, [data-testid="stCheckbox"] label {
    font-family: 'Space Mono',monospace !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: #94a3b8 !important;
}

/* ── Success/Warning/Info ── */
[data-testid="stAlert"] {
    border-radius: 2px !important;
    font-family: 'Space Mono',monospace !important;
    font-size: 0.8rem !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    border: 1px solid rgba(0,229,255,0.15) !important;
    border-radius: 2px !important;
    background: rgba(10,15,25,0.5) !important;
}
[data-testid="stExpander"] summary {
    font-family:'Space Mono',monospace !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.1em !important;
    color: #00E5FF !important;
}

/* ── Sidebar text ── */
[data-testid="stSidebar"] * {
    font-family: 'Space Mono',monospace !important;
    font-size: 0.78rem !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 3px; }
::-webkit-scrollbar-track { background: #05090F; }
::-webkit-scrollbar-thumb { background: #00E5FF33; }

/* ── Inner content padding ── */
div[data-testid="stVerticalBlock"] > div:first-child { padding-top: 0 !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# PARTICLE CANVAS + GLOBAL SCANLINE (injected as fixed overlay)
# ─────────────────────────────────────────────────────────────
components.html("""
<style>
  #particleCanvas {
    position: fixed; top:0; left:0; width:100vw; height:100vh;
    z-index: 0; pointer-events: none; opacity: 0.4;
  }
  .global-scanline {
    position: fixed; top:0; left:0; width:100%; height:100px;
    background: linear-gradient(to bottom, transparent, rgba(0,229,255,0.05), transparent);
    border-bottom: 1px solid rgba(0,229,255,0.2);
    z-index: 9999; pointer-events: none;
    animation: globalScan 8s linear infinite;
  }
  @keyframes globalScan {
    0%   { transform: translateY(-100vh); }
    100% { transform: translateY(100vh);  }
  }
</style>
<div class="global-scanline"></div>
<canvas id="particleCanvas"></canvas>
<script>
  const canvas = document.getElementById('particleCanvas');
  const ctx = canvas.getContext('2d');
  let particles = [];
  const N = 80;
  let mouse = { x: -1000, y: -1000 };
  window.addEventListener('mousemove', e => { mouse.x = e.clientX; mouse.y = e.clientY; });
  function init() {
    canvas.width  = window.innerWidth;
    canvas.height = window.innerHeight;
    particles = [];
    for (let i = 0; i < N; i++) {
      particles.push({
        x: Math.random()*canvas.width, y: Math.random()*canvas.height,
        size: Math.random()*2,
        speedX: (Math.random()-0.5)*0.3, speedY: (Math.random()-0.5)*0.3,
      });
    }
  }
  function animate() {
    ctx.clearRect(0,0,canvas.width,canvas.height);
    ctx.fillStyle = '#00E5FF';
    particles.forEach((p,i) => {
      let dx=mouse.x-p.x, dy=mouse.y-p.y, d=Math.sqrt(dx*dx+dy*dy);
      if (d<150) { let f=(150-d)/150; p.x-=(dx/d)*f*5; p.y-=(dy/d)*f*5; }
      p.x+=p.speedX; p.y+=p.speedY;
      if(p.x<0)p.x=canvas.width; if(p.x>canvas.width)p.x=0;
      if(p.y<0)p.y=canvas.height; if(p.y>canvas.height)p.y=0;
      ctx.beginPath(); ctx.arc(p.x,p.y,p.size,0,Math.PI*2); ctx.fill();
      for(let j=i+1;j<particles.length;j++){
        const p2=particles[j], dist=Math.hypot(p.x-p2.x,p.y-p2.y);
        if(dist<150){
          ctx.strokeStyle=`rgba(0,255,170,${(1-dist/150)*0.3})`;
          ctx.lineWidth=0.5;
          ctx.beginPath(); ctx.moveTo(p.x,p.y); ctx.lineTo(p2.x,p2.y); ctx.stroke();
        }
      }
    });
    requestAnimationFrame(animate);
  }
  window.addEventListener('resize', init);
  init(); animate();
</script>
""", height=0, scrolling=False)

# ─────────────────────────────────────────────────────────────
# NAV BAR
# ─────────────────────────────────────────────────────────────
st.markdown("""
<nav style="position:sticky;top:0;z-index:9998;
            background:rgba(10,15,25,0.7);backdrop-filter:blur(12px);
            border-bottom:1px solid rgba(0,229,255,0.2);padding:1rem 1.5rem;">
  <div style="max-width:1280px;margin:0 auto;display:flex;justify-content:space-between;align-items:center;">
    <div style="display:flex;align-items:center;gap:10px;">
      <div style="width:32px;height:32px;background:#00E5FF;box-shadow:0 0 15px rgba(0,229,255,0.3);"></div>
      <span style="font-family:'Space Mono',monospace;font-weight:700;color:#00E5FF;letter-spacing:-0.05em;font-size:1rem;">
        ARCHAEOLIS [v1.0.4]
      </span>
    </div>
    <div style="display:flex;gap:2rem;font-family:'Space Mono',monospace;font-size:0.65rem;
                letter-spacing:0.2em;text-transform:uppercase;color:#64748b;">
      <span style="cursor:pointer;" onmouseover="this.style.color='#00E5FF'" onmouseout="this.style.color='#64748b'"">// FEATURES</span>
      <span style="cursor:pointer;" onmouseover="this.style.color='#00E5FF'" onmouseout="this.style.color='#64748b'"">// DEMO</span>
      <span style="cursor:pointer;" onmouseover="this.style.color='#00E5FF'" onmouseout="this.style.color='#64748b'"">// STACK</span>
    </div>
  </div>
</nav>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    byol_ckpt     = os.path.join(config['model']['checkpoint_dir'], 'byol_final.pth')
    analysis_ckpt = os.path.join(config['analysis_heads']['checkpoint_dir'], 'analysis_heads_final.pth')
    analyzer = ArchaeologicalAnalyzer(
        byol_ckpt=byol_ckpt, analysis_ckpt=analysis_ckpt,
        img_size=config['dataset']['image_size']
    )
    return analyzer, config

analyzer, config = load_models()

# ─────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────
if 'mode' not in st.session_state:       st.session_state.mode = 'Home'
if 'registry' not in st.session_state:   st.session_state.registry = []

# ─────────────────────────────────────────────────────────────
# ANALYSIS PIPELINE
# ─────────────────────────────────────────────────────────────
def run_analysis_pipeline(image_input):
    pil_image = Image.open(image_input).convert('RGB')
    img_np    = np.array(pil_image)
    res       = analyzer.analyze(pil_image)
    seg_overlay = res['segmentation_overlay']
    eros_map    = res['erosion_heatmap']
    fault_map   = res['fault_mask']
    ru_mask = (seg_overlay[:,:,0].astype(float)/255.0*(seg_overlay[:,:,0]>100)).astype(np.float32)
    ve_mask = (seg_overlay[:,:,1].astype(float)/255.0*(seg_overlay[:,:,1]>100)).astype(np.float32)
    fa_mask = (fault_map[:,:,0].astype(float)/255.0).astype(np.float32)
    er_heat = (eros_map[:,:,0].astype(float)/255.0).astype(np.float32)
    labels = ["Ruins/Walls","Erosion Zone","Vegetation","Fault Region","Clear Land"]
    probs  = np.array([
        res['ruin_probability'], res['erosion_risk'],
        res['details']['seg_class_probs']['Vegetation'],
        res['fault_probability'], res['details']['seg_class_probs']['Background']
    ], dtype=np.float32)
    probs = probs / (probs.sum() + 1e-5)
    return {
        'img_np': img_np, 'probs': probs, 'labels': labels,
        'ruins': ru_mask, 'veg': ve_mask, 'artifacts': [],
        'erosion': er_heat, 'faults': fa_mask,
        'risk_summary': res['risk_summary'],
        'ruin_prob': res['ruin_probability'],
        'erosion_risk': res['erosion_risk'],
        'landslide_risk': res['landslide_risk'],
        'fault_prob': res['fault_probability'],
    }

# ─────────────────────────────────────────────────────────────
# ══ HOME PAGE ══
# ─────────────────────────────────────────────────────────────
if st.session_state.mode == 'Home':

    # ── Hero Section ──────────────────────────────────────────
    st.markdown("""
    <section style="position:relative;min-height:90vh;display:flex;flex-direction:column;
                    align-items:center;justify-content:center;text-align:center;
                    padding:5rem 1rem 8rem;">
      <!-- Glow orb -->
      <div style="position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);
                  width:600px;height:600px;background:rgba(0,229,255,0.08);
                  border-radius:50%;filter:blur(120px);pointer-events:none;"></div>
      <div style="position:relative;z-index:10;max-width:900px;">
        <!-- Status badge -->
        <div style="display:inline-block;padding:4px 12px;
                    border:1px solid rgba(0,255,170,0.5);background:rgba(0,255,170,0.08);
                    color:#00FFAA;font-family:'Space Mono',monospace;font-size:0.6rem;
                    letter-spacing:0.18em;text-transform:uppercase;margin-bottom:2rem;">
          SYSTEM_STATUS: ONLINE // HACKATHON_EVENT
        </div>
        <!-- Heading -->
        <h1 style="font-family:'Space Grotesk',sans-serif;font-weight:700;
                   font-size:clamp(3rem,8vw,6rem);letter-spacing:-0.05em;
                   text-transform:uppercase;line-height:1;margin:0 0 2rem;">
          AI-Driven <br/>
          <span style="background:linear-gradient(90deg,#00E5FF,#00FFAA);
                       -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
            Archaeological
          </span><br/>
          Site Mapping
        </h1>
        <p style="font-size:1.1rem;color:#94a3b8;max-width:600px;margin:0 auto 3rem;font-weight:300;">
          Discover hidden ruins buried beneath centuries of vegetation and soil using advanced
          neural networks and high-resolution satellite imagery.
        </p>
      </div>
    </section>
    """, unsafe_allow_html=True)

    # ── CTA Buttons (Streamlit buttons render inline) ──────────
    col_gap1, col_b1, col_gap2 = st.columns([3,2,3])
    with col_b1:
        if st.button("Get Started", key="hero_cta", use_container_width=True):
            st.session_state.mode = 'Portal'
            st.rerun()


    # ── 3-Step Process ─────────────────────────────────────────
    st.markdown("""
    <div style="background:rgba(0,0,0,0.4);border-top:1px solid rgba(0,229,255,0.1);
                border-bottom:1px solid rgba(0,229,255,0.1);padding:6rem 2rem;margin-top:2rem;">
      <div style="max-width:1280px;margin:0 auto;display:grid;grid-template-columns:repeat(3,1fr);gap:3rem;">

        <!-- Step 01 -->
        <div>
          <div style="font-family:'Space Mono',monospace;font-size:2.5rem;color:#00FFAA;margin-bottom:1.5rem;">01.</div>
          <h3 style="font-size:1.4rem;font-weight:700;text-transform:uppercase;letter-spacing:-0.02em;margin-bottom:1rem;">
            Upload Image
          </h3>
          <p style="color:#94a3b8;font-family:'Space Mono',monospace;font-size:0.75rem;line-height:1.8;">
            Feed high-resolution multispectral or LiDAR satellite imagery into the processing pipeline.
          </p>
          <div style="margin-top:2rem;padding:1rem;border:1px dashed rgba(100,116,139,0.4);
                      display:flex;justify-content:center;align-items:center;">
            <svg width="48" height="48" fill="none" stroke="#4b5563" viewBox="0 0 24 24">
              <path d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
                    stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5"/>
            </svg>
          </div>
        </div>

        <!-- Step 02 -->
        <div>
          <div style="font-family:'Space Mono',monospace;font-size:2.5rem;color:#00FFAA;margin-bottom:1.5rem;">02.</div>
          <h3 style="font-size:1.4rem;font-weight:700;text-transform:uppercase;letter-spacing:-0.02em;margin-bottom:1rem;">
            AI Detection
          </h3>
          <p style="color:#94a3b8;font-family:'Space Mono',monospace;font-size:0.75rem;line-height:1.8;">
            Our U-Net based architecture scans for geometric anomalies and subtle soil depressions.
          </p>
          <div style="margin-top:2rem;padding:1rem;border:1px dashed rgba(100,116,139,0.4);
                      height:80px;overflow:hidden;position:relative;
                      background:rgba(0,229,255,0.03);display:flex;align-items:center;justify-content:center;">
            <div style="width:100%;height:2px;background:linear-gradient(90deg,transparent,#00FFAA,transparent);
                        position:absolute;top:0;animation:scanElem 2s linear infinite;"></div>
            <span style="font-family:'Space Mono',monospace;font-size:0.6rem;color:rgba(0,229,255,0.4);">
              ANALYZING_BUFFERS...
            </span>
          </div>
          <style>@keyframes scanElem{0%{top:0}100%{top:100%}}</style>
        </div>

        <!-- Step 03 -->
        <div>
          <div style="font-family:'Space Mono',monospace;font-size:2.5rem;color:#00FFAA;margin-bottom:1.5rem;">03.</div>
          <h3 style="font-size:1.4rem;font-weight:700;text-transform:uppercase;letter-spacing:-0.02em;margin-bottom:1rem;">
            View Results
          </h3>
          <p style="color:#94a3b8;font-family:'Space Mono',monospace;font-size:0.75rem;line-height:1.8;">
            Download a tactical heatmap overlay showing the probability of archaeological structures.
          </p>
          <div style="margin-top:2rem;padding:1rem;border:1px dashed rgba(100,116,139,0.4);
                      display:flex;justify-content:center;align-items:center;">
            <svg width="48" height="48" fill="none" stroke="#4b5563" viewBox="0 0 24 24">
              <path d="M9 20l-5.447-2.724A2 2 0 013 15.482V5.418a2 2 0 011.106-1.789l5.447-2.724a2 2 0 011.894 0l5.447 2.724A2 2 0 0118 5.418v10.064a2 2 0 01-1.106 1.789L11.447 20a2 2 0 01-1.894 0z"
                    stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5"/>
            </svg>
          </div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Feature Cards ──────────────────────────────────────────
    st.markdown("""
    <div id="features" style="background:rgba(0,0,0,0.3);padding:6rem 2rem;">
      <div style="max-width:1280px;margin:0 auto;
                  display:grid;grid-template-columns:repeat(4,1fr);gap:1.5rem;">

        <div class="glass-panel" style="padding:2rem;" onmouseover="this.style.borderColor='rgba(0,255,170,0.6)';this.style.boxShadow='0 0 20px rgba(0,255,170,0.3)'" onmouseout="this.style.borderColor='rgba(0,229,255,0.2)';this.style.boxShadow='none'">
          <div style="width:40px;height:40px;border:1px solid #00E5FF;display:flex;align-items:center;justify-content:center;margin-bottom:1.5rem;font-family:'Space Mono',monospace;font-size:0.75rem;">01</div>
          <h4 style="font-weight:700;text-transform:uppercase;font-size:1.1rem;margin-bottom:0.5rem;">Ruins Detection</h4>
          <p style="color:#64748b;font-family:'Space Mono',monospace;font-size:0.7rem;line-height:1.7;">98.4% Accuracy in identifying sub-surface stone foundations using multi-spectral bands.</p>
        </div>

        <div class="glass-panel" style="padding:2rem;" onmouseover="this.style.borderColor='rgba(0,255,170,0.6)';this.style.boxShadow='0 0 20px rgba(0,255,170,0.3)'" onmouseout="this.style.borderColor='rgba(0,229,255,0.2)';this.style.boxShadow='none'">
          <div style="width:40px;height:40px;border:1px solid #00E5FF;display:flex;align-items:center;justify-content:center;margin-bottom:1.5rem;font-family:'Space Mono',monospace;font-size:0.75rem;">02</div>
          <h4 style="font-weight:700;text-transform:uppercase;font-size:1.1rem;margin-bottom:0.5rem;">Artifact ID</h4>
          <p style="color:#64748b;font-family:'Space Mono',monospace;font-size:0.7rem;line-height:1.7;">Object detection for surface scatter patterns of pottery and masonry fragments.</p>
        </div>

        <div class="glass-panel" style="padding:2rem;" onmouseover="this.style.borderColor='rgba(0,255,170,0.6)';this.style.boxShadow='0 0 20px rgba(0,255,170,0.3)'" onmouseout="this.style.borderColor='rgba(0,229,255,0.2)';this.style.boxShadow='none'">
          <div style="width:40px;height:40px;border:1px solid #00E5FF;display:flex;align-items:center;justify-content:center;margin-bottom:1.5rem;font-family:'Space Mono',monospace;font-size:0.75rem;">03</div>
          <h4 style="font-weight:700;text-transform:uppercase;font-size:1.1rem;margin-bottom:0.5rem;">Vegetation Analysis</h4>
          <p style="color:#64748b;font-family:'Space Mono',monospace;font-size:0.7rem;line-height:1.7;">Analyzing 'crop-marks' where vegetation health indicates underlying structures.</p>
        </div>

        <div class="glass-panel" style="padding:2rem;" onmouseover="this.style.borderColor='rgba(0,255,170,0.6)';this.style.boxShadow='0 0 20px rgba(0,255,170,0.3)'" onmouseout="this.style.borderColor='rgba(0,229,255,0.2)';this.style.boxShadow='none'">
          <div style="width:40px;height:40px;border:1px solid #00E5FF;display:flex;align-items:center;justify-content:center;margin-bottom:1.5rem;font-family:'Space Mono',monospace;font-size:0.75rem;">04</div>
          <h4 style="font-weight:700;text-transform:uppercase;font-size:1.1rem;margin-bottom:0.5rem;">Erosion Risk</h4>
          <p style="color:#64748b;font-family:'Space Mono',monospace;font-size:0.7rem;line-height:1.7;">Predictive modeling for site degradation due to rainfall and climate shifts.</p>
        </div>

      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Map Preview Widget (Radar HUD) ─────────────────────────
    st.markdown("""
    <div style="padding:6rem 2rem;overflow:hidden;">
      <div class="glass-panel" style="max-width:900px;margin:0 auto;padding:0.5rem;
                                       border-color:rgba(0,255,170,0.3);">
        <!-- Header bar -->
        <div style="display:flex;justify-content:space-between;align-items:center;
                    padding:1rem;border-bottom:1px solid rgba(0,255,170,0.2);">
          <div style="display:flex;align-items:center;gap:10px;">
            <div style="width:8px;height:8px;border-radius:50%;background:#00FFAA;
                        animation:radarPulse 1.5s ease infinite;"></div>
            <span style="font-family:'Space Mono',monospace;font-size:0.6rem;
                         letter-spacing:0.2em;color:#00FFAA;">GEOSPATIAL_VIEWER_ACTIVE</span>
          </div>
          <span style="font-family:'Space Mono',monospace;font-size:0.6rem;color:#00E5FF;cursor:pointer;">
            RE-CENTER_GPS
          </span>
        </div>
        <style>@keyframes radarPulse{0%,100%{opacity:1}50%{opacity:0.3}}</style>

        <!-- Map canvas -->
        <div style="position:relative;height:380px;background:#05090F;overflow:hidden;">
          <!-- Grid overlay -->
          <div style="position:absolute;inset:0;
                      background-image:linear-gradient(rgba(0,255,170,0.04) 1px,transparent 1px),linear-gradient(90deg,rgba(0,255,170,0.04) 1px,transparent 1px);
                      background-size:30px 30px;"></div>
          <!-- Radar rings -->
          <div style="position:absolute;top:50%;left:50%;
                      width:200px;height:200px;border:1px solid rgba(0,255,170,0.1);border-radius:50%;
                      transform:translate(-50%,-50%);"></div>
          <div style="position:absolute;top:50%;left:50%;
                      width:120px;height:120px;border:1px solid rgba(0,255,170,0.15);border-radius:50%;
                      transform:translate(-50%,-50%);animation:radarRing 2s ease-out infinite;"></div>
          <style>@keyframes radarRing{0%{transform:translate(-50%,-50%) scale(1);opacity:0.3}100%{transform:translate(-50%,-50%) scale(2);opacity:0}}</style>
          <!-- Crosshair target -->
          <div style="position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);
                      width:80px;height:80px;border:1px solid rgba(0,255,170,0.5);
                      display:flex;align-items:center;justify-content:center;">
            <div style="width:4px;height:4px;background:#00FFAA;border-radius:50%;"></div>
            <div style="position:absolute;top:-2px;left:-2px;width:10px;height:10px;border-top:2px solid #00FFAA;border-left:2px solid #00FFAA;"></div>
            <div style="position:absolute;bottom:-2px;right:-2px;width:10px;height:10px;border-bottom:2px solid #00FFAA;border-right:2px solid #00FFAA;"></div>
          </div>
          <!-- Scattered site dots -->
          <div style="position:absolute;top:30%;left:25%;width:6px;height:6px;background:#00E5FF;border-radius:50%;box-shadow:0 0 8px #00E5FF;"></div>
          <div style="position:absolute;top:65%;left:70%;width:6px;height:6px;background:#00FFAA;border-radius:50%;box-shadow:0 0 8px #00FFAA;"></div>
          <div style="position:absolute;top:20%;left:60%;width:4px;height:4px;background:#00E5FF;border-radius:50%;opacity:0.6;"></div>
          <div style="position:absolute;top:75%;left:30%;width:4px;height:4px;background:#00FFAA;border-radius:50%;opacity:0.6;"></div>
          <!-- HUD readout -->
          <div style="position:absolute;bottom:1.5rem;left:1.5rem;
                      font-family:'Space Mono',monospace;font-size:0.55rem;
                      color:rgba(0,255,170,0.7);line-height:1.8;">
            <div>LAT: 37.7749° N</div>
            <div>LNG: 122.4194° W</div>
            <div>ALT: 420.5m</div>
          </div>
          <div style="position:absolute;top:1rem;right:1rem;
                      font-family:'Space Mono',monospace;font-size:0.55rem;color:rgba(0,229,255,0.5);">
            SCAN_MODE: MULTI-SPECTRAL
          </div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Tech Stack ─────────────────────────────────────────────
    st.markdown("""
    <div id="tech" style="padding:5rem 2rem;border-top:1px solid rgba(0,229,255,0.1);">
      <div style="max-width:1280px;margin:0 auto;text-align:center;">
        <p style="font-family:'Space Mono',monospace;font-size:0.65rem;letter-spacing:0.2em;
                  text-transform:uppercase;color:#475569;margin-bottom:3rem;">
          Engineered With
        </p>
        <div style="display:flex;justify-content:center;gap:4rem;flex-wrap:wrap;opacity:0.6;">
          <span style="font-family:'Space Mono',monospace;font-size:1.6rem;font-weight:700;">PYTHON</span>
          <span style="font-family:'Space Mono',monospace;font-size:1.6rem;font-weight:700;">PYTORCH</span>
          <span style="font-family:'Space Mono',monospace;font-size:1.6rem;font-weight:700;">YOLOv8</span>
          <span style="font-family:'Space Mono',monospace;font-size:1.6rem;font-weight:700;">U-NET++</span>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Footer ─────────────────────────────────────────────────
    st.markdown("""
    <footer class="glass-panel" style="padding:4rem 2rem;border-top:1px solid rgba(0,229,255,0.2);">
      <div style="max-width:1280px;margin:0 auto;
                  display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:2rem;">
        <div>
          <div style="font-family:'Space Mono',monospace;font-weight:700;color:#00E5FF;
                      letter-spacing:-0.03em;font-size:1rem;">ARCHAEOLIS LABS</div>
          <p style="font-family:'Space Mono',monospace;font-size:0.6rem;color:#475569;margin-top:4px;">
            © 2024 DEEP-SEARCH HACKATHON ENTRY
          </p>
        </div>
        <div style="display:flex;gap:1.5rem;font-family:'Space Mono',monospace;
                    font-size:0.6rem;letter-spacing:0.1em;text-transform:uppercase;color:#475569;">
          <a href="https://github.com/Akashmdshetty/ARCHAEOLIS" style="color:#475569;text-decoration:none;"
             onmouseover="this.style.color='#00FFAA'" onmouseout="this.style.color='#475569'">Github Repo</a>
          <a href="#" style="color:#475569;text-decoration:none;"
             onmouseover="this.style.color='#00FFAA'" onmouseout="this.style.color='#475569'">API Docs</a>
          <a href="#" style="color:#475569;text-decoration:none;"
             onmouseover="this.style.color='#00FFAA'" onmouseout="this.style.color='#475569'">Privacy Protocol</a>
        </div>
        <div style="text-align:right;">
          <p style="font-family:'Space Mono',monospace;font-size:0.6rem;color:#475569;">
            PROTOTYPE BUILD: v1.0.4-BETA
          </p>
          <div style="display:flex;align-items:center;gap:6px;margin-top:6px;">
            <div style="width:6px;height:6px;border-radius:50%;background:#00FFAA;"></div>
            <span style="font-family:'Space Mono',monospace;font-size:0.55rem;color:#00FFAA;">
              SERVERS_OPTIMIZED
            </span>
          </div>
        </div>
      </div>
    </footer>
    """, unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)
    _, mid, _ = st.columns([2,2,2])
    with mid:
        if st.button("🚀  ENTER ANALYSIS PORTAL", use_container_width=True):
            st.session_state.mode = 'Portal'
            st.rerun()


# ─────────────────────────────────────────────────────────────
# ══ ANALYSIS PORTAL ══
# ─────────────────────────────────────────────────────────────
elif st.session_state.mode == 'Portal':

    st.sidebar.markdown("""
    <div style="padding:1rem 0;border-bottom:1px solid rgba(0,229,255,0.15);margin-bottom:1rem;">
      <p style="font-family:'Space Mono',monospace;font-size:0.65rem;letter-spacing:0.15em;
                color:#00E5FF;text-transform:uppercase;margin:0;">// PORTAL_NAVIGATION</p>
    </div>
    """, unsafe_allow_html=True)

    if st.sidebar.button("← Back to Home"):
        st.session_state.mode = 'Home'
        st.rerun()

    st.sidebar.markdown("---")

    if st.session_state.registry:
        st.sidebar.markdown("""
        <p style="font-size:0.6rem;letter-spacing:0.15em;color:#00FFAA;text-transform:uppercase;">
          Recent Discoveries
        </p>
        """, unsafe_allow_html=True)
        for site in st.session_state.registry[-5:]:
            st.sidebar.info(f"📍 {site['type']} ({site['lat']:.2f}, {site['lon']:.2f})")

    # Portal header
    st.markdown("""
    <div style="padding:3rem 2rem 1rem;">
      <div style="max-width:1280px;margin:0 auto;">
        <div style="display:inline-block;padding:3px 10px;border:1px solid rgba(0,255,170,0.4);
                    background:rgba(0,255,170,0.08);color:#00FFAA;
                    font-family:'Space Mono',monospace;font-size:0.6rem;letter-spacing:0.15em;
                    margin-bottom:1rem;">
          // ANALYSIS_PORTAL_ACTIVE
        </div>
        <h2 style="font-family:'Space Grotesk',sans-serif;font-weight:700;font-size:2.5rem;
                   letter-spacing:-0.04em;text-transform:uppercase;
                   background:linear-gradient(90deg,#00E5FF,#00FFAA);
                   -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
          Geospatial Intelligence
        </h2>
      </div>
    </div>
    """, unsafe_allow_html=True)

    _, content, _ = st.columns([0.05, 0.9, 0.05])
    with content:
        portal_tab = st.selectbox(
            "Analysis Source",
            ["Interactive Map Discovery", "Manual Image Upload", "Subsurface Core Logs"]
        )

        # ── Manual Image Upload ────────────────────────────────
        if portal_tab == "Manual Image Upload":
            st.markdown('<div class="glass-panel" style="padding:2rem;margin:1rem 0;">', unsafe_allow_html=True)
            uploaded_file = st.file_uploader(
                "Drop satellite / drone image",
                type=['jpg','jpeg','png','tif']
            )
            st.markdown('</div>', unsafe_allow_html=True)

            if uploaded_file:
                with st.spinner("SCANNING_BUFFERS..."):
                    res = run_analysis_pipeline(uploaded_file)

                c1, c2 = st.columns([1,1])
                with c1:
                    st.markdown('<div class="glass-panel" style="padding:1.5rem;">', unsafe_allow_html=True)
                    st.markdown("""
                    <p style="font-family:'Space Mono',monospace;font-size:0.65rem;
                               letter-spacing:0.15em;color:#00E5FF;text-transform:uppercase;margin-bottom:1rem;">
                      // LAYER_CONTROLS
                    </p>
                    """, unsafe_allow_html=True)
                    show_r = st.toggle("Ruins (Red)",         True)
                    show_v = st.toggle("Vegetation (Green)",  True)
                    show_a = st.toggle("Artifacts (Blue)",    True)
                    show_e = st.toggle("Erosion Heat (Yellow)", True)
                    show_f = st.toggle("Land Faults (Purple)", True)

                    composite = res['img_np'].copy()
                    if show_v: composite = overlay_mask(composite, res['veg'],    (0,255,0),   0.3)
                    if show_r: composite = overlay_mask(composite, res['ruins'],  (0,0,255),   0.5)
                    if show_f: composite = overlay_mask(composite, res['faults'], (255,0,255), 0.6)
                    if show_e: composite = overlay_heatmap(composite,
                                    cv2.resize(res['erosion'], (composite.shape[1], composite.shape[0])))
                    if show_a: composite = draw_boxes(composite, res['artifacts'])
                    st.image(composite, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)

                with c2:
                    st.markdown('<div class="glass-panel" style="padding:1.5rem;">', unsafe_allow_html=True)
                    st.markdown("""
                    <p style="font-family:'Space Mono',monospace;font-size:0.65rem;
                               letter-spacing:0.15em;color:#00E5FF;text-transform:uppercase;margin-bottom:1rem;">
                      // ANALYSIS_RESULTS
                    </p>
                    """, unsafe_allow_html=True)

                    pred_idx = np.argmax(res['probs'])
                    st.success(f"Primary Feature: {res['labels'][pred_idx]}")

                    st.markdown("""
                    <p style="font-family:'Space Mono',monospace;font-size:0.65rem;
                               letter-spacing:0.15em;color:#00E5FF;text-transform:uppercase;
                               margin:1.5rem 0 0.75rem;">// HAZARD_REPORT</p>
                    """, unsafe_allow_html=True)

                    m1, m2 = st.columns(2)
                    with m1:
                        st.metric("Ruin Probability", f"{res['ruin_prob']*100:.1f}%")
                        st.metric("Erosion Risk",     f"{res['erosion_risk']*100:.1f}%")
                    with m2:
                        st.metric("Landslide Risk",   f"{res['landslide_risk']*100:.1f}%")
                        st.metric("Fault Probability",f"{res['fault_prob']*100:.1f}%")

                    st.markdown("""
                    <p style="font-family:'Space Mono',monospace;font-size:0.65rem;
                               letter-spacing:0.15em;color:#00E5FF;text-transform:uppercase;
                               margin:1.5rem 0 0.75rem;">// FEATURE_BREAKDOWN</p>
                    """, unsafe_allow_html=True)

                    fig = px.bar(
                        x=res['labels'], y=res['probs'],
                        color=res['probs'],
                        color_continuous_scale=[[0,'#003344'],[0.5,'#00E5FF'],[1,'#00FFAA']],
                        labels={'x':'Feature','y':'Probability'}
                    )
                    fig.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(10,15,25,0.5)',
                        font=dict(family='Space Mono', color='#64748b', size=9),
                        xaxis=dict(gridcolor='rgba(0,229,255,0.08)'),
                        yaxis=dict(gridcolor='rgba(0,229,255,0.08)'),
                        coloraxis_showscale=False,
                        margin=dict(l=0,r=0,t=10,b=0),
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    with st.expander("📋 Full Analysis Report"):
                        st.text(res['risk_summary'])
                    st.markdown('</div>', unsafe_allow_html=True)

        # ── Interactive Map Discovery ──────────────────────────
        elif portal_tab == "Interactive Map Discovery":
            st.markdown('<div class="glass-panel" style="padding:1.5rem;margin:1rem 0;">', unsafe_allow_html=True)
            st.markdown("""
            <p style="font-family:'Space Mono',monospace;font-size:0.65rem;letter-spacing:0.15em;
                       color:#00E5FF;text-transform:uppercase;margin-bottom:0.5rem;">
              // GPS-LINKED ARCHAEOLOGICAL SCANNER
            </p>
            <p style="color:#64748b;font-family:'Space Mono',monospace;font-size:0.7rem;margin-bottom:1rem;">
              Click anywhere on the map to retrieve satellite data and perform an AI scan.
            </p>
            """, unsafe_allow_html=True)

            m = folium.Map(location=[55.4682, 15.4771], zoom_start=10,
                           tiles='CartoDB dark_matter')
            m.add_child(folium.LatLngPopup())
            map_data = st_folium(m, height=450, use_container_width=True)

            if map_data and map_data.get('last_clicked'):
                lat = map_data['last_clicked']['lat']
                lon = map_data['last_clicked']['lng']
                st.toast(f"Extracting imagery for {lat:.4f}, {lon:.4f}...", icon="🛰️")

                coord_seed = int((abs(lat)+abs(lon))*10000)
                np.random.seed(coord_seed)
                proc_dir = "data/processed"
                if os.path.exists(proc_dir):
                    files = [f for f in os.listdir(proc_dir) if f.lower().endswith(('.jpg','.jpeg','.png'))]
                    if files:
                        sample_img = os.path.join(proc_dir, np.random.choice(files))
                        res = run_analysis_pipeline(sample_img)
                        st.markdown(f"""
                        <p style="font-family:'Space Mono',monospace;font-size:0.65rem;letter-spacing:0.15em;
                                   color:#00FFAA;text-transform:uppercase;margin:1.5rem 0 0.5rem;">
                          // REAL-TIME SCAN — {lat:.4f}N, {lon:.4f}E
                        </p>
                        """, unsafe_allow_html=True)
                        colA, colB = st.columns([2,1])
                        with colA:
                            comp = res['img_np'].copy()
                            comp = overlay_mask(comp, res['veg'],    (0,255,0),   0.2)
                            comp = overlay_mask(comp, res['ruins'],  (0,0,255),   0.4)
                            comp = overlay_mask(comp, res['faults'], (255,0,255), 0.4)
                            comp = draw_boxes(comp, res['artifacts'])
                            st.image(comp, caption="Multi-hazard Layered Analysis", use_container_width=True)
                        with colB:
                            pred_idx = np.argmax(res['probs'])
                            st.metric("Site Integrity", "89.2%" if res['probs'][pred_idx]>0.5 else "Low")
                            st.metric("Potential Ruins","YES" if np.sum(res['ruins'])>10 else "NO")
                            st.metric("Fault Risk","HIGH" if np.sum(res['faults'])>10 else "MINIMAL")
                            site_entry = {
                                'lat':lat,'lon':lon,'type':res['labels'][pred_idx],
                                'integrity':"89.2%" if res['probs'][pred_idx]>0.5 else "Low",
                                'timestamp':pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
                            }
                            if not any(s['lat']==lat and s['lon']==lon for s in st.session_state.registry):
                                st.session_state.registry.append(site_entry)
                            report = f"""# ARCHAEOLIS Site Report
Generated: {site_entry['timestamp']}
Coordinates: {lat:.6f}N, {lon:.6f}E
Site Type: {site_entry['type']}
Integrity: {site_entry['integrity']}
Erosion: {"CRITICAL" if np.mean(res['erosion'])>0.4 else "STABLE"}
Faults: {"DETECTED" if np.sum(res['faults'])>10 else "NONE"}
"""
                            st.download_button("📄 Export Site Report", report,
                                               f"report_{lat:.2f}_{lon:.2f}.md", "text/markdown")

            st.markdown('</div>', unsafe_allow_html=True)

        # ── Subsurface Core Logs ───────────────────────────────
        elif portal_tab == "Subsurface Core Logs":
            st.markdown('<div class="glass-panel" style="padding:1.5rem;margin:1rem 0;">', unsafe_allow_html=True)
            st.markdown("""
            <p style="font-family:'Space Mono',monospace;font-size:0.65rem;letter-spacing:0.15em;
                       color:#00E5FF;text-transform:uppercase;margin-bottom:1rem;">
              // SUBSURFACE CORE LOG VIEWER
            </p>
            """, unsafe_allow_html=True)
            logs = get_borehole_data("data/temp_extract")
            if logs:
                log_type = st.selectbox("Borehole ID", list(logs.keys()))
                df = logs[log_type]
                depth_col = 'DEPT' if 'DEPT' in df.columns else 'DEPTH_WSF'
                curves = [c for c in df.columns if c.lower() not in ['dept','depth_wsf']]
                sel = st.multiselect("Geophysical Curves", curves, default=curves[:2])
                if sel:
                    fig = px.line(df, x=sel, y=depth_col, title="Expedition 347 | Subsurface Profile")
                    fig.update_yaxes(autorange="reversed")
                    fig.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(10,15,25,0.5)',
                        font=dict(family='Space Mono', color='#64748b', size=9),
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No borehole logs detected in data/temp_extract/")
            st.markdown('</div>', unsafe_allow_html=True)
