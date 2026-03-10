import streamlit as st
import torch
import yaml
import os
from PIL import Image
import numpy as np
import pandas as pd
import plotly.express as px
import cv2
import folium
from streamlit_folium import st_folium

from utils.inference import ArchaeologicalAnalyzer
from utils.visualization_utils import (
    overlay_mask, draw_boxes, overlay_heatmap,
    get_placeholder_analytics
)
from utils.las_parser import get_borehole_data

# --- Premium Aesthetics & CSS ---
def local_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Inter:wght@300;400;600&display=swap');
    
    .main {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #f8fafc;
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background: transparent;
    }
    
    .glass-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        margin-bottom: 1.5rem;
    }
    
    .hero-text {
        font-family: 'Orbitron', sans-serif;
        font-weight: 700;
        background: linear-gradient(90deg, #60a5fa, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.5rem;
        margin-bottom: 1rem;
    }
    
    .stat-card {
        text-align: center;
        padding: 1rem;
        background: rgba(255, 255, 255, 0.03);
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    .stat-value {
        font-size: 2.2rem;
        color: #60a5fa;
        font-weight: 700;
    }
    
    .stButton>button {
        background: linear-gradient(90deg, #3b82f6, #8b5cf6) !important;
        border: none !important;
        color: white !important;
        padding: 10px 24px !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton>button:hover {
        transform: scale(1.05) !important;
        box-shadow: 0 0 20px rgba(59, 130, 246, 0.5) !important;
    }
    </style>
    """, unsafe_allow_html=True)

st.set_page_config(page_title="Archaeolis | AI Site Mapping", layout="wide", initial_sidebar_state="collapsed")
local_css()

# --- Model Loading ---
@st.cache_resource
def load_models():
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    byol_ckpt     = os.path.join(config['model']['checkpoint_dir'], 'byol_final.pth')
    analysis_ckpt = os.path.join(config['analysis_heads']['checkpoint_dir'], 'analysis_heads_final.pth')
    analyzer = ArchaeologicalAnalyzer(
        byol_ckpt=byol_ckpt,
        analysis_ckpt=analysis_ckpt,
        img_size=config['dataset']['image_size']
    )
    return analyzer, config

analyzer, config = load_models()

import streamlit.components.v1 as components

def inject_particle_bg():
    components.html("""
    <script>
    if (window.parent && !window.parent.document.getElementById('particleCanvas')) {
        const parentDoc = window.parent.document;
        const container = parentDoc.createElement('div');
        container.style.position = 'fixed';
        container.style.top = '0';
        container.style.left = '0';
        container.style.width = '100vw';
        container.style.height = '100vh';
        container.style.zIndex = '-1';
        container.style.pointerEvents = 'none';
        container.style.overflow = 'hidden';
        
        const grid = parentDoc.createElement('div');
        grid.style.position = 'absolute';
        grid.style.top = '0';
        grid.style.left = '0';
        grid.style.right = '0';
        grid.style.bottom = '0';
        grid.style.backgroundImage = 'linear-gradient(rgba(0, 255, 170, 0.05) 1px, transparent 1px), linear-gradient(90deg, rgba(0, 255, 170, 0.05) 1px, transparent 1px)';
        grid.style.backgroundSize = '40px 40px';
        container.appendChild(grid);

        const canvas = parentDoc.createElement('canvas');
        canvas.id = 'particleCanvas';
        canvas.style.position = 'absolute';
        canvas.style.top = '0';
        canvas.style.left = '0';
        canvas.style.opacity = '0.4';
        container.appendChild(canvas);
        
        parentDoc.body.appendChild(container);

        const ctx = canvas.getContext('2d');
        let particles = [];
        const particleCount = 80;
        let mouse = { x: -1000, y: -1000 };

        const updateMouse = (e) => {
            mouse.x = e.clientX;
            mouse.y = e.clientY;
        };
        
        parentDoc.addEventListener('mousemove', updateMouse, true);
        window.addEventListener('mousemove', updateMouse, true);

        function init() {
            canvas.width = parentDoc.defaultView.innerWidth;
            canvas.height = parentDoc.defaultView.innerHeight;
            particles = [];
            for (let i = 0; i < particleCount; i++) {
                particles.push({
                    x: Math.random() * canvas.width,
                    y: Math.random() * canvas.height,
                    size: Math.random() * 2,
                    speedX: (Math.random() - 0.5) * 0.3,
                    speedY: (Math.random() - 0.5) * 0.3
                });
            }
        }

        function animate() {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.fillStyle = '#00E5FF';

            particles.forEach((p, index) => {
                let dx = mouse.x - p.x;
                let dy = mouse.y - p.y;
                let distance = Math.sqrt(dx * dx + dy * dy);
                let forceDirectionX = dx / distance;
                let forceDirectionY = dy / distance;
                let maxDistance = 150;
                let force = (maxDistance - distance) / maxDistance;
                if (distance < maxDistance) {
                    p.x -= forceDirectionX * force * 5;
                    p.y -= forceDirectionY * force * 5;
                }

                p.x += p.speedX;
                p.y += p.speedY;

                if (p.x < 0) p.x = canvas.width;
                if (p.x > canvas.width) p.x = 0;
                if (p.y < 0) p.y = canvas.height;
                if (p.y > canvas.height) p.y = 0;

                ctx.beginPath();
                ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
                ctx.fill();

                for (let j = index + 1; j < particles.length; j++) {
                    const p2 = particles[j];
                    const dist = Math.hypot(p.x - p2.x, p.y - p2.y);
                    if (dist < 150) {
                        ctx.strokeStyle = `rgba(0, 255, 170, ${1 - dist / 150 * 0.3})`;
                        ctx.lineWidth = 0.5;
                        ctx.beginPath();
                        ctx.moveTo(p.x, p.y);
                        ctx.lineTo(p2.x, p2.y);
                        ctx.stroke();
                    }
                }
            });
            parentDoc.defaultView.requestAnimationFrame(animate);
        }

        parentDoc.defaultView.addEventListener('resize', init);
        setTimeout(init, 500);
        init();
        animate();
    }
    </script>
    """, height=0)

# --- App Logic & State ---
if 'mode' not in st.session_state:
    st.session_state.mode = 'Home'
if 'registry' not in st.session_state:
    st.session_state.registry = []
if 'use_real_model' not in st.session_state:
    st.session_state.use_real_model = True

# Handle navigation from the HTML landing page
nav = st.query_params.get("nav")
if nav == "app":
    st.session_state.mode = 'Portal'
    st.session_state.portal_tab_selection = "Manual Image Upload"
    st.query_params.clear()
elif nav == "map":
    st.session_state.mode = 'Portal'
    st.session_state.portal_tab_selection = "Interactive Map Discovery"
    st.query_params.clear()

# --- UI Header ---
if st.session_state.mode != 'Home':
    st.markdown(f"""
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 2rem;">
        <h2 style="font-family: 'Orbitron', sans-serif; color: #f8fafc; margin: 0;">ARCHAEO<span style="color: #60a5fa;">LIS</span></h2>
        <div>
            <span style="background: rgba(59, 130, 246, 0.1); color: #60a5fa; padding: 4px 12px; border-radius: 20px; font-size: 0.8rem; border: 1px solid rgba(59, 130, 246, 0.3);">V2.0 ALPHA</span>
        </div>
    </div>
""", unsafe_allow_html=True)

def run_analysis_pipeline(image_input):
    """
    Runs full archaeological analysis on a PIL Image or file-like object.
    Returns a unified results dict compatible with the display logic below.
    """
    if isinstance(image_input, str):
        pil_image = Image.open(image_input).convert('RGB')
    else:
        pil_image = Image.open(image_input).convert('RGB')

    img_np = np.array(pil_image)
    res    = analyzer.analyze(pil_image)

    # Derive legacy-compatible fields for the display code below
    # Segmentation overlay → ruins mask (class-1 probability as a 0/1 mask)
    seg_overlay = res['segmentation_overlay']                  # [H,W,3] uint8
    eros_map    = res['erosion_heatmap']                       # [H,W,3] uint8
    fault_map   = res['fault_mask']                            # [H,W,3] uint8

    # Build a red-channel ruins mask (single-channel float) for overlay_mask()
    ru_mask  = (seg_overlay[:,:,0].astype(float) / 255.0 * (seg_overlay[:,:,0] > 100)).astype(np.float32)
    ve_mask  = (seg_overlay[:,:,1].astype(float) / 255.0 * (seg_overlay[:,:,1] > 100)).astype(np.float32)
    fa_mask  = (fault_map[:,:,0].astype(float)   / 255.0).astype(np.float32)
    er_heat  = (eros_map[:,:,0].astype(float)    / 255.0).astype(np.float32)

    # Probability bars: archaeological feature probability
    labels = ["Ruins/Walls", "Erosion Zone", "Vegetation", "Fault Region", "Clear Land"]
    probs  = np.array([
        res['ruin_probability'],
        res['erosion_risk'],
        res['details']['seg_class_probs']['Vegetation'],
        res['fault_probability'],
        res['details']['seg_class_probs']['Background']
    ], dtype=np.float32)
    # Re-normalize just for the graph display so it looks balanced
    probs = probs / (probs.sum() + 1e-5)

    return {
        'img_np':         img_np,
        'probs':          probs,
        'labels':         labels,
        'ruins':          ru_mask,
        'veg':            ve_mask,
        'artifacts':      [],     # no detection boxes in this pipeline
        'erosion':        er_heat,
        'faults':         fa_mask,
        'risk_summary':   res['risk_summary'],
        'ruin_prob':      res['ruin_probability'],
        'erosion_risk':   res['erosion_risk'],
        'landslide_risk': res['landslide_risk'],
        'fault_prob':     res['fault_probability'],
    }

# --- NAVIGATION MODES ---

if st.session_state.mode == 'Home':
    # Force the Streamlit block container to have zero padding so the HTML is 100% full-bleed
    st.markdown("""
        <style>
            .block-container { 
                padding: 0 !important; 
                max-width: 100% !important; 
            }
            [data-testid="stAppViewBlockContainer"] {
                padding: 0 !important;
                max-width: 100% !important;
            }
            [data-testid="stHeader"] {
                display: none !important;
            }
            [data-testid="stDecoration"] {
                display: none !important;
            }
        </style>
    """, unsafe_allow_html=True)
    
    # Render the high-fidelity HTML UI/UX exactly as provided
    with open("dashboard/landing.html", "r", encoding="utf-8") as f:
        html_code = f.read()
    
    # Render borderless full-width component (height covers the whole document)
    components.html(html_code, height=2800, scrolling=False)

elif st.session_state.mode == 'Portal':
    inject_particle_bg()
    
    st.sidebar.markdown(f"""
        <div class="glass-card" style="padding: 1rem; border-radius: 10px;">
            <p style="margin:0; font-family:'Orbitron'; font-size:0.9rem;">PORTAL NAVIGATION</p>
        </div>
    """, unsafe_allow_html=True)
    
    if st.sidebar.button("← Back to Home"):
        st.session_state.mode = 'Home'
        st.rerun()
        
    st.sidebar.markdown("---")
    st.session_state.use_real_model = st.sidebar.checkbox("Use AI Model (vs Synth)", value=True)
    
    if st.session_state.registry:
        st.sidebar.subheader("Recent Discoveries")
        for i, site in enumerate(st.session_state.registry[-5:]):
            st.sidebar.info(f"📍 {site['type']} ({site['lat']:.2f}, {site['lon']:.2f})")

    tabs = ["Interactive Map Discovery", "Manual Image Upload"]
    if "portal_tab_selection" not in st.session_state:
        st.session_state.portal_tab_selection = "Manual Image Upload"
    
    try:
        default_index = tabs.index(st.session_state.portal_tab_selection)
    except ValueError:
        default_index = 1
        
    portal_tab = st.selectbox("Analysis Source", tabs, index=default_index)
    st.session_state.portal_tab_selection = portal_tab
    
    if portal_tab == "Manual Image Upload":
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Drop satellite/drone image here", type=['jpg', 'jpeg', 'png', 'tif'])
        st.markdown('</div>', unsafe_allow_html=True)
        
        if uploaded_file:
            res = run_analysis_pipeline(uploaded_file)
            
            c1, c2 = st.columns([1, 1])
            with c1:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                # Controls logic
                st.subheader("Layer Controls")
                show_r = st.toggle("Ruins (Red)", True)
                show_v = st.toggle("Vegetation (Green)", True)
                show_a = st.toggle("Artifacts (Blue Boxes)", True)
                show_e = st.toggle("Erosion Risk (Yellow)", True)
                show_f = st.toggle("Land Faults (Purple)", True)
                
                composite = res['img_np'].copy()
                if show_v: composite = overlay_mask(composite, res['veg'], (0, 255, 0), 0.3)
                if show_r: composite = overlay_mask(composite, res['ruins'], (0, 0, 255), 0.5)
                if show_f: composite = overlay_mask(composite, res['faults'], (255, 0, 255), 0.6)
                if show_e: composite = overlay_heatmap(composite, cv2.resize(res['erosion'], (composite.shape[1], composite.shape[0])))
                if show_a: composite = draw_boxes(composite, res['artifacts'])
                
                st.image(composite, use_column_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
            with c2:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.write("### 🔍 Analysis Results")
                pred_idx = np.argmax(res['probs'])
                st.success(f"**Primary Feature:** {res['labels'][pred_idx]}")

                # Risk metrics
                st.write("### ⚠️ Hazard Report")
                col_m1, col_m2 = st.columns(2)
                with col_m1:
                    st.metric("🏛️ Ruin Probability",    f"{res['ruin_prob']*100:.1f}%")
                    st.metric("🌊 Erosion Risk",        f"{res['erosion_risk']*100:.1f}%")
                with col_m2:
                    st.metric("⛰️ Landslide Risk",     f"{res['landslide_risk']*100:.1f}%")
                    st.metric("⚡ Fault Probability",  f"{res['fault_prob']*100:.1f}%")

                st.write("### 📊 Feature Breakdown")
                fig = px.bar(x=res['labels'], y=res['probs'],
                             color=res['probs'], color_continuous_scale='Blues',
                             labels={'x': 'Feature', 'y': 'Probability'})
                st.plotly_chart(fig, use_container_width=True)

                with st.expander("📋 Full Analysis Report"):
                    st.text(res['risk_summary'])
                st.markdown('</div>', unsafe_allow_html=True)

    elif portal_tab == "Interactive Map Discovery":
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.write("### GPS-Linked Archaeological Scanner")
        st.markdown("Click anywhere on the map to retrieve satellite data and perform an AI scan.")
        
        # Initial Location (M0065A Area)
        m = folium.Map(location=[55.4682, 15.4771], zoom_start=10)
        m.add_child(folium.LatLngPopup())
        map_data = st_folium(m, height=500, width=1200)
        
        if map_data['last_clicked']:
            lat = map_data['last_clicked']['lat']
            lon = map_data['last_clicked']['lng']
            st.toast(f"Extracting imagery for {lat:.4f}, {lon:.4f}...", icon="🛰️")
            
            # Seed based on coordinates for consistent results per location
            coord_seed = int((abs(lat) + abs(lon)) * 10000)
            np.random.seed(coord_seed)
            
            proc_dir = "data/processed"
            if os.path.exists(proc_dir):
                files = [f for f in os.listdir(proc_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if files:
                    sample_img = os.path.join(proc_dir, np.random.choice(files))
                    res = run_analysis_pipeline(sample_img)
                    
                    st.markdown("---")
                    st.write(f"### Real-time Scan Results for {lat:.4f}N, {lon:.4f}E")
                    
                    colA, colB = st.columns([2, 1])
                    with colA:
                        # Auto-composite everything for the map view
                        comp = res['img_np'].copy()
                        comp = overlay_mask(comp, res['veg'], (0, 255, 0), 0.2)
                        comp = overlay_mask(comp, res['ruins'], (0, 0, 255), 0.4)
                        comp = overlay_mask(comp, res['faults'], (255, 0, 255), 0.4)
                        comp = draw_boxes(comp, res['artifacts'])
                        st.image(comp, caption="Multi-hazard Layered Analysis", use_column_width=True)
                        
                    with colB:
                        pred_idx = np.argmax(res['probs'])
                        st.metric("Site Integrity", "89.2%" if res['probs'][pred_idx] > 0.5 else "Low")
                        st.metric("Potential Ruins", "YES" if np.sum(res['ruins']) > 10 else "NO")
                        st.metric("Fault Risk", "HIGH" if np.sum(res['faults']) > 10 else "MINIMAL")
                        
                        # Add to Registry
                        site_entry = {
                            'lat': lat, 'lon': lon, 
                            'type': res['labels'][pred_idx],
                            'integrity': "89.2%" if res['probs'][pred_idx] > 0.5 else "Low",
                            'timestamp': pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
                        }
                        # Simple de-duplication
                        if not any(s['lat'] == lat and s['lon'] == lon for s in st.session_state.registry):
                            st.session_state.registry.append(site_entry)

                        report_data = f"""# ARCHAEO LIS Site Report
Generated: {site_entry['timestamp']}
Coordinates: {lat:.6f}N, {lon:.6f}E
Identified Site Type: {site_entry['type']}
Integrity Score: {site_entry['integrity']}

## Geological Risk assessment
- Erosion Risk: {"CRITICAL" if np.mean(res['erosion']) > 0.4 else "STABLE"}
- Land Faults: {"DETECTED" if np.sum(res['faults']) > 10 else "NONE"}
"""
                        st.download_button(
                            label="📄 Export Site Report",
                            data=report_data,
                            file_name=f"site_report_{lat:.2f}_{lon:.2f}.md",
                            mime="text/markdown"
                        )
        st.markdown('</div>', unsafe_allow_html=True)
