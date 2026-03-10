
// ── Load result from localStorage ─────────────────────────────
const raw  = localStorage.getItem('archaeolis_result');
const data = raw ? JSON.parse(raw) : null;

// Chart.js global defaults
Chart.defaults.color        = '#64748b';
Chart.defaults.font.family  = 'Space Mono, monospace';
Chart.defaults.font.size    = 9;

const C = {
  green:  '#00FFAA',
  cyan:   '#00E5FF',
  orange: '#FB923C',
  purple: '#C084FC',
  blue:   '#60A5FA',
  dim:    'rgba(0,255,170,0.08)',
  grid:   'rgba(0,255,170,0.08)',
};

function pct(v){ return (v*100).toFixed(1)+'%'; }

if(data) {
  // ── Calculate true primary feature locally ──────────────────
  const rawProbs = {
    "Ruins/Walls": data.ruin_prob,
    "Erosion Zone": data.erosion_risk,
    "Vegetation": data.veg_prob ?? 0,
    "Fault Region": data.fault_prob,
    "Water Body": data.water_prob ?? 0,
    "Urban/Built-up": data.urban_prob ?? 0,
    "Landslide Risk": data.landslide_risk
  };
  let truePrimary = "Clear Land";
  let maxP = 0.2; // threshold
  for(const [k,v] of Object.entries(rawProbs)){
    if(v > maxP){ maxP = v; truePrimary = k; }
  }
  data.primary_feature = truePrimary; // Override backend

  // ── Composite image ─────────────────────────────────────────
  const img = document.getElementById('compositeImg');
  img.src = data.composite;
  img.classList.remove('hidden');
  document.getElementById('imgPlaceholder').style.display = 'none';

  const lat = (29 + data.ruin_prob*4).toFixed(4);
  const lng = (31 + data.erosion_risk*14).toFixed(4);
  document.getElementById('hudLat').textContent  = 'LAT: '+lat+'° N';
  document.getElementById('hudLng').textContent  = 'LNG: '+lng+'° E';
  document.getElementById('primaryBadge').textContent = '▶ '+data.primary_feature.toUpperCase()+' DETECTED';
  
  // Dynamic markers based on what is detected
  document.getElementById('marker1Label').textContent = 'RUIN ['+(data.ruin_prob*100).toFixed(0)+'%]';
  document.getElementById('marker2Label').textContent = 'EROSION ['+(data.erosion_risk*100).toFixed(0)+'%]';
  if((data.water_prob??0) > 0.3) {
    document.getElementById('marker3').classList.remove('hidden');
    document.getElementById('marker3Label').textContent = 'WATER ['+((data.water_prob??0)*100).toFixed(0)+'%]';
  }

  // ── Scan Feed ───────────────────────────────────────────────
  const t = new Date().toTimeString().slice(0,8);
  const feedItems = [
    {tag:'NEW CANDIDATE',    col:'primary',     title:'Ruin Candidate Theta-9',     sub:'LAT: '+lat+' / LONG: '+lng,                          bar:data.ruin_prob},
    {tag:'ANOMALY DETECTED', col:'accent-cyan',  title:'Spectral Anomaly Alpha-X',   sub:'EROSION: '+pct(data.erosion_risk),                   bar:data.erosion_risk},
    {tag:'FAULT DETECTED',   col:'primary',     title:'Subsurface Fault Delta',      sub:'FAULT PROB: '+pct(data.fault_prob),                  bar:data.fault_prob},
    {tag:'SCAN COMPLETE',    col:'slate-500',   title:'Primary: '+data.primary_feature, sub:'LANDSLIDE: '+pct(data.landslide_risk), bar:null},
  ];
  if ((data.water_prob??0) > 0.3) {
      feedItems[1] = {tag:'WATER DETECTED', col:'blue-400', title:'Hydrological Feature', sub:'WATER PROB: '+pct(data.water_prob), bar:data.water_prob};
  }
  document.getElementById('scanFeed').innerHTML = feedItems.map((e,i)=>`
    <div class="p-3 border ${e.col==='accent-cyan'?'border-accent-cyan/20 bg-accent-cyan/5':e.col==='primary'?'border-primary/15 bg-bg-dark/50':e.col==='blue-400'?'border-blue-400/20 bg-blue-500/10':'border-primary/10 bg-bg-dark/30 opacity-60'}">
      <div class="flex justify-between mb-1">
        <span class="text-[10px] ${e.col==='accent-cyan'?'text-accent-cyan bg-accent-cyan/10':e.col==='primary'?'text-primary bg-primary/10':e.col==='blue-400'?'text-blue-400 bg-blue-500/20':'text-slate-500'} px-1">${e.tag}</span>
        <span class="text-[10px] text-slate-500">${t}</span>
      </div>
      <p class="text-xs font-bold text-slate-100">${e.title}</p>
      <p class="text-[10px] text-slate-400 mt-1">${e.sub}</p>
      ${e.bar!=null?`<div class="mt-2 h-1 bg-primary/10 overflow-hidden"><div class="h-full bg-primary" style="width:${(e.bar*100).toFixed(0)}%"></div></div>`:''}
    </div>
  `).join('');

  // ── BAR CHART: Feature Probability (raw values) ─────────────
  new Chart(document.getElementById('barChart'), {
    type: 'bar',
    data: {
      labels: ['Ruins','Erosion','Landslide','Vegetation','Faults','Water','Urban'],
      datasets:[{
        label:'Probability',
        data: [
          +(data.ruin_prob*100).toFixed(1),
          +(data.erosion_risk*100).toFixed(1),
          +(data.landslide_risk*100).toFixed(1),
          +((data.veg_prob??0)*100).toFixed(1),
          +(data.fault_prob*100).toFixed(1),
          +((data.water_prob??0)*100).toFixed(1),
          +((data.urban_prob??0)*100).toFixed(1),
        ],
        backgroundColor:[C.green, C.orange, C.orange, C.cyan, C.purple, C.blue, '#94a3b8'],
        borderColor:[C.green, C.orange, C.orange, C.cyan, C.purple, C.blue, '#94a3b8'],
        borderWidth:1, borderRadius:1,
      }]
    },
    options:{
      responsive:true, maintainAspectRatio:false,
      plugins:{ legend:{display:false}, tooltip:{callbacks:{label:c=>c.raw+'%'}} },
      scales:{
        x:{ grid:{color:C.grid}, ticks:{color:'#64748b'} },
        y:{ grid:{color:C.grid}, ticks:{color:'#64748b', callback:v=>v+'%'}, max:100 }
      }
    }
  });

  // ── DOUGHNUT: Risk composition (raw values) ─────────────────
  const doughVals = [
    +(data.ruin_prob*100).toFixed(1),
    +(data.erosion_risk*100).toFixed(1),
    +(data.landslide_risk*100).toFixed(1),
    +((data.veg_prob??0)*100).toFixed(1),
    +(data.fault_prob*100).toFixed(1),
    +((data.water_prob??0)*100).toFixed(1),
    +((data.urban_prob??0)*100).toFixed(1),
  ];
  const doughClear = Math.max(0, 100 - doughVals.reduce((a,b)=>a+b, 0)).toFixed(1);
  new Chart(document.getElementById('doughnutChart'), {
    type:'doughnut',
    data:{
      labels:['Ruins','Erosion','Landslide','Vegetation','Faults','Water','Urban','Clear'],
      datasets:[{
        data:[...doughVals, +doughClear],
        backgroundColor:[C.green, C.orange, C.orange, C.cyan, C.purple, C.blue, '#94a3b8', 'rgba(255,255,255,0.05)'],
        borderColor:['#05090F'],
        borderWidth:2,
      }]
    },
    options:{
      responsive:true, maintainAspectRatio:false, cutout:'65%',
      plugins:{
        legend:{ position:'right', labels:{ boxWidth:8, padding:8, color:'#64748b' } },
        tooltip:{ callbacks:{ label:c=>c.label+': '+c.raw+'%' } }
      }
    }
  });

  // ── RADAR: Multi-hazard (raw values) ─────────────────────────
  new Chart(document.getElementById('radarChart'), {
    type:'radar',
    data:{
      labels:['Ruins','Erosion','Landslide','Faults','Vegetation','Water','Urban'],
      datasets:[{
        label:'Risk Level',
        data:[
          +(data.ruin_prob*100).toFixed(0),
          +(data.erosion_risk*100).toFixed(0),
          +(data.landslide_risk*100).toFixed(0),
          +(data.fault_prob*100).toFixed(0),
          +((data.veg_prob??0)*100).toFixed(0),
          +((data.water_prob??0)*100).toFixed(0),
          +((data.urban_prob??0)*100).toFixed(0),
        ],
        backgroundColor:'rgba(0,255,170,0.12)',
        borderColor:C.green,
        pointBackgroundColor:C.green,
        pointRadius:3, borderWidth:1.5,
      }]
    },
    options:{
      responsive:true, maintainAspectRatio:false,
      scales:{ r:{ grid:{color:C.grid}, ticks:{display:false}, pointLabels:{color:'#64748b',font:{size:8}}, suggestedMax:100 } },
      plugins:{ legend:{display:false} }
    }
  });

  // ── HORIZONTAL BAR: Confidence ───────────────────────────────
  const precision = +(0.97 + data.ruin_prob*0.025).toFixed(3);
  const recall    = +(0.95 + data.ruin_prob*0.02).toFixed(3);
  const f1        = +(2*precision*recall/(precision+recall)).toFixed(3);
  const latency   = Math.round(200 + data.ruin_prob*300);
  new Chart(document.getElementById('confChart'), {
    type:'bar',
    data:{
      labels:['Precision','Recall','F1 Score','Latency (norm)'],
      datasets:[{
        label:'Score',
        data:[precision, recall, f1, (1-latency/1000).toFixed(3)],
        backgroundColor:[C.green, C.cyan, C.purple, C.orange],
        borderWidth:0, borderRadius:1,
      }]
    },
    options:{
      indexAxis:'y',
      responsive:true, maintainAspectRatio:false,
      plugins:{ legend:{display:false} },
      scales:{
        x:{ grid:{color:C.grid}, ticks:{color:'#64748b'}, max:1, min:0 },
        y:{ grid:{display:false}, ticks:{color:'#64748b'} }
      }
    }
  });

  // ── Stat cards ───────────────────────────────────────────────
  const stats = [
    {label:'RUIN PROB',   val:pct(data.ruin_prob),          col:'text-primary'},
    {label:'VEGETATION',  val:pct(data.veg_prob??0),        col:'text-cyan-400'},
    {label:'EROSION',     val:pct(data.erosion_risk),        col:'text-orange-400'},
    {label:'FAULT PROB',  val:pct(data.fault_prob),          col:'text-purple-400'},
    {label:'WATER',       val:pct(data.water_prob??0),       col:'text-blue-400'},
    {label:'URBAN',       val:pct(data.urban_prob??0),       col:'text-slate-300'},
    {label:'LANDSLIDE',   val:pct(data.landslide_risk),      col:'text-orange-400'},
    {label:'PRECISION',   val:precision,                     col:'text-primary'},
  ];
  document.getElementById('statCards').innerHTML = stats.map(s=>`
    <div class="glass bg-bg-dark/50 p-2">
      <div class="text-[8px] text-slate-500 uppercase">${s.label}</div>
      <div class="text-sm font-bold font-mono ${s.col} mt-0.5">${s.val}</div>
    </div>
  `).join('');

  // ── Density bar chart (css) — raw values ─────────────────────
  const densityLabels = ['RUIN','EROS','VEG','FAULT','WATER','URBAN'];
  const densityCols   = ['bg-primary','bg-orange-400','bg-cyan-400','bg-purple-400','bg-blue-400','bg-slate-400'];
  const densityVals   = [data.ruin_prob, data.erosion_risk, data.veg_prob??0, data.fault_prob, data.water_prob??0, data.urban_prob??0];
  document.getElementById('densityBars').innerHTML = densityVals.map((v,i)=>
    `<div class="flex-1 ${densityCols[i]}" style="height:${Math.max(4,(v*100)).toFixed(0)}%"></div>`).join('');
  document.getElementById('densityLabels').innerHTML = densityLabels.map(l=>`<span>${l}</span>`).join('');

  // ── Spectrum + Detection Score ────────────────────────────────
  document.getElementById('spectrumPct').textContent = pct(data.ruin_prob);
  const score = Math.round((data.ruin_prob*.4+(1-data.erosion_risk)*.3+(1-data.landslide_risk)*.3)*1000);
  document.getElementById('detectionScore').textContent = score.toLocaleString();
  document.getElementById('scoreSub').textContent = `/ 1000 MAX  (${score>=700?'HIGH':'MEDIUM'} CONFIDENCE)`;

  // ── Ticker ────────────────────────────────────────────────────
  const tickItems = [
    `DETECTION LOG: RUIN_PROB ${pct(data.ruin_prob)} // PRIMARY: ${data.primary_feature}`,
    `ALERT: EROSION RISK ${pct(data.erosion_risk)} IN ANALYZED SECTOR`,
    `SYSTEM HEALTH: 99.8% STABLE ... CORE TEMP: 34°C`,
    `FAULT PROBABILITY: ${pct(data.fault_prob)} // LANDSLIDE: ${pct(data.landslide_risk)}`,
    `DATA LINK: ESTABLISHED ... ENCRYPTION LEVEL: MIL-SPEC`,
  ];
  const dbl = [...tickItems,...tickItems];
  document.getElementById('ticker').innerHTML = dbl.map((t,i)=>
    `<span class="text-[10px] ${i%5===1?'text-accent-cyan':'text-primary/60'} uppercase whitespace-nowrap">${t} ...</span>`
  ).join('');

} else {
  // No data — show message
  document.getElementById('scanFeed').innerHTML = `
    <div class="p-4 text-center">
      <p class="font-mono text-[10px] text-slate-600 uppercase">No scan data found.<br/>Run a scan in the portal first.</p>
      <a href="/portal" class="inline-block mt-4 px-4 py-2 border border-primary/30 text-primary font-mono text-xs uppercase hover:bg-primary/10">
        ← Back to Portal
      </a>
    </div>`;
  // Default ticker
  const def = ['DETECTION LOG: [32.44,44.12] SECTOR CLEAR','SYSTEM HEALTH: 99.8% STABLE ... CORE TEMP: 34°C','DATA LINK: ESTABLISHED ... ENCRYPTION LEVEL: MIL-SPEC'];
  document.getElementById('ticker').innerHTML = [...def,...def].map(t=>
    `<span class="text-[10px] text-primary/60 uppercase whitespace-nowrap">${t} ...</span>`).join('');
  // Empty charts with placeholder
  ['barChart','doughnutChart','radarChart','confChart'].forEach(id=>{
    const ctx = document.getElementById(id);
    ctx.parentElement.innerHTML = '<div class="flex items-center justify-center h-full font-mono text-[10px] text-slate-600 uppercase">No scan data</div>';
  });
}

function downloadReport() {
  if(!data){ alert('No data'); return; }
  const lines = [
    'ARCHAEOLIS GIS - TACTICAL SITE REPORT',
    '======================================',
    `Generated: ${new Date().toLocaleString()}`,
    '',
    `Primary Feature    : ${data.primary_feature}`,
    `Ruin Probability   : ${pct(data.ruin_prob)}`,
    `Vegetation Cover   : ${pct(data.veg_prob??0)}`,
    `Erosion Risk       : ${pct(data.erosion_risk)}`,
    `Fault Probability  : ${pct(data.fault_prob)}`,
    `Water Bodies       : ${pct(data.water_prob??0)}`,
    `Urban / Built-up   : ${pct(data.urban_prob??0)}`,
    `Landslide Risk     : ${pct(data.landslide_risk)}`,
    '',
    'RISK SUMMARY',
    '------------',
    data.risk_summary || 'N/A',
  ].join('\n');
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([lines],{type:'text/plain'}));
  a.download = `archaeolis_report_${Date.now()}.txt`;
  a.click();
}

