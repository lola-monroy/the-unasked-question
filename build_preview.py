"""Build browser-only, self-contained HTML *demo-mode preview* from exact core engine.

This bundles frozen ExtraTrees predictions for 3 synthetic held-out startups. It
cannot and does not run TabPFN inside a browser; use app.py for real inference.
"""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from src.data import (FEATURES, QUESTIONS, KNOWN_FEATURES, CATALOG,
                      generate_synthetic_data)
from src.benchmark import split_data
from src.model import train_model
from src.core import ConditionalDonorSampler, analyze, stable_seed

ROOT=Path(__file__).resolve().parent

def build_payload():
    train,test=split_data(generate_synthetic_data())
    model=train_model(train)
    sampler=ConditionalDonorSampler(train)
    cases={}
    presets=[('borderline',1,'The borderline startup','Tiny changes could flip the decision.'),
             ('surprise',22,'The counterintuitive startup','Strong metrics, but unanswered diligence risks.'),
             ('growth',26,'The fast-growing startup','Attractive growth; what is worth asking next?')]
    for key,rowid,label,subtitle in presets:
        truth=test.iloc[rowid]
        core={k:float(truth[k]) for k in KNOWN_FEATURES}
        answers={k:float(truth[k]) for k in QUESTIONS}
        states={}
        for mask in range(64):
            known=dict(core)
            for j,k in enumerate(QUESTIONS):
                if mask & (1<<j): known[k]=answers[k]
            a=analyze(model,sampler,known,upside=3,downside=1,n=180,
                      seed=stable_seed(known,rowid))
            ranking=[]
            for r in a.summary.itertuples(index=False):
                ranking.append({
                    'key':r.feature,'label':r.label,'question':r.question,
                    'gross':round(r.gross_evsi,5),'cost':round(r.cost,5),
                    'net':round(r.net_evsi,5),'information':round(r.info_gain,5),
                    'ranges':[
                        {'answer':rr.answer_range,'p':round(rr.probability,5),'weight':round(rr.weight,4),'decision':rr.decision}
                        for rr in a.response_groups[r.feature].itertuples(index=False)
                    ]
                })
            states[str(mask)]={'p':round(a.outcome_probability,5),
                               'decision':a.decision,'ranking':ranking}
        cases[key]={'name':label,'subtitle':subtitle,'initial':core,'answers':answers,
                    'outcome':int(truth.milestone_hit),'states':states}
        print(f'{key}: {len(states)} states, at start p={states["0"]["p"]:.3f}, top=',
              [(r['label'],round(r['net'],3)) for r in states['0']['ranking'][:2]])
    bench=pd.read_csv(ROOT/'results/benchmark_demo.csv')
    grouped=bench.groupby(['strategy','step'],as_index=False).agg(mean_utility=('utility','mean'))
    benchmarks={str(policy): [round(float(x),5) for x in
        grouped[grouped.strategy==policy].sort_values('step').mean_utility.tolist()]
        for policy in ['evsi','entropy','checklist']}
    meta={k:{'label':CATALOG[k].label,'question':CATALOG[k].question,'unit':CATALOG[k].unit} for k in FEATURES}
    return {'cases':cases,'benchmark':benchmarks,'features':meta,'questions':QUESTIONS,'known':KNOWN_FEATURES}

HTML=r'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1" />
<title>The Unasked Question | Decision-aware interviewer</title>
<style>
*{box-sizing:border-box} :root{--bg:#0b171a;--panel:#132528;--panel2:#1c3433;--text:#f4f7ed;--muted:#91afaa;--accent:#d7ff9b;--teal:#78b7af;--orange:#e5a885;--line:#33504b}
body{margin:0;background:radial-gradient(ellipse 70% 38% at 80% -8%,#325248 0,#0c1c20 70%),var(--bg);font-family:Inter,ui-sans-serif,system-ui,Arial,sans-serif;color:var(--text)}
header{max-width:1260px;margin:auto;padding:27px 34px 18px;display:flex;align-items:center;justify-content:space-between;border-bottom:1px solid #29413c;gap:22px}
.logo{font-weight:850;letter-spacing:.08em;font-size:.82rem;color:#e2f8ca}.logo span{color:var(--accent)}.mode{color:#cfdece;font-size:.71rem;letter-spacing:.08em;border:1px solid #50765a;padding:9px 13px;border-radius:99px;background:#1d362d}
main{max-width:1260px;margin:auto;padding:37px 34px 80px}.overline{font-size:.75rem;color:var(--accent);font-weight:760;letter-spacing:.2em;text-transform:uppercase}.title{font-size:clamp(2.6rem,6vw,5rem);font-weight:790;letter-spacing:-.075em;line-height:1;margin:14px 0 14px}.title span{color:var(--accent)}.lede{max-width:710px;color:#aec5bb;line-height:1.65;font-size:1.04rem;margin-bottom:27px}
.toolbar{display:flex;flex-wrap:wrap;gap:11px;align-items:center;margin:24px 0 22px}select,button{font:inherit}select{background:#1b3536;border:1px solid #51716a;color:white;border-radius:9px;padding:12px;outline:none;max-width:100%;cursor:pointer}button{background:var(--accent);color:#182321;border:0;border-radius:10px;font-weight:760;padding:13px 17px;cursor:pointer;transition:transform .14s,opacity .14s}button:hover{transform:translateY(-1px)}button:disabled{opacity:.4;cursor:not-allowed;transform:none}.outline{border:1px solid #607970;background:transparent;color:#ecf5e5}.divider{flex:1}.hint{color:var(--muted);font-size:.83rem}
.metrics{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin:12px 0 24px}.metric{background:#162a2b;border:1px solid #35504b;border-radius:12px;padding:15px 19px}.metric small{display:block;color:#9db6ae;font-size:.74rem;letter-spacing:.035em;margin-bottom:6px}.metric strong{font-size:1.7rem;font-weight:730;color:var(--accent);letter-spacing:-.03em}
.grid{display:grid;grid-template-columns:1.48fr 1fr;gap:20px}.stack{display:flex;flex-direction:column;gap:18px}.card{background:#132528;border:1px solid #2b4842;border-radius:15px;padding:23px}.eyebrow{font-size:.7rem;letter-spacing:.16em;text-transform:uppercase;color:var(--accent);font-weight:800}.card h2{font-size:1.13rem;letter-spacing:-.025em;margin:3px 0 17px}.recommend{padding:25px 29px;border-radius:15px;border:1px solid #5b8766;background:linear-gradient(125deg,#2d4c42,#182d31)}.recommend h3{font-size:clamp(1.2rem,2.7vw,1.75rem);margin:11px 0;line-height:1.25;letter-spacing:-.035em}.recommend p{font-size:.82rem;color:#b9d3c3;margin:5px 0}
.profile{display:grid;grid-template-columns:1fr 1fr;gap:17px}.label{font-size:.72rem;color:#97b8ab;margin-bottom:3px}.value{font-size:1.13rem;font-weight:710}.lines{border-top:1px solid #34504c;margin:16px 0;padding-top:14px}.answered{font-size:.83rem;color:#d6ead7;margin-top:11px}.small{color:var(--muted);font-size:.79rem;line-height:1.55}.chart-row{display:grid;grid-template-columns:180px 1fr 58px;gap:8px;align-items:center;font-size:.84rem;margin:10px 0;cursor:pointer}.chart-row:hover{color:var(--accent)}.bartrack{height:12px;border-radius:3px;background:#243b3d;position:relative;overflow:hidden}.barfill{height:100%;background:var(--accent);border-radius:3px}.barfill.negative{background:#718b85}.barvalue{text-align:right;color:#c4d7cf;font-size:.8rem}
.answerrow{display:grid;grid-template-columns:1fr 1fr 72px;align-items:center;gap:10px;font-size:.75rem;margin:10px 0}.progress{background:#234046;height:13px;border-radius:3px;overflow:hidden;position:relative}.progress:after{content:"";position:absolute;left:25%;top:0;bottom:0;border-left:1px dashed #f9e4a2}.progress>div{background:#83b5bd;height:100%}.progress>div.invest{background:var(--accent)}.segment{padding:9px 0;border-bottom:1px solid #2e4641}.segment:last-child{border:0}.section-title{font-size:1.35rem;font-weight:760;letter-spacing:-.035em;margin:40px 0 12px}.legend{display:flex;gap:24px;flex-wrap:wrap;margin:12px 0}.dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:7px}.result{background:#1b3432;border-radius:9px;padding:13px 16px;margin-top:15px;font-size:.88rem}.result strong{color:var(--accent)}.history-item{margin:6px 0;padding:11px;border:1px solid #36534b;border-radius:8px;font-size:.84rem}.tabs{margin:10px 0 16px}.tabs button{background:transparent;border:1px solid #43665d;color:#dcebd7;margin-right:7px}.tabs button.active{background:var(--accent);color:#102323}.badge{display:inline-flex;background:#284b3b;color:var(--accent);padding:6px 11px;border-radius:99px;font-size:.75rem;letter-spacing:.06em}
svg{width:100%;height:auto}.foot{margin-top:28px;color:#829e95;font-size:.76rem;line-height:1.6}.research{background:#18312d;border-left:3px solid var(--accent);padding:20px 24px;border-radius:2px 10px 10px 2px;margin-top:15px}.research p{margin:8px 0;color:#c3d8cd;line-height:1.5}
@media(max-width:870px){header{padding:22px 17px}main{padding:28px 16px 60px}.grid{grid-template-columns:1fr}.metrics{grid-template-columns:1fr 1fr}.chart-row{grid-template-columns:140px 1fr 52px}.toolbar{flex-wrap:wrap}.divider{display:none}}@media(max-width:420px){.metric strong{font-size:1.3rem}.card{padding:17px}.chart-row{grid-template-columns:110px 1fr 38px;font-size:.76rem}.mode{font-size:.6rem}}
</style></head><body>
<header><div class="logo">◈ &nbsp; THE UNASKED <span>QUESTION</span> <small style="font-weight:400;color:#a2b9b0">/ A PRIOR LABS CONCEPT</small></div><div class="mode">OFFLINE PREVIEW • SYNTHETIC DATA • NOT TABPFN</div></header>
<main><div class="overline">A DECISION-AWARE INTERVIEWER</div>
<h1 class="title">What is worth <span>asking?</span></h1>
<div class="lede">Predicting an outcome is easy to demonstrate. Deciding <i>which piece of missing information is worth paying for</i> is a different problem. This prototype ranks founder questions by how much their answers could improve an actual decision.</div>
<div class="toolbar"><select id="case"><option value="borderline">The borderline startup</option><option value="surprise">The counterintuitive startup</option><option value="growth">The fast-growing startup</option></select><button class="outline" onclick="reset()">↻ &nbsp; Reset interview</button><span class="divider"></span><span class="hint">GOAL &nbsp; €500k ARR within 18 months</span></div>
<div class="metrics"><div class="metric"><small>Estimated milestone chance</small><strong id="prob">—</strong></div><div class="metric"><small>Best decision now</small><strong id="decision">—</strong></div><div class="metric"><small>Invest threshold</small><strong>25%</strong></div><div class="metric"><small>Answers gathered</small><strong id="count">0 / 6</strong></div></div>
<div class="grid"><div class="stack"><div class="recommend" id="recommend"></div><div class="card"><div class="eyebrow">Decision value ≠ uncertainty reduction</div><h2 style="margin-top:7px">Which question is worth asking?</h2><div id="rank"></div><p class="small">Green = net expected decision value &gt; 0 after acquiring the answer. Gray = answer cost exceeds estimated value. Utility units are illustrative, not euros.</p></div><div class="card"><div class="eyebrow">Before asking</div><h2 style="margin-top:8px">How might the answer change the decision?</h2><select id="inspect" style="width:100%" onchange="drawRanges()"></select><div id="ranges"></div><p class="small">Each bar is the predicted milestone probability after grouping plausible answers. The vertical decision threshold is 25%. These are conditional predictions, not causal effects.</p></div></div>
<div class="stack"><div class="card"><div class="eyebrow">In the room</div><h2 style="margin:7px 0 18px">Founder snapshot</h2><div class="profile" id="profile"></div><div class="lines"><div class="eyebrow">Answers uncovered</div><div id="uncovered" class="small">None yet — six unanswered questions.</div></div></div><div class="card"><div class="eyebrow">Ask the founder</div><h2 style="margin:8px 0 13px">Acquisition, not prediction</h2><select id="ask" style="width:100%;margin-bottom:11px"></select><button id="askBtn" style="width:100%" onclick="revealSelectedQuestion()">Ask → update the decision</button><p class="small">Click to reveal the held-out founder's actual <b>synthetic</b> answer. It is never shown to the ranking algorithm before you ask.</p><div id="lastAnswer"></div></div><div class="card"><div class="eyebrow">Interview trail</div><h2 style="margin-top:8px">Information acquired</h2><div id="history" class="small">Your interview will appear here.</div></div></div></div>
<h2 class="section-title">Does asking better questions improve decisions?</h2><div class="grid"><div class="card"><div class="eyebrow">Offline synthetic benchmark • 40 held-out companies</div><h2 style="margin-top:8px">Same cases. Three questioning policies.</h2><div class="legend"><span><i class="dot" style="background:#d7ff9b"></i>Decision-aware EVSI</span><span><i class="dot" style="background:#78b7ee"></i>Max entropy reduction</span><span><i class="dot" style="background:#e4aa81"></i>Fixed checklist</span></div><div id="bench"></div><p class="small">Mean realized utility after 0–4 questions, net of answer costs. A toy experiment: not real venture performance, not evidence of TabPFN superiority.</p></div><div class="card"><div class="eyebrow">The mathematical distinction</div><h2 style="margin-top:8px">Ask what changes the <i>action</i>, not merely the forecast.</h2><p style="color:#c5d8cc;line-height:1.68">Predict plausible answers → update conditional outcome estimates → compute the best decision for each answer → average those best-decision payoffs → subtract the value of deciding now and the cost of asking.</p><div class="research"><div class="eyebrow">Research question</div><p>Can a tabular foundation model make <b>information acquisition</b> more data-efficient — especially when historical observations are few and questions are expensive?</p></div><p class="small" style="margin-top:20px">Value of sample information (EVSI) is an established technique. The novel hypothesis worth testing is whether TabPFN helps in scarce-data interactive decision settings.</p></div></div>
<div class="foot">◈ <b>Important:</b> This standalone preview is driven by frozen ExtraTrees classification results and synthetic startup outcomes. It does not run TabPFN or contact a server. The accompanying Streamlit project runs <b>actual TabPFN 3.5 / 3.5 Fast</b> after official model access is configured, using the same question-selection logic. Neither mode is investment advice.</div>
</main>
<script>const DATA=__PAYLOAD__;
let key='borderline',mask=0,history=[];const $=id=>document.getElementById(id);function meta(q){return DATA.features[q]};function fmt(q,v){return Number(v).toLocaleString('en-US',{maximumFractionDigits:1})+' '+meta(q).unit}
function current(){return DATA.cases[key].states[String(mask)]};function available(){return current().ranking};function best(){return available().find(r=>r.net>0)||null};function reset(){mask=0;history=[];update()};
$('case').addEventListener('change',e=>{key=e.target.value;reset()});
function update(){let c=DATA.cases[key],s=current(),rank=available(),recommend=best();$('prob').textContent=(s.p*100).toFixed(1)+'%';$('decision').textContent=s.decision;$('count').textContent=history.length+' / 6';
let html='';if(!rank.length){html='<div class="eyebrow">INTERVIEW COMPLETE</div><h3>All questions answered.</h3><p>There is no more information to acquire.</p>'}else if(recommend){html='<div class="eyebrow">NEXT BEST QUESTION • NET VALUE '+recommend.net.toFixed(3)+'</div><h3>“'+recommend.question+'”</h3><p>Most likely to improve the decision after considering the cost of obtaining an answer.</p>'}else{html='<div class="eyebrow">OPTIMAL STOPPING POINT</div><h3>Don’t ask another question.</h3><p>No remaining answer has positive estimated net decision value given these utility and acquisition-cost assumptions.</p>'}$('recommend').innerHTML=html;
let max=Math.max(.005,...rank.map(r=>Math.abs(r.net)));$('rank').innerHTML=rank.map(r=>'<div class="chart-row" onclick="selectQ(\''+r.key+'\')" title="Information gain: '+r.information.toFixed(3)+' nats"><div>'+r.label+'</div><div class="bartrack"><div class="barfill '+(r.net<=0?'negative':'')+'" style="width:'+(Math.abs(r.net)/max*100).toFixed(0)+'%"></div></div><div class="barvalue">'+(r.net>0?'+':'')+r.net.toFixed(3)+'</div></div>').join('')||'<p class="small">Everything requested has been revealed.</p>';
$('profile').innerHTML=DATA.known.map(q=>'<div><div class="label">'+meta(q).label+'</div><div class="value">'+fmt(q,c.initial[q])+'</div></div>').join('');$('uncovered').innerHTML=history.length?history.map(h=>'<div class="answered">'+meta(h.q).label+': <b>'+fmt(h.q,h.answer)+'</b></div>').join(''):'None yet — six unanswered questions.';
let select=rank.map(r=>'<option value="'+r.key+'">'+r.question+'</option>').join('');let oldQ=$('inspect').value;
$('inspect').innerHTML=select;$('ask').innerHTML=select;if(recommend){$('ask').value=recommend.key;$('inspect').value=rank.some(r=>r.key===oldQ)?oldQ:recommend.key}if(!rank.length){$('ask').disabled=true;$('inspect').disabled=true;$('askBtn').disabled=true;$('ranges').innerHTML='No remaining questions.'}else{$('ask').disabled=false;$('inspect').disabled=false;$('askBtn').disabled=false;drawRanges()}
$('history').innerHTML=history.length?history.slice().reverse().map((h,i)=>'<div class="history-item"><span class="badge">ANSWER '+(history.length-i)+'</span> <b>'+meta(h.q).label+'</b><div style="margin-top:6px">'+fmt(h.q,h.answer)+'</div><div class="small">Before: '+(h.oldp*100).toFixed(1)+'% / '+h.olddecision+'</div></div>').join(''):'Your interview will appear here.';}
function selectQ(q){$('inspect').value=q;$('ask').value=q;drawRanges()}
function drawRanges(){let r=available().find(x=>x.key===$('inspect').value);if(!r)return;$('ranges').innerHTML=r.ranges.map(a=>'<div class="segment"><div class="answerrow"><div>'+a.answer+'</div><div class="progress"><div class="'+(a.p>=.25?'invest':'')+'" style="width:'+(100*a.p).toFixed(1)+'%"></div></div><div style="text-align:right;color:'+(a.p>=.25?'#d7ff9b':'#abd4df')+'">'+(a.p*100).toFixed(1)+'%</div></div><div class="small">'+Math.round(a.weight*100)+'% of simulated scenarios · '+a.decision+'</div></div>').join('')}
function revealSelectedQuestion(){let q=$('ask').value;if(!q||mask&(1<<DATA.questions.indexOf(q)))return;let state=current(),c=DATA.cases[key],answer=c.answers[q];history.push({q,answer,oldp:state.p,olddecision:state.decision});mask|=(1<<DATA.questions.indexOf(q));update();$('lastAnswer').innerHTML='<div class="result">Founder answered <strong>'+fmt(q,answer)+'</strong>.<br>New recommendation and decision computed.</div>'}
function drawBench(){let svgW=560,svgH=300,left=47,top=26,right=14,bottom=42;
let vals=Object.values(DATA.benchmark).flat(),lo=Math.min(...vals)-.04,hi=Math.max(...vals)+.04,XP=x=>left+(svgW-left-right)*(x/4),YP=y=>top+(svgH-top-bottom)*(1-(y-lo)/(hi-lo));
let out='<svg viewBox="0 0 '+svgW+' '+svgH+'" role="img" aria-label="Average held-out utility by policy and number of questions">';for(let i=0;i<5;i++){let y=lo+(hi-lo)*i/4;out+='<line x1="'+left+'" x2="'+(svgW-right)+'" y1="'+YP(y)+'" y2="'+YP(y)+'" stroke="#34524d"/><text x="'+(left-6)+'" y="'+(YP(y)+4)+'" fill="#abc3bb" font-size="11" text-anchor="end">'+y.toFixed(2)+'</text>'}for(let i=0;i<5;i++)out+='<text x="'+XP(i)+'" y="'+(svgH-15)+'" fill="#abc3bb" font-size="12" text-anchor="middle">'+i+'</text>';
let co={evsi:'#d7ff9b',entropy:'#78b7ee',checklist:'#e4aa81'};for(let name of ['checklist','entropy','evsi']){let values=DATA.benchmark[name],path=values.map((v,i)=>(i?'L':'M')+XP(i)+','+YP(v)).join(' ');out+='<path d="'+path+'" stroke="'+co[name]+'" stroke-width="3" fill="none" />';values.forEach((v,i)=>out+='<circle cx="'+XP(i)+'" cy="'+YP(v)+'" r="4.5" fill="'+co[name]+'" />')};out+='<text x="'+(svgW/2)+'" y="'+(svgH-1)+'" fill="#afc6b9" font-size="12" text-anchor="middle">Question budget</text></svg>';$('bench').innerHTML=out}update();drawBench();
</script></body></html>'''

if __name__=='__main__':
    data=build_payload()
    text=HTML.replace('__PAYLOAD__',json.dumps(data,ensure_ascii=False,separators=(',',':')))
    out=ROOT/'preview.html'
    out.write_text(text,encoding='utf8')
    print(f'Wrote {out}, {out.stat().st_size//1024} KiB')
