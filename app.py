"""The Unasked Question | A decision-aware TabPFN experiment.

Run from repository root: streamlit run app.py
"""
from __future__ import annotations
import io
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from src.data import (CATALOG, FEATURES, KNOWN_FEATURES, QUESTIONS, TARGET,
                      generate_synthetic_data, validate_uploaded_data)
from src.benchmark import split_data, evaluate
from src.model import train_model
from src.core import (ConditionalDonorSampler, analyze, next_question,
                      stable_seed, decision_label)

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title='The Unasked Question — TabPFN × Decision Intelligence',
                   page_icon='◈', layout='wide', initial_sidebar_state='expanded')
st.markdown('''<style>
:root {color-scheme:dark}
.stApp {background: radial-gradient(ellipse at 75% -18%, #243e3d 0%, #111b21 38%, #0d141b 85%); color:#f6f6ec;}
.block-container {padding-top:1.65rem; max-width:1320px;}
[data-testid="stSidebar"] {background:#101d21;}
[data-testid="stMetric"] {background:#17252c;border:1px solid #314443;border-radius:12px;padding:13px 17px;}
[data-testid="stMetricLabel"] {color:#afc3bf}
[data-testid="stMetricValue"] {color:#d9ffa0;}
.stButton>button[kind="primary"] {background:#d4ff93; color:#14231d; font-weight:720; border:0;}
.stButton>button[kind="secondary"] {border:1px solid #65766c;}
div[data-testid="stDataFrame"] {border-radius:10px;}
.eyebrow {text-transform:uppercase;letter-spacing:.19em;color:#c9f79b;font-size:.75rem;font-weight:750}
.hero {font-size:3.1rem;line-height:1.07;font-weight:800;letter-spacing:-.065em;color:#fafdf6;margin:.45rem 0 .5rem}
.hero span {color:#d3fc91;}
.subtitle {font-size:1.07rem;max-width:760px;color:#b2c4bf;line-height:1.53;}
.pill {display:inline-block;padding:5px 11px;border-radius:99px;background:#223d33;border:1px solid #426747;font-size:.73rem;letter-spacing:.05em;color:#d8ffa9;margin-bottom:.65rem;}
.question-box {background:linear-gradient(130deg,#25413d,#17272e);border:1px solid #4f7864;padding:19px 24px;border-radius:16px; margin:10px 0 18px;}
.question-box .micro {letter-spacing:.13em;font-size:.71rem;color:#d3ff9b;text-transform:uppercase;font-weight:800;}
.question-box h2 {color:white;font-size:1.55rem;margin:.4rem 0 .6rem;line-height:1.25;}
.note {color:#9db1a9;font-size:.82rem;line-height:1.55}
.smallcap {color:#afc1bf;text-transform:uppercase;font-size:.7rem;letter-spacing:.13em;font-weight:750}
.tag {color:#d9ffa0;font-size:.8rem;font-weight:650}
hr {border-color:#344743}
</style>''', unsafe_allow_html=True)


def fmt_answer(q, value):
    return f'{value:,.1f} {CATALOG[q].unit}' if q not in ('team_size','founder_experience_years') else f'{int(round(value))} {CATALOG[q].unit}'


def chart_bar(a):
    df = a.summary.iloc[::-1]
    colors=['#bbff88' if v>0 else '#647a78' for v in df.net_evsi]
    fig=go.Figure(go.Bar(x=df.net_evsi, y=df.label, orientation='h',
                         marker_color=colors,
                         customdata=df[['gross_evsi','cost','info_gain']].to_numpy(),
                         hovertemplate='<b>%{y}</b><br>Net EVSI: %{x:.4f}<br>Gross EVSI: %{customdata[0]:.4f}<br>Cost: %{customdata[1]:.4f}<br>Information gain: %{customdata[2]:.3f} nats<extra></extra>'))
    fig.add_vline(x=0,line_color='#879995',line_width=1)
    fig.update_layout(height=320,margin=dict(l=0,r=10,t=13,b=15),paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)',font_color='#d7e6dd',showlegend=False,
                      xaxis=dict(title='Net decision value (utility units)',showgrid=True,gridcolor='#293a40',zeroline=False),
                      yaxis=dict(title='',tickfont=dict(size=12)))
    return fig


def chart_answer_ranges(a,q,upside,downside):
    frame=a.response_groups[q].copy()
    cut=downside/(upside+downside)
    frame['pct']=(frame.probability*100).round(1)
    fig=go.Figure()
    fig.add_trace(go.Bar(x=frame.answer_range, y=frame.pct, marker_color=[
        '#befe8b' if p>=cut else '#80b5bd' for p in frame.probability],
        text=[f'{p:.0f}%' for p in frame.pct], textposition='outside',
        hovertemplate='%{x}<br>Probability: %{y:.1f}%<extra></extra>'))
    fig.add_hline(y=cut*100,line_dash='dot',line_color='#ffe9ae',
                  annotation_text=f'Invest threshold {cut*100:.0f}%',annotation_position='top left')
    fig.update_layout(height=310,margin=dict(l=4,r=0,t=27,b=18),
                      paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',
                      font_color='#d7e6dd',xaxis=dict(title='',tickangle=-10),
                      yaxis=dict(title='Estimated milestone probability (%)',range=[0,105],gridcolor='#293a40'))
    return fig


@st.cache_data

def synthetic_data():
    return generate_synthetic_data()


@st.cache_resource(show_spinner='Loading the selected prediction model…')
def build_engine(dataset_csv: bytes, backend: str):
    full = pd.read_csv(io.BytesIO(dataset_csv))
    train,test = split_data(full)
    model = train_model(train,backend)
    sampler=ConditionalDonorSampler(train)
    return model,sampler,train,test


def render_profile(row, known):
    st.markdown('#### Startup snapshot')
    cols=st.columns(2)
    for j,q in enumerate(KNOWN_FEATURES):
        with cols[j%2]:
            st.markdown(f'<div class="smallcap">{CATALOG[q].label}</div><div style="font-size:1.25rem;color:#f1f5e9;font-weight:650;margin:4px 0 16px">{fmt_answer(q,known[q])}</div>',unsafe_allow_html=True)
    st.markdown('<div class="smallcap">Additional information collected</div>',unsafe_allow_html=True)
    if len(known)==len(KNOWN_FEATURES):
        st.caption('No additional answers yet. All six diligence questions are hidden.')
    else:
        for q in known:
            if q in QUESTIONS:
                st.markdown(f'**{CATALOG[q].label}:** {fmt_answer(q,known[q])}')


st.sidebar.markdown('<div class="eyebrow">◈ PRIOR LABS • CONCEPT DEMO</div>',unsafe_allow_html=True)
st.sidebar.subheader('Experiment controls')
source=st.sidebar.radio('Data', ['Synthetic sandbox','My labeled CSV'], horizontal=False)
if source=='My labeled CSV':
    uploaded=st.sidebar.file_uploader('Upload labeled startup data',type='csv')
    st.sidebar.caption('Requires 10 numeric feature columns + binary milestone_hit; at least 80 complete rows.')
    st.sidebar.download_button('↓ Download example CSV schema', data=synthetic_data().head(25).to_csv(index=False),
                               file_name='startup_data_schema_example.csv',mime='text/csv')
    if uploaded is None:
        st.info('Upload a labeled CSV to begin. You can start with **Synthetic sandbox** to explore the experience.')
        st.stop()
    try:
        data=validate_uploaded_data(pd.read_csv(uploaded))
    except Exception as exc:
        st.error(str(exc)); st.stop()
else:
    data=synthetic_data()

backend=st.sidebar.radio('Prediction engine',
    ['Instant demo · ExtraTrees','TabPFN 3.5 Fast · real model','TabPFN 3.5 · real model'],
    index=0)
backend_id={'Instant demo · ExtraTrees':'demo',
            'TabPFN 3.5 Fast · real model':'tabpfn-fast',
            'TabPFN 3.5 · real model':'tabpfn'}[backend]
if backend_id=='demo':
    st.sidebar.warning('DEMO MODEL: This is **not** TabPFN. Switch engines to run the actual TabPFN checkpoint.')
else:
    st.sidebar.info('Actual TabPFN. First use may require downloading weights and accepting Prior Labs model terms. No fallback is applied.')

upside=st.sidebar.slider('Gain if milestone reached', 1.0,8.0,3.0,.5)
downside=st.sidebar.slider('Loss if milestone missed', .5,4.0,1.0,.5)
question_cost_multiplier=st.sidebar.slider('Cost of obtaining answers',0.0,5.0,1.0,.25)
scenarios=st.sidebar.select_slider('Plausible worlds per question',options=[80,120,180,260,400],value=180)

try:
    raw=data.to_csv(index=False).encode()
    model,sampler,train,test=build_engine(raw,backend_id)
except Exception as exc:
    st.error(f'Unable to initialize {backend}. {exc}')
    st.info('Use the Instant demo engine to explore the UI without downloading TabPFN. This is an explicitly labeled replacement model, not TabPFN.')
    st.stop()

if source == 'Synthetic sandbox':
    choices={'The borderline startup':1,'The counterintuitive startup':22,'The fast-growing startup':26}
    name=st.sidebar.selectbox('Founder entering the room',list(choices))
    case_id=choices[name]
else:
    name=st.sidebar.selectbox('Held-out founder case', [f'Founder #{i+1}' for i in range(min(30,len(test)))])
    case_id=int(name.split('#')[-1])-1

dataset_key=(hash(raw),backend_id,upside,downside,question_cost_multiplier)
if st.session_state.get('dataset_key')!=dataset_key:
    st.session_state['dataset_key']=dataset_key
    st.session_state.pop('benchmark_result',None)
case_key=(hash(raw),backend_id,case_id)
if st.session_state.get('case_key')!=case_key:
    st.session_state['case_key']=case_key
    st.session_state['revealed']={}
    st.session_state['history']=[]

actual=test.iloc[case_id]
known={k:float(actual[k]) for k in KNOWN_FEATURES}
known.update(st.session_state.revealed)

st.markdown('<div class="pill">THE DECISION-AWARE INTERVIEWER · INTERACTIVE RESEARCH PROTOTYPE</div>',unsafe_allow_html=True)
st.markdown('<div class="hero">The <span>Unasked</span> Question.</div>',unsafe_allow_html=True)
st.markdown('<div class="subtitle">Most AI systems predict an outcome from the data you have. This one tries to calculate which missing answer is <b>worth obtaining next</b> — because not all uncertainty matters to a decision.</div>',unsafe_allow_html=True)
st.caption(f"DATA: {'SYNTHETIC & SIMULATED' if source=='Synthetic sandbox' else 'USER-PROVIDED'}  ·  MODEL: {backend}  ·  MILESTONE: €500k ARR WITHIN 18 MONTHS")

with st.sidebar.expander('What is the decision?',expanded=False):
    st.write('Two actions: INVEST or PASS. INVEST yields the gain above if the milestone is met; otherwise it loses the downside. PASS yields zero. Net EVSI subtracts the stated cost of an answer.')
    st.caption('Utilities and question costs are illustrative, not investment advice. Predictions are not guaranteed calibrated.')

try:
    a=analyze(model,sampler,known,upside=upside,downside=downside,n=scenarios,
              seed=stable_seed(known,case_id),question_cost_multiplier=question_cost_multiplier)
except Exception as exc:
    st.error(f'Inference failed: {exc}'); st.stop()

recommended=next_question(a,'evsi')
pct=a.outcome_probability*100
cut=downside/(upside+downside)
metrics=st.columns(4)
metrics[0].metric('Estimated milestone chance', f'{pct:.1f}%')
metrics[1].metric('Decision now',a.decision)
metrics[2].metric('Investment threshold',f'{cut*100:.1f}%')
metrics[3].metric('Answers gathered',f'{len(st.session_state.revealed)}/6')

interview,experiment,method=st.tabs(['◈  Live interview','↗  Policy experiment','⚙  Method and limitations'])

with interview:
    main,side=st.columns([1.55,1],gap='large')
    with main:
        if recommended is not None:
            top=a.summary.iloc[0]
            st.markdown(f'''<div class="question-box"><div class="micro">Next best question · net value {top.net_evsi:.3f}</div>
                <h2>“{CATALOG[recommended].question}”</h2>
                <div class="note">Ranked by its potential to improve the actual invest/pass decision, after the cost of asking.</div></div>''',unsafe_allow_html=True)
        elif a.summary.empty:
            st.success('All diligence questions have been answered. No more questions to ask.')
        else:
            st.markdown('''<div class="question-box"><div class="micro">Intelligent stop</div>
                <h2>Don’t ask another question.</h2><div class="note">Under the assumed payoffs and acquisition costs, no remaining question has positive estimated net decision value. You can still explore one manually.</div></div>''',unsafe_allow_html=True)
        st.markdown('#### What should we ask?')
        if not a.summary.empty:
            st.plotly_chart(chart_bar(a),use_container_width=True,config={'displayModeBar':False})
            st.caption('Green = positive estimated net value. Gray = cost exceeds estimated decision benefit. Values are in hypothetical utility units, not euros.')
            opts=list(a.summary.feature)
            chosen=st.selectbox('Explore any remaining question',opts,
                format_func=lambda q: CATALOG[q].label,
                index=opts.index(recommended) if recommended in opts else 0)
            st.markdown('##### How different answers could change the decision')
            st.plotly_chart(chart_answer_ranges(a,chosen,upside,downside),use_container_width=True,config={'displayModeBar':False})
            st.caption('Each bar averages model predictions across plausible joint-feature scenarios with the answer in the indicated range. This is not a causal effect of changing the variable.')

    with side:
        render_profile(actual,known)
        if not a.summary.empty:
            st.divider()
            st.markdown('#### Interview the founder')
            question_to_ask=st.selectbox('Question to ask',list(a.summary.feature),
                     format_func=lambda q:CATALOG[q].question,
                     index=list(a.summary.feature).index(recommended) if recommended in list(a.summary.feature) else 0)
            hypothetical=st.checkbox('What-if: type an answer instead of revealing the hidden value',value=False)
            override=None
            if hypothetical:
                override=st.number_input(f"Founder’s answer ({CATALOG[question_to_ask].unit})",value=float(train[question_to_ask].median()),step=1.0)
            if st.button('Ask → update the decision',type='primary',use_container_width=True):
                answer=float(override) if hypothetical else float(actual[question_to_ask])
                st.session_state.revealed[question_to_ask]=answer
                st.session_state.history.append((question_to_ask,answer,a.outcome_probability,a.decision,hypothetical))
                st.rerun()
            st.caption('Without what-if mode, the answer is revealed from the held-out case. It is not shown or used by the question selector beforehand.')
        if st.button('↺ Reset interview',use_container_width=True):
            st.session_state.revealed={}
            st.session_state.history=[]
            st.rerun()
        if st.session_state.history:
            st.divider()
            st.markdown('#### Interview log')
            for i,(q,ans,old_p,old_d,override) in enumerate(st.session_state.history[::-1]):
                with st.expander(f'{len(st.session_state.history)-i}. {CATALOG[q].label}',expanded=i==0):
                    st.write(f'**Answer:** {fmt_answer(q,ans)}'+(' *(what-if)*' if override else ''))
                    st.caption(f'Before question: {old_p*100:.1f}% · {old_d}')
                    if i==0:
                        st.write(f'**Now: {a.outcome_probability*100:.1f}% · {a.decision}**')
        st.divider()
        with st.expander('Detailed question ranking'):
            if not a.summary.empty:
                display=a.summary[['label','gross_evsi','cost','net_evsi','info_gain']].copy()
                display.columns=['Question','Gross EVSI','Answer cost','Net EVSI','Info gain (nats)']
                st.dataframe(display,hide_index=True,use_container_width=True)
                st.caption('Compare the EVSI ranking to the largest information gain: they can disagree.')

with experiment:
    st.markdown('### Same budget. Different questioning strategy.')
    st.write('Compare **decision-aware EVSI**, **entropy reduction**, and a **fixed diligence checklist** on the same held-out cases. Every strategy uses the same prediction model, imputation assumptions, and invest/pass payoffs.')
    preset=ROOT/'results'/'benchmark_demo.csv'
    can_use_preset=(source=='Synthetic sandbox' and backend_id=='demo' and
                    upside==3.0 and downside==1.0 and question_cost_multiplier==1.0)
    if can_use_preset and preset.exists() and 'benchmark_result' not in st.session_state:
        st.caption('Showing a saved, reproducible experiment on synthetic companies using ExtraTrees; not TabPFN. Change settings and rerun below for a fresh result.')
        bench=pd.read_csv(preset)
    else:
        bench=st.session_state.get('benchmark_result')
        if bench is not None:
            st.caption('Custom run with the selected prediction engine and data.')
    b1,b2,b3=st.columns([1,1,1])
    n_cases=b1.slider('Cases',8,min(60,len(test)),min(24,len(test)))
    n_steps=b2.slider('Questions allowed',1,6,4)
    n_worlds=b3.selectbox('Scenarios per evaluation',[40,70,100],index=1)
    if st.button('Run / re-run held-out experiment'):
        with st.spinner('Evaluating identical held-out cases under all three policies…'):
            bench=evaluate(test,model,sampler,cases=n_cases,steps=n_steps,scenarios=n_worlds,
                           upside=upside,downside=downside,
                           cost_multiplier=question_cost_multiplier)
            st.session_state['benchmark_result']=bench
    if bench is not None:
        labels={'evsi':'Decision-aware EVSI','entropy':'Maximum information gain','checklist':'Fixed checklist'}
        aggregate=bench.groupby(['strategy','step'],as_index=False).agg(
            mean_utility=('utility','mean'),mean_questions=('questions_asked','mean'),
            n=('case','size'))
        fig=go.Figure()
        for policy,color in [('evsi','#c6ff85'),('entropy','#87b9f5'),('checklist','#f0a971')]:
            sub=aggregate[aggregate.strategy==policy]
            fig.add_trace(go.Scatter(x=sub.step,y=sub.mean_utility,mode='lines+markers',
                                     name=labels[policy],line=dict(color=color,width=3),marker=dict(size=8)))
        fig.update_layout(height=420,paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',
                          font_color='#e4eee4',margin=dict(l=0,r=10,t=25,b=20),
                          xaxis_title='Question budget',yaxis_title='Mean realized utility, net of answer costs',
                          xaxis=dict(dtick=1,gridcolor='#283a3e'),yaxis=dict(gridcolor='#283a3e'),
                          legend=dict(orientation='h',y=1.18))
        st.plotly_chart(fig,use_container_width=True,config={'displayModeBar':False})
        end=aggregate[aggregate.step==aggregate.step.max()].copy()
        end['Strategy']=end.strategy.map(labels)
        st.dataframe(end[['Strategy','mean_utility','mean_questions','n']].rename(
            columns={'mean_utility':'Mean net utility','mean_questions':'Mean questions actually asked','n':'Held-out cases'}),
            hide_index=True,use_container_width=True)
        st.download_button('↓ Download benchmark results',bench.to_csv(index=False),
                           file_name='unasked_question_benchmark.csv',mime='text/csv')
        st.caption('Realized utility depends on one noisy binary outcome per synthetic case. These are not backtested VC returns and not a scientific superiority claim; larger, real and temporally held-out datasets are needed.')
    else:
        st.info('Click **Run / re-run held-out experiment** to generate a direct policy comparison.')

with method:
    st.markdown('### One sentence')
    st.write('Estimate the **expected benefit of learning a missing fact**, instead of simply predicting with existing facts.')
    st.latex(r'\mathrm{EVSI}(q)=\mathbb E_{a_q}\left[\max_d\mathbb E[U(d,Y)\mid x,a_q]\right]-\max_d\mathbb E[U(d,Y)\mid x]')
    st.write('**How this implementation approximates it:** (1) sample coherent missing-feature profiles using a distance-weighted donor model fit on training features; (2) score the completed scenarios with TabPFN or the explicitly labeled ExtraTrees demo model; (3) bin plausible answers to each question and recompute the best investment decision; (4) subtract the cost of acquiring each answer. Both value before and after use the same scenario distribution to avoid artificial positive EVSI from inconsistent baselines.')
    st.write('**Information gain baseline:** choose the feature with the largest expected reduction in binary outcome entropy. **Checklist baseline:** always ask gross margin, founder experience, runway, revenue concentration, churn, then CAC payback.')
    st.warning('Research caveats: (a) donor matching is approximate and may misrepresent the joint conditional distribution; (b) TabPFN classification probabilities are **not** a full posterior over startup outcomes, nor are they automatically calibrated; (c) synthetic startup success labels do **not** validate a venture-screening tool; (d) missingness, strategic misreporting, and shift over time are not modeled; (e) EVSI is an established decision-theory idea, not a claim of a new algorithm.')
    st.write('**Making this publishable / credible:** replace the synthetic data with time-stamped, consented historical outcomes; train only on information available at deal time; assess temporal OOD generalization, calibration, selective stop quality, and conditional-sampler fidelity; compare against a fixed checklist, entropy acquisition, and a standard LLM interviewer under matched cost budgets.')
    st.markdown('**Live implementation:** `src/core.py` holds the EVSI selector; `src/model.py` switches between real TabPFN-3.5-Fast and clearly labeled ExtraTrees; `src/benchmark.py` runs the held-out experiment.')
    st.caption('TabPFN checkpoint/model weights are provided by Prior Labs under their separate terms. See their official repository for latest usage and licensing. This app ships no model weights or real startup data.')

st.markdown('---')
st.caption('THE UNASKED QUESTION  /  A prototype, not a due-diligence recommendation.  ·  Built to expose a research question, not to claim that VC success can reliably be predicted.')
