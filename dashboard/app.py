from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
import plotly.graph_objects as go

from src.reporting.report_tables import build_bundle
try:
    from src.reporting.charts import build_all_figures
except Exception:
    build_all_figures = None

st.set_page_config(
    page_title='Japan Bond Shock Risk Dashboard',
    layout='wide',
    initial_sidebar_state='expanded'
)

ACCENT = '#f97316'
ACCENT_2 = '#fb923c'
BG = '#0b1220'
CARD = '#111827'
TEXT = '#e5e7eb'
MUTED = '#9ca3af'
BORDER = '#1f2937'

st.markdown(f"""
<style>
.stApp {{ background: {BG}; color: {TEXT}; }}
[data-testid="stSidebar"] {{ background: #0f172a; border-right: 1px solid {BORDER}; }}
.block-container {{ padding-top: 1.2rem; padding-bottom: 2rem; }}
h1, h2, h3 {{ color: {TEXT}; }}
.small-note {{ color: {MUTED}; font-size: 0.92rem; }}
.metric-card {{
    background: linear-gradient(180deg, rgba(17,24,39,0.98), rgba(15,23,42,0.98));
    border: 1px solid {BORDER}; border-radius: 16px; padding: 16px 18px;
    box-shadow: 0 0 0 1px rgba(249,115,22,0.03), 0 10px 30px rgba(0,0,0,0.25);
}}
.section-card {{
    background: linear-gradient(180deg, rgba(17,24,39,0.95), rgba(15,23,42,0.95));
    border: 1px solid {BORDER}; border-radius: 18px; padding: 18px;
}}
.kpi-label {{ color: {MUTED}; font-size: 0.88rem; margin-bottom: 0.2rem; }}
.kpi-value {{ color: {TEXT}; font-size: 1.65rem; font-weight: 700; }}
.kpi-accent {{ color: {ACCENT_2}; font-size: 0.9rem; }}
hr {{ border-color: {BORDER}; }}
</style>
""", unsafe_allow_html=True)


def fmt_pct(x):
    return '—' if x is None else f'{x:,.2f}%'


def fallback_fig(title: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        title=title,
        template='plotly_dark',
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        xaxis={'visible': False},
        yaxis={'visible': False},
        annotations=[dict(text='Chart unavailable', x=0.5, y=0.5, showarrow=False, font=dict(size=18))],
    )
    return fig


bundle = build_bundle()
figs = build_all_figures(bundle) if build_all_figures else {}

st.sidebar.title('Risk Dashboard')
st.sidebar.markdown('Interactive view of the Japan bond shock project.')
st.sidebar.markdown(f"**Curve source:** {bundle.notes.get('curve_source', 'n/a')}")
st.sidebar.markdown(f"**Returns source:** {bundle.notes.get('returns_source', 'n/a')}")
show_var = st.sidebar.toggle('Show VaR section', value=bundle.latest_risk is not None)
show_tables = st.sidebar.toggle('Show raw tables', value=True)

st.title('Japan Bond Shock Risk Dashboard')
st.markdown('<div class="small-note">Dark, Python-first dashboard using the same analytical tables that will feed exports and the final report.</div>', unsafe_allow_html=True)

scenario_losses = bundle.notes.get('scenario_loss_pct', {})
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f'<div class="metric-card"><div class="kpi-label">Baseline shock</div><div class="kpi-value">{fmt_pct(scenario_losses.get("japan_baseline"))}</div><div class="kpi-accent">Scenario loss vs NAV</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown(f'<div class="metric-card"><div class="kpi-label">Severe shock</div><div class="kpi-value">{fmt_pct(scenario_losses.get("japan_severe"))}</div><div class="kpi-accent">Scenario loss vs NAV</div></div>', unsafe_allow_html=True)
with col3:
    st.markdown(f'<div class="metric-card"><div class="kpi-label">Extreme shock</div><div class="kpi-value">{fmt_pct(scenario_losses.get("japan_extreme"))}</div><div class="kpi-accent">Scenario loss vs NAV</div></div>', unsafe_allow_html=True)
with col4:
    latest_99 = None if bundle.latest_risk is None else float(bundle.latest_risk.loc['Latest 99% VaR', 'value_pct'])
    st.markdown(f'<div class="metric-card"><div class="kpi-label">Latest 99% VaR</div><div class="kpi-value">{fmt_pct(latest_99)}</div><div class="kpi-accent">Historical rolling estimate</div></div>', unsafe_allow_html=True)

st.markdown('## Portfolio')
a, b = st.columns((1.1, 1.3))
with a:
    st.plotly_chart(figs.get('book_weights', fallback_fig('Book weights')), use_container_width=True)
with b:
    st.plotly_chart(figs.get('scenario_losses', fallback_fig('Scenario losses')), use_container_width=True)

if show_tables:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('### Book table')
        if bundle.book is not None:
            st.dataframe(bundle.book, use_container_width=True)
        else:
            st.info('Book table unavailable.')
    with c2:
        st.markdown('### Scenario table')
        if bundle.scenarios is not None:
            st.dataframe(bundle.scenarios, use_container_width=True)
        else:
            st.info('Scenario table unavailable.')

if show_var:
    st.markdown('## VaR and history')
    c3, c4 = st.columns((1.45, 1.0))
    with c3:
        st.plotly_chart(figs.get('var_backtest', fallback_fig('Daily returns vs VaR')), use_container_width=True)
    with c4:
        st.plotly_chart(figs.get('worst_days', fallback_fig('Worst days')), use_container_width=True)

    c5, c6 = st.columns(2)
    with c5:
        st.markdown('### Latest risk metrics')
        if bundle.latest_risk is not None:
            st.dataframe(bundle.latest_risk, use_container_width=True)
        else:
            st.info('VaR metrics unavailable until book_history.parquet exists.')
    with c6:
        st.markdown('### Backtest summary')
        if bundle.backtest is not None:
            st.dataframe(bundle.backtest, use_container_width=True)
        else:
            st.info('Backtest unavailable until book history exists.')

    st.markdown('### Scenario rarity versus 5-day history')
    st.plotly_chart(figs.get('scenario_vs_history_5d', fallback_fig('Scenario rarity vs 5-day history')), use_container_width=True)

    if show_tables:
        c7, c8 = st.columns(2)
        with c7:
            st.markdown('### Single-day rarity')
            if bundle.scen_hist_1d is not None:
                st.dataframe(bundle.scen_hist_1d, use_container_width=True)
            else:
                st.info('Single-day rarity unavailable.')
        with c8:
            st.markdown('### Five-day rarity')
            if bundle.scen_hist_5d is not None:
                st.dataframe(bundle.scen_hist_5d, use_container_width=True)
            else:
                st.info('Five-day rarity unavailable.')

st.markdown('## Notes')
st.markdown('<div class="section-card">This dashboard is intentionally tolerant of missing files. If returns history is missing, the portfolio and scenario sections still render; once book history is available, the VaR section fills in automatically.</div>', unsafe_allow_html=True)