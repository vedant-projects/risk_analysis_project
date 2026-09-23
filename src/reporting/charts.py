"""charts.py - figure factory for dashboard + PDF report.

Consumes a ReportBundle from report_tables.build_bundle() and emits Plotly
figures plus optional PNG files. Keep this module presentation-only: all
numerical logic should stay in report_tables / models / risk.
"""
from __future__ import annotations

import json
from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go

from src.reporting.report_tables import ReportBundle, build_bundle


def _write(fig: go.Figure, path: str | Path, caption: str, description: str = "") -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(p))
    with open(str(p) + '.meta.json', 'w') as f:
        json.dump({'caption': caption, 'description': description}, f)


def fig_book_weights(bundle: ReportBundle) -> go.Figure:
    df = bundle.book.copy()
    df = df.loc[df.index != 'TOTAL', ['sleeve', 'weight']].reset_index(names='position')
    fig = px.bar(df, x='position', y='weight', color='sleeve',
                 title='Book weights by position (current mix)')
    fig.update_layout(title={'text': 'Book weights by position (current mix)<br><span style="font-size: 18px; font-weight: normal;">Source: portfolio book | allocation and sleeve mix</span>'})
    fig.update_xaxes(title_text='Position')
    fig.update_yaxes(title_text='Weight')
    fig.update_traces(cliponaxis=False)
    fig.update_layout(legend=dict(orientation='h', yanchor='bottom', y=1.05, xanchor='center', x=0.5))
    return fig


def fig_scenario_losses(bundle: ReportBundle) -> go.Figure:
    df = bundle.scenarios.reset_index()
    fig = px.bar(df, x='scenario', y='total_%NAV', color='scenario',
                 title='Scenario losses across Japan shocks (pct NAV)')
    fig.update_layout(title={'text': 'Scenario losses across Japan shocks (pct NAV)<br><span style="font-size: 18px; font-weight: normal;">Source: scenario engine | baseline to extreme loss range</span>'}, showlegend=False)
    fig.update_xaxes(title_text='Scenario')
    fig.update_yaxes(title_text='Loss %NAV')
    fig.update_traces(cliponaxis=False)
    return fig


def fig_var_backtest(bundle: ReportBundle) -> go.Figure:
    if bundle.latest_risk is None:
        raise ValueError('VaR tables unavailable - build_history/book_history missing.')
    from src.risk.var_bridge import load_book_returns, rolling_hist_var_es
    r = load_book_returns()
    roll = rolling_hist_var_es(r)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=roll.index, y=100 * roll['return'], mode='lines', name='Return', fill='tozeroy'))
    fig.add_trace(go.Scatter(x=roll.index, y=-100 * roll['var_95'], mode='lines', name='95% VaR'))
    fig.add_trace(go.Scatter(x=roll.index, y=-100 * roll['var_99'], mode='lines', name='99% VaR'))
    fig.update_layout(title={'text': 'Daily returns vs VaR (rolling 252d)<br><span style="font-size: 18px; font-weight: normal;">Source: book history | realized losses against historical VaR</span>'})
    fig.update_xaxes(title_text='Date')
    fig.update_yaxes(title_text='Return %')
    fig.update_layout(legend=dict(orientation='h', yanchor='bottom', y=1.05, xanchor='center', x=0.5))
    return fig


def fig_worst_days(bundle: ReportBundle) -> go.Figure:
    if bundle.worst_days is None:
        raise ValueError('Worst-days table unavailable - book_history missing.')
    df = (100 * bundle.worst_days.reset_index()).rename(columns={'book_return': 'loss_pct'})
    date_col = df.columns[0]
    fig = px.bar(df, x=date_col, y='loss_pct', title='Worst daily losses in history (pct)')
    fig.update_layout(title={'text': 'Worst daily losses in history (pct)<br><span style="font-size: 18px; font-weight: normal;">Source: book history | most severe realized book days</span>'}, showlegend=False)
    fig.update_xaxes(title_text='Date')
    fig.update_yaxes(title_text='Loss %')
    fig.update_traces(cliponaxis=False)
    return fig


def fig_scenario_vs_history_5d(bundle: ReportBundle) -> go.Figure:
    if bundle.scen_hist_5d is None:
        raise ValueError('5-day scenario/history table unavailable - book_history missing.')
    df = bundle.scen_hist_5d.reset_index()
    fig = px.bar(df, x='scenario', y='share_of_history_pct', color='scenario',
                 title='Scenario rarity vs 5d history (share pct)')
    fig.update_layout(title={'text': 'Scenario rarity vs 5d history (share pct)<br><span style="font-size: 18px; font-weight: normal;">Source: scenario and history bridge | multi-day realism check</span>'}, showlegend=False)
    fig.update_xaxes(title_text='Scenario')
    fig.update_yaxes(title_text='Hist share')
    fig.update_traces(cliponaxis=False)
    return fig


def build_all_figures(bundle: ReportBundle | None = None) -> dict[str, go.Figure]:
    bundle = build_bundle() if bundle is None else bundle
    figs = {}
    if bundle.book is not None:
        figs['book_weights'] = fig_book_weights(bundle)
    if bundle.scenarios is not None:
        figs['scenario_losses'] = fig_scenario_losses(bundle)
    if bundle.latest_risk is not None:
        figs['var_backtest'] = fig_var_backtest(bundle)
        figs['worst_days'] = fig_worst_days(bundle)
        figs['scenario_vs_history_5d'] = fig_scenario_vs_history_5d(bundle)
    return figs


def export_all_charts(out_dir='output/charts', bundle: ReportBundle | None = None) -> list[Path]:
    bundle = build_bundle() if bundle is None else bundle
    figs = build_all_figures(bundle)
    out = []
    captions = {
        'book_weights': ('Book weights by position', 'Bar chart of sleeve and position allocation.'),
        'scenario_losses': ('Japan scenario losses', 'Bar chart of total percent-NAV loss by scenario.'),
        'var_backtest': ('Returns versus VaR', 'Line chart comparing realized daily returns with rolling VaR bands.'),
        'worst_days': ('Worst daily losses', 'Bar chart of the largest realized one-day losses.'),
        'scenario_vs_history_5d': ('Scenario rarity versus 5-day history', 'Bar chart showing how often 5-day history matches scenario severity.'),
    }
    for name, fig in figs.items():
        p = Path(out_dir) / f'{name}.png'
        cap, desc = captions[name]
        _write(fig, p, cap, desc)
        out.append(p)
    return out