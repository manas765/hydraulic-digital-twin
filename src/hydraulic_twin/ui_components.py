"""HTML building blocks for the dashboard.

Everything here is theme-neutral: text inherits the page colour and panels
use translucent grey, so it reads correctly on both light and dark Streamlit
themes. Only the header band and the badges use fixed colours, and they carry
their own background so they never depend on the page.
"""
from html import escape

from .dashboard_data import SOURCE_LABELS

STYLE = """<style>
.block-container { padding-top: 4.5rem; }
.hx-header { display: flex; justify-content: space-between;
  flex-wrap: wrap; align-items: center; gap: .6rem 1rem; background: #14304d; color: #ffffff;
  border-bottom: 3px solid #3b8eea; border-radius: 4px;
  padding: .85rem 1.1rem; margin-bottom: 1.1rem; }
.hx-header > div:first-child { flex: 1 1 15rem; min-width: 0; }
.hx-title { font-size: 1.3rem; font-weight: 700; color: #ffffff;
  line-height: 1.25; }
.hx-subtitle { color: #b8c7d8; font-size: .83rem; margin-top: .2rem; }
.hx-mode { font-size: .68rem; font-weight: 700; letter-spacing: .06em;
  color: #1b2733; background: #f2b84b; padding: .3rem .6rem;
  border-radius: 3px; white-space: nowrap; }
.hx-section { font-size: .76rem; font-weight: 700; letter-spacing: .08em;
  text-transform: uppercase; opacity: .78; margin: 1.2rem 0 .6rem;
  padding-bottom: .3rem; border-bottom: 1px solid rgba(128,140,155,.35); }
.hx-grid { display: grid; gap: .75rem; margin: .3rem 0 .4rem;
  grid-template-columns: repeat(auto-fit, minmax(185px, 1fr)); }
.hx-card { background: rgba(128,140,155,.10); color: inherit;
  border: 1px solid rgba(128,140,155,.32); border-top: 3px solid #3b8eea;
  border-radius: 4px; padding: .65rem .8rem; }
.hx-card.tone-ok { border-top-color: #2e9e5b; }
.hx-card.tone-minor { border-top-color: #d9a520; }
.hx-card.tone-major { border-top-color: #e8821a; }
.hx-card.tone-critical { border-top-color: #e5484d; }
.hx-card.tone-none { border-top-color: #8a97a6; }
.hx-badge-row { margin-bottom: .4rem; }
.hx-label { font-size: .7rem; font-weight: 600; text-transform: uppercase;
  letter-spacing: .05em; opacity: .75; white-space: nowrap; }
.hx-value { font-size: 1.5rem; font-weight: 700; margin: .25rem 0 .1rem;
  font-variant-numeric: tabular-nums; }
.hx-sub { font-size: .76rem; opacity: .72; min-height: 2.3rem;
  line-height: 1.25; }
.hx-badge { font-size: .6rem; font-weight: 700; letter-spacing: .05em;
  padding: .12rem .42rem; border-radius: 2px; white-space: nowrap;
  color: #ffffff; }
.hx-dataset { background: #1f8a4c; }
.hx-ml { background: #b8620a; }
.hx-sim { background: #1f6fd1; }
.hx-user { background: #7a4fd6; }
.hx-alert { display: flex; flex-wrap: wrap; gap: .1rem .7rem;
  align-items: baseline;
  padding: .42rem .65rem; margin: .25rem 0; font-size: .87rem;
  color: inherit; background: rgba(128,140,155,.12);
  border-left: 4px solid #8a97a6; border-radius: 0 3px 3px 0; }
.hx-alert.CRITICAL { border-left-color: #e5484d;
  background: rgba(229,72,77,.15); }
.hx-alert.WARNING { border-left-color: #f0a020;
  background: rgba(240,160,32,.15); }
.hx-alert.INFO { border-left-color: #3b8eea;
  background: rgba(59,142,234,.13); }
.hx-alert-level { font-weight: 700; font-size: .68rem; min-width: 4.3rem;
  letter-spacing: .05em; }
.hx-alert-src { font-size: .7rem; min-width: 10.5rem; opacity: .7; }
.hx-alert-msg { flex: 1 1 12rem; min-width: 0; overflow-wrap: anywhere; }
.hx-footer { font-size: .74rem; opacity: .6; margin-top: 1.6rem;
  padding-top: .6rem; border-top: 1px solid rgba(128,140,155,.35); }
@media (max-width: 640px) {
  .hx-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); gap: .5rem; }
  .hx-card { padding: .5rem .6rem; }
  .hx-value { font-size: 1.2rem; }
  .hx-sub { min-height: 0; font-size: .72rem; }
  .hx-title { font-size: 1.1rem; }
  .hx-subtitle { font-size: .78rem; }
  .hx-alert-src { min-width: 0; }
}
</style>"""

BADGES = {
    "dataset": ("DATASET", "hx-dataset"),
    "ml": ("ML ESTIMATE", "hx-ml"),
    "sim": ("SIMULATED", "hx-sim"),
    "user": ("USER SETTING", "hx-user"),
}


def header_html(title: str, subtitle: str, mode_text: str) -> str:
    return (
        '<div class="hx-header"><div>'
        f'<div class="hx-title">{escape(title)}</div>'
        f'<div class="hx-subtitle">{escape(subtitle)}</div></div>'
        f'<div class="hx-mode">{escape(mode_text)}</div></div>'
    )


def section_html(title: str) -> str:
    return f'<div class="hx-section">{escape(title)}</div>'


def card(label, value, kind, sub="", tone="none") -> str:
    text, css = BADGES[kind]
    return (
        f'<div class="hx-card tone-{escape(tone)}">'
        f'<div class="hx-badge-row"><span class="hx-badge {css}">{text}'
        '</span></div>'
        f'<div class="hx-label">{escape(label)}</div>'
        f'<div class="hx-value">{escape(value)}</div>'
        f'<div class="hx-sub">{escape(sub)}</div></div>'
    )


def card_grid(cards) -> str:
    return '<div class="hx-grid">' + "".join(cards) + "</div>"


def alerts_html(alerts) -> str:
    if not alerts:
        return ('<div class="hx-alert"><span class="hx-alert-level">NONE'
                '</span><span>No alerts for this cycle.</span></div>')
    rows = []
    for a in alerts:
        source = SOURCE_LABELS.get(a["source"], a["source"])
        level = escape(a["level"])
        rows.append(
            f'<div class="hx-alert {level}">'
            f'<span class="hx-alert-level">{level}</span>'
            f'<span class="hx-alert-src">{escape(source)}</span>'
            f'<span class="hx-alert-msg">{escape(a["message"])}</span></div>'
        )
    return "".join(rows)


def footer_html(text: str) -> str:
    return f'<div class="hx-footer">{escape(text)}</div>'
