import streamlit as st
import numpy as np
import plotly.graph_objects as go
import base64
from pathlib import Path

from probability_engine import (
    COMPONENTS, TESTS, TEST_NAMES, FAULT_STATES,
    BN_EDGES, BN_CONDITIONAL, BN_MARGINAL_PRIORS,
    compute_prior, bayesian_update, simulate_measurement,
    entropy, marginal_fault_probs, expected_fault_count,
    map_fault_state, posterior_confidence, recommend_next_test,
    value_of_information, bayesian_network_edges,
    geometric_tests_to_confidence,
    bootstrap_marginal_cis, posterior_predictive_pdf,
    component_hypothesis_tests,
)

st.set_page_config(
    page_title="CPU Debug Oracle",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: #0a0c10;
    color: #e0e6f0;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem; max-width: 1400px; }

.app-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.4rem; font-weight: 600;
    color: #fff; letter-spacing: -0.5px; margin: 0;
    display: inline-block;
}
.app-sub {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem; color: #3a5070;
    letter-spacing: 1px; text-transform: uppercase;
    margin-left: 16px;
}
.tag-row { margin: 10px 0 22px; display: flex; gap: 7px; flex-wrap: wrap; }
.tag {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.66rem;
    padding: 3px 10px; border: 1px solid #1a3050; border-radius: 3px;
    color: #3a7abd; background: #0a1520; letter-spacing: 0.3px;
}
.hdivider { border: none; border-top: 1px solid #121820; margin: 18px 0; }

.metric-row { display: flex; gap: 10px; margin-bottom: 18px; }
.metric-card {
    flex: 1; background: #0c1018; border: 1px solid #151e2a;
    border-radius: 6px; padding: 14px 18px;
}
.metric-label {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.65rem;
    color: #3a5070; text-transform: uppercase; letter-spacing: 1.2px; margin-bottom: 5px;
}
.metric-value {
    font-family: 'IBM Plex Mono', monospace; font-size: 1.65rem;
    font-weight: 600; color: #d8e8ff; line-height: 1;
}
.metric-sub {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.68rem;
    color: #2a5a3a; margin-top: 4px;
}

.sec-hdr {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.65rem;
    color: #3a5070; text-transform: uppercase; letter-spacing: 2px;
    margin-bottom: 12px; border-bottom: 1px solid #121820; padding-bottom: 7px;
}
.comp-row { display: flex; align-items: center; margin-bottom: 9px; gap: 10px; }
.comp-name {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.73rem;
    color: #6a7a90; width: 162px; flex-shrink: 0;
}
.comp-bar-bg {
    flex: 1; height: 7px; background: #0c1018;
    border: 1px solid #151e2a; border-radius: 2px; overflow: hidden;
}
.comp-bar-fill { height: 100%; border-radius: 2px; }
.comp-prob {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.73rem;
    color: #4a6080; width: 40px; text-align: right; flex-shrink: 0;
}
.test-entry {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.73rem;
    padding: 7px 11px; margin-bottom: 5px; background: #0c1018;
    border: 1px solid #151e2a; border-left: 3px solid #1a4a7a;
    border-radius: 0 4px 4px 0; color: #6a7a90;
}
.result-box {
    background: #0c1018; border: 1px solid; border-radius: 6px;
    padding: 18px 22px; margin-top: 14px;
    font-family: 'IBM Plex Mono', monospace;
}
.stButton > button {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.75rem !important; letter-spacing: 0.3px !important;
    border-radius: 4px !important; border: 1px solid #1a3a5a !important;
    background: #0a1520 !important; color: #3a7abd !important;
    padding: 8px 14px !important;
}
.stButton > button:hover {
    background: #152030 !important; border-color: #3a7abd !important;
    color: #6aa0e0 !important;
}
.stSelectbox label { display: none; }
</style>
""", unsafe_allow_html=True)

def sample_faulty_state(prior):
    faulty = [s for s in FAULT_STATES if any(s)]
    faulty_probs = np.array([prior[s] for s in faulty])
    faulty_probs /= faulty_probs.sum()
    idx = np.random.choice(len(faulty), p=faulty_probs)
    return faulty[idx]

def init_state():
    prior = compute_prior()
    np.random.seed(np.random.randint(0, 9999))
    st.session_state.update({
        "true_state":      sample_faulty_state(prior),
        "posterior":       dict(prior),
        "prior":           dict(prior),
        "tests_run":       [],
        "entropy_history": [entropy(prior)],
        "revealed":        False,
        "game_over":       False,
    })

if "posterior" not in st.session_state:
    init_state()

if "started" not in st.session_state:
    st.session_state["started"] = False

def init_h2h():
    prior = compute_prior()
    np.random.seed(np.random.randint(0, 9999))
    true_state = sample_faulty_state(prior)
    st.session_state.update({
        "h2h_true_state":   true_state,
        "h2h_post_guided":  dict(prior),
        "h2h_post_random":  dict(prior),
        "h2h_prior":        dict(prior),
        "h2h_tests_guided": [],
        "h2h_tests_random": [],
        "h2h_eh_guided":    [entropy(prior)],
        "h2h_eh_random":    [entropy(prior)],
        "h2h_done":         False,
        "h2h_sim_results":  None,
        "h2h_geo_results":  None,
    })

if "h2h_true_state" not in st.session_state:
    init_h2h()

def p_color(p):
    if p < 0.15: return "#1a7a3a"
    if p < 0.40: return "#7a6010"
    if p < 0.70: return "#8a3a10"
    return "#8a1a1a"

_CHRIS_LEVELS = [
    {
        "max_p":       0.15,
        "label":       "ALL SYSTEMS NOMINAL",
        "label_color": "#2a7a4a",
        "border":      "#1a3a2a",
        "emoji":       "😊",
        "image":       "assets/chris_0_nominal.jpg",
        "messages": [
            "All systems nominal. This chip gets an A+!",
            "P(fault) = 0. Great job!",
            "Looking good! I would accept this chip as a pset submission.",
        ],
    },
    {
        "max_p":       0.40,
        "label":       "MILD CONCERN",
        "label_color": "#9a8a10",
        "border":      "#3a3010",
        "emoji":       "🤔",
        "image":       "assets/chris_1_thinking.jpg",
        "messages": [
            "Mild degradation detected. I've seen worse on the CS106 to CS109 transition.",
            "Something's off. Consider reviewing the Bayesian Network lecture slides.",
            "Hmm. Not ideal, but a few more tests should clear things up.",
            "Slight concern. This reminds me of students who skip the first three weeks.",
        ],
    },
    {
        "max_p":       0.70,
        "label":       "ELEVATED FAULT RISK",
        "label_color": "#cc7733",
        "border":      "#4a2a10",
        "emoji":       "😰",
        "image":       "assets/chris_2_worried.jpg",
        "messages": [
            "Significant fault probability. This is giving me midterm grading flashbacks.",
            "We have a problem. I'm pulling up the office hours Zoom link.",
            "P(this chip is fine | data) << 0.5. Uh Oh.",
        ],
    },
    {
        "max_p":       1.01,
        "label":       "CRITICAL FAILURE",
        "label_color": "#cc3333",
        "border":      "#4a1010",
        "emoji":       "😱",
        "image":       "assets/chris_3_panic.jpg",
        "messages": [
            "CRITICAL. This chip needs an extension request.",
            "Full system failure. Even a Naive Bayes classifier saw this coming.",
            "The posterior has spoken. This chip is deader than a flat prior.",
            "P(this chip works | data) = 0. Time to drop and shop.",
            "Maximum entropy achieved... in the wrong direction. Catastrophic.",
        ],
    },
]

def _load_img_b64(path: str) -> str | None:
    """Return a base64 data-URI for the image at path, or None if missing."""
    try:
        data = Path(path).read_bytes()
        ext  = Path(path).suffix.lstrip(".").lower()
        mime = "jpeg" if ext in ("jpg", "jpeg") else ext
        return f"data:image/{mime};base64,{base64.b64encode(data).decode()}"
    except FileNotFoundError:
        return None

def chris_reaction_html(posterior: dict, n_tests_run: int) -> str:
    """
    Build the full Chris Gregg reaction panel as an HTML string.

    Severity is driven by the worst-case marginal fault probability.
    Message is picked deterministically from n_tests_run so it stays
    stable between Streamlit rerenders but changes after each test.
    """
    marg    = marginal_fault_probs(posterior)
    worst_p = max(marg.values())

    level = _CHRIS_LEVELS[-1]
    for lvl in _CHRIS_LEVELS:
        if worst_p <= lvl["max_p"]:
            level = lvl
            break

    msg      = level["messages"][n_tests_run % len(level["messages"])]
    img_src  = _load_img_b64(level["image"])
    lc       = level["label_color"]
    border   = level["border"]

    if img_src:
        face_html = (
            f'<img src="{img_src}" style="width:88px;height:88px;'
            f'object-fit:cover;object-position:center;border-radius:50%;'
            f'border:2px solid {lc};flex-shrink:0">'
        )
    else:
        face_html = (
            f'<div style="font-size:3.6rem;line-height:88px;width:88px;'
            f'text-align:center;flex-shrink:0">{level["emoji"]}</div>'
        )

    return f"""
<div style="
    display:flex;align-items:center;gap:20px;
    background:#0c1018;border:1px solid {border};
    border-left:4px solid {lc};border-radius:8px;
    padding:16px 22px;margin-bottom:20px">
  {face_html}
  <div style="flex:1;min-width:0">
    <div style="
        font-family:'IBM Plex Mono',monospace;font-size:0.58rem;
        color:{lc};text-transform:uppercase;letter-spacing:2px;margin-bottom:7px">
      ◈ System Status: {level["label"]}
      &nbsp;·&nbsp; Max P(fault) = {worst_p:.3f}
    </div>
    <div style="
        font-family:'IBM Plex Mono',monospace;font-size:0.85rem;
        color:#d0dff0;line-height:1.55">
      {msg}
    </div>
    <div style="
        font-family:'IBM Plex Mono',monospace;font-size:0.58rem;
        color:#2a4060;margin-top:7px">
      — chris gregg
    </div>
  </div>
</div>
"""

def comp_bars_html(m):
    h = '<div class="sec-hdr">Marginal Fault Probabilities</div>'
    for c in COMPONENTS:
        p  = m[c]
        w  = int(p * 100)
        cl = p_color(p)
        h += f'''<div class="comp-row">
          <div class="comp-name">{c}</div>
          <div class="comp-bar-bg"><div class="comp-bar-fill" style="width:{w}%;background:{cl}"></div></div>
          <div class="comp-prob">{p:.3f}</div>
        </div>'''
    return h

def hist_html(tests, eh):
    if not tests:
        return '<div style="font-family:\'IBM Plex Mono\',monospace;font-size:0.73rem;color:#1e2e40;padding:10px">No tests run yet.</div>'
    h = '<div class="sec-hdr">Test History</div>'
    for i, (name, val) in enumerate(tests):
        dh = eh[i+1] - eh[i]
        h += f'''<div class="test-entry">
          <span style="color:#3a7abd">{i+1}. {name}</span>
          &nbsp;→&nbsp;<span style="color:#76b900">{val:.3f}</span>
          &nbsp;<span style="color:#2a5a4a">ΔH={dh:+.3f}</span>
        </div>'''
    return h

def floorplan(m):
    fig = go.Figure()
    fig.update_layout(
        height=340, margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="#0a0c10", plot_bgcolor="#0a0c10",
        xaxis=dict(range=[0, 10], showticklabels=False, showgrid=False,
                   zeroline=False, fixedrange=True),
        yaxis=dict(range=[0, 7],  showticklabels=False, showgrid=False,
                   zeroline=False, fixedrange=True),
        dragmode=False, showlegend=False,
    )
    fig.add_shape(type="rect", x0=0.1, y0=0.1, x1=9.9, y1=6.9,
                  fillcolor="#0c1018", line=dict(color="#1e2e40", width=1.5))

    blocks = [
        (0.3,  3.8, 3.5,  6.7, "Out-of-Order Engine", "Out-of-Order Engine"),
        (3.7,  3.8, 6.5,  6.7, "L1 Cache",            "L1 Cache (per-core)"),
        (6.7,  3.8, 9.7,  6.7, "Branch Predictor",    "Branch Predictor"),
        (0.3,  2.2, 9.7,  3.6, "L2/L3 Cache",         "L2/L3 Cache  (shared LLC)"),
        (0.3,  0.3, 5.3,  2.0, "Memory Controller",   "Memory Controller"),
        (5.5,  0.3, 9.7,  2.0, "PCIe Interface",       "PCIe Interface"),
    ]

    block_centers = {}
    for x0, y0, x1, y1, comp, label in blocks:
        p  = m[comp]
        cx = (x0 + x1) / 2
        cy = (y0 + y1) / 2
        block_centers[comp] = (cx, cy, x0, y0, x1, y1)
        fig.add_shape(type="rect", x0=x0, y0=y0, x1=x1, y1=y1,
                      fillcolor=p_color(p),
                      line=dict(color="#0a0c10", width=1.5), opacity=0.72)
        fig.add_annotation(x=cx, y=cy + 0.18, text=f"<b>{label}</b>",
                           showarrow=False,
                           font=dict(size=9.5, color="#c8d8f0",
                                     family="IBM Plex Mono"))
        fig.add_annotation(x=cx, y=cy - 0.28,
                           text=f"P(fault) = {p:.3f}",
                           showarrow=False,
                           font=dict(size=8.5, color="#7090b0",
                                     family="IBM Plex Mono"))

    mc_cx,  _, _, _, mc_x1,  mc_y1  = block_centers["Memory Controller"]
    _,      _, _, l2_y0, _,  _      = block_centers["L2/L3 Cache"]
    fig.add_annotation(
        x=mc_cx, y=l2_y0 - 0.05,
        ax=mc_cx, ay=mc_y1 + 0.05,
        xref="x", yref="y", axref="x", ayref="y",
        arrowhead=2, arrowsize=1.0, arrowwidth=2.0,
        arrowcolor="#2a6a3a", text="",
    )
    ooo_cx, ooo_cy, _, _, ooo_x1, _ = block_centers["Out-of-Order Engine"]
    l1_cx,  l1_cy,  l1_x0, _, _, _  = block_centers["L1 Cache"]
    fig.add_annotation(
        x=l1_x0 + 0.05, y=l1_cy,
        ax=ooo_x1 - 0.05, ay=ooo_cy,
        xref="x", yref="y", axref="x", ayref="y",
        arrowhead=2, arrowsize=1.0, arrowwidth=2.0,
        arrowcolor="#2a6a3a", text="",
    )
    bp_cx, bp_cy, bp_x0, _, _, _ = block_centers["Branch Predictor"]
    fig.add_annotation(
        x=bp_x0 + 0.05, y=bp_cy + 0.5,
        ax=ooo_x1 - 0.05, ay=ooo_cy + 0.5,
        xref="x", yref="y", axref="x", ayref="y",
        arrowhead=2, arrowsize=1.0, arrowwidth=2.0,
        arrowcolor="#2a6a3a", text="",
    )

    fig.add_annotation(
        x=0.3, y=0.06,
        text="<span style='color:#2a6a3a'>→</span> Bayesian network dependency",
        showarrow=False, xref="x", yref="y",
        font=dict(size=8, color="#2a5a3a", family="IBM Plex Mono"),
        xanchor="left",
    )

    return fig

def bn_diagram():
    """
    Compact plotly visualization of the fault dependency DAG.
    Shows nodes, directed edges, and conditional probabilities.
    """
    fig = go.Figure()
    fig.update_layout(
        height=175, margin=dict(l=0, r=0, t=5, b=5),
        paper_bgcolor="#0a0c10", plot_bgcolor="#0a0c10",
        xaxis=dict(range=[0, 10], showticklabels=False, showgrid=False,
                   zeroline=False, fixedrange=True),
        yaxis=dict(range=[0, 3],  showticklabels=False, showgrid=False,
                   zeroline=False, fixedrange=True),
        dragmode=False, showlegend=False,
    )

    pos = {
        "Memory Controller":   (1.8, 2.2),
        "L2/L3 Cache":        (1.8, 0.7),
        "Out-of-Order Engine": (5.0, 2.2),
        "Branch Predictor":    (3.6, 0.7),
        "L1 Cache":            (6.4, 0.7),
        "PCIe Interface":      (8.5, 1.45),
    }

    edges = [
        ("Memory Controller",   "L2/L3 Cache",        "P=.30|fault\nP=.10|ok"),
        ("Out-of-Order Engine", "Branch Predictor",   "P=.28|fault\nP=.07|ok"),
        ("Out-of-Order Engine", "L1 Cache",           "P=.25|fault\nP=.08|ok"),
    ]

    for parent, child, label in edges:
        px, py = pos[parent]
        cx, cy = pos[child]
        fig.add_annotation(
            x=cx, y=cy + 0.28, ax=px, ay=py - 0.28,
            xref="x", yref="y", axref="x", ayref="y",
            arrowhead=2, arrowsize=0.9, arrowwidth=1.8,
            arrowcolor="#2a6a3a", text="",
        )
        mid_x = (px + cx) / 2 + 0.3
        mid_y = (py + cy) / 2
        fig.add_annotation(
            x=mid_x, y=mid_y, text=label, showarrow=False,
            font=dict(size=7, color="#2a6a3a", family="IBM Plex Mono"),
            align="left",
        )

    for name, (x, y) in pos.items():
        is_root = name in BN_MARGINAL_PRIORS
        fill    = "#0a1520" if is_root else "#0a1f10"
        border  = "#1a3a5a" if is_root else "#1a4a2a"
        fig.add_shape(type="rect", x0=x-1.1, y0=y-0.28,
                      x1=x+1.1, y1=y+0.28,
                      fillcolor=fill, line=dict(color=border, width=1.5))
        node_label = f"{'Root' if is_root else 'Child'} · {name}"
        fig.add_annotation(x=x, y=y + 0.07, text=f"<b>{name}</b>",
                           showarrow=False,
                           font=dict(size=8.5, color="#c0d0e8",
                                     family="IBM Plex Mono"))
        type_label = "independent root" if is_root else "conditional child"
        fig.add_annotation(x=x, y=y - 0.13, text=type_label,
                           showarrow=False,
                           font=dict(size=7, color="#3a5a3a" if not is_root else "#2a4a6a",
                                     family="IBM Plex Mono"))
    return fig

def entropy_chart(eh):
    n   = len(eh)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(n)), y=eh, mode="lines+markers",
        line=dict(color="#1a5a8a", width=2.5),
        marker=dict(size=7, color="#3a90d9",
                    line=dict(color="#0a0c10", width=1.5)),
        fill="tozeroy", fillcolor="rgba(10,30,60,0.35)",
    ))
    fig.add_hline(y=0, line_dash="dot", line_color="#151e2a")
    fig.update_layout(
        height=155, margin=dict(l=40, r=10, t=8, b=30),
        paper_bgcolor="#0a0c10", plot_bgcolor="#0c1018",
        xaxis=dict(title=dict(text="tests run",
                              font=dict(family="IBM Plex Mono", size=8)),
                   color="#2a4060", gridcolor="#0c1420",
                   tickfont=dict(family="IBM Plex Mono", size=8),
                   range=[-0.3, max(5.7, n - 0.5)],
                   tickmode="linear", dtick=1),
        yaxis=dict(title=dict(text="H", font=dict(family="IBM Plex Mono", size=8)),
                   color="#2a4060", gridcolor="#0c1420",
                   tickfont=dict(family="IBM Plex Mono", size=8),
                   rangemode="tozero"),
        font=dict(family="IBM Plex Mono"),
    )
    return fig

def voi_chart(scores, best):
    names  = list(scores.keys())
    vals   = list(scores.values())
    colors = ["#4a8a10" if n == best else "#152010" for n in names]
    tc     = ["#99d020" if n == best else "#2a4a15" for n in names]
    fig = go.Figure(go.Bar(
        x=vals, y=names, orientation="h",
        marker_color=colors,
        text=[f"{v:.4f}" for v in vals],
        textposition="outside",
        textfont=dict(family="IBM Plex Mono", size=8, color=tc),
    ))
    fig.update_layout(
        height=195, margin=dict(l=10, r=55, t=8, b=30),
        paper_bgcolor="#0a0c10", plot_bgcolor="#0c1018",
        xaxis=dict(title=dict(text="VoI (bits)",
                              font=dict(family="IBM Plex Mono", size=8)),
                   color="#2a4060", gridcolor="#0c1420",
                   tickfont=dict(family="IBM Plex Mono", size=8)),
        yaxis=dict(color="#5a7090",
                   tickfont=dict(family="IBM Plex Mono", size=8)),
        font=dict(family="IBM Plex Mono"), bargap=0.38,
    )
    return fig

def posterior_predictive_chart(test_name, posterior):
    x, pdf, p_deg, p_hlt = posterior_predictive_pdf(test_name, posterior)
    test  = TESTS[test_name]
    mu_h  = test["mu_healthy"]
    mu_d  = test["mu_degraded"]
    sigma = test["sigma"]
    import scipy.stats as ss

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=p_hlt * ss.norm.pdf(x, mu_h, sigma), mode="lines",
        name=f"Healthy (w={p_hlt:.2f})",
        line=dict(color="#1a7a3a", width=1.5, dash="dot"),
        fill="tozeroy", fillcolor="rgba(20,80,30,0.12)",
    ))
    fig.add_trace(go.Scatter(
        x=x, y=p_deg * ss.norm.pdf(x, mu_d, sigma), mode="lines",
        name=f"Degraded (w={p_deg:.2f})",
        line=dict(color="#8a2a10", width=1.5, dash="dot"),
        fill="tozeroy", fillcolor="rgba(100,30,10,0.12)",
    ))
    fig.add_trace(go.Scatter(
        x=x, y=pdf, mode="lines",
        name="P(x) — mixture",
        line=dict(color="#4a90d9", width=2.5),
    ))
    for mu, label, color in [
        (mu_h, "healthy", "#1a7a3a"),
        (mu_d, "degraded", "#8a2a10"),
    ]:
        fig.add_vline(x=mu, line_dash="dash", line_color=color, line_width=1,
                      opacity=0.5, annotation_text=label,
                      annotation_font=dict(size=7.5, color=color,
                                           family="IBM Plex Mono"),
                      annotation_position="top")
    fig.update_layout(
        height=195, margin=dict(l=10, r=10, t=30, b=30),
        paper_bgcolor="#0a0c10", plot_bgcolor="#0c1018",
        legend=dict(font=dict(family="IBM Plex Mono", size=7.5, color="#5a7090"),
                    bgcolor="rgba(0,0,0,0)", orientation="h", yanchor="bottom", y=1.02),
        xaxis=dict(title=dict(text=f"measurement ({test_name})",
                              font=dict(family="IBM Plex Mono", size=8)),
                   color="#2a4060", gridcolor="#0c1420",
                   tickfont=dict(family="IBM Plex Mono", size=8)),
        yaxis=dict(title=dict(text="density",
                              font=dict(family="IBM Plex Mono", size=8)),
                   color="#2a4060", gridcolor="#0c1420",
                   tickfont=dict(family="IBM Plex Mono", size=8)),
        font=dict(family="IBM Plex Mono"),
    )
    return fig

def comp_bars_with_ci_html(marg, cis, htests):
    h = '<div class="sec-hdr">Marginal Fault Probabilities + 95% Bootstrap CI</div>'
    for c in COMPONENTS:
        p      = marg[c]
        lo, hi = cis[c]
        w      = int(p * 100)
        w_lo   = int(lo * 100)
        w_hi   = min(int(hi * 100), 100)
        cl     = p_color(p)
        ht     = htests[c]
        if ht["reject"]:
            badge = (f'<span style="font-family:\'IBM Plex Mono\',monospace;font-size:0.60rem;'
                     f'background:#2a1a0a;border:1px solid #7a3a10;color:#cc7733;padding:1px 6px;'
                     f'border-radius:2px;margin-left:6px">p={ht["p_value"]:.2f} · reject H₀</span>')
        elif ht["evidence"] != "none":
            badge = (f'<span style="font-family:\'IBM Plex Mono\',monospace;font-size:0.60rem;'
                     f'background:#0a1a0a;border:1px solid #1a4a1a;color:#3a7a3a;padding:1px 6px;'
                     f'border-radius:2px;margin-left:6px">p={ht["p_value"]:.2f}</span>')
        else:
            badge = (f'<span style="font-family:\'IBM Plex Mono\',monospace;font-size:0.60rem;'
                     f'color:#1e3050;margin-left:6px">p={ht["p_value"]:.2f}</span>')

        h += f"""<div style="margin-bottom:11px">
  <div style="display:flex;align-items:center;margin-bottom:3px">
    <div class="comp-name">{c}</div>
    <div style="font-family:'IBM Plex Mono',monospace;font-size:0.73rem;
                color:#4a6080;width:40px;text-align:right">{p:.3f}</div>
    {badge}
  </div>
  <div style="position:relative;height:10px;background:#0c1018;
              border:1px solid #151e2a;border-radius:2px;
              margin-left:162px">
    <div style="position:absolute;left:0;top:1px;height:8px;width:{w}%;
                background:{cl};border-radius:2px"></div>
    <div style="position:absolute;left:{w_lo}%;top:3px;height:4px;
                width:{w_hi - w_lo}%;background:rgba(255,255,255,0.15);
                border-left:2px solid rgba(255,255,255,0.4);
                border-right:2px solid rgba(255,255,255,0.4)"></div>
  </div>
  <div style="margin-left:162px;font-family:'IBM Plex Mono',monospace;
              font-size:0.60rem;color:#2a4060;margin-top:2px">
    95% CI [{lo:.3f}, {hi:.3f}]
  </div>
</div>"""
    return h

def geometric_pmf_chart(geo_results):
    """
    Overlay of theoretical Geometric(p̂) PMF against empirical
    simulation histogram, for both guided and random strategies.
    """
    support = geo_results["guided"]["pmf_support"]
    fig = go.Figure()

    for strategy, color_emp, color_theory, label in [
        ("guided", "rgba(74,138,16,0.55)", "#76b900", "Entropy-Guided"),
        ("random", "rgba(138,58,32,0.55)", "#cc5533", "Random"),
    ]:
        r = geo_results[strategy]
        if r["mean_T"] is None:
            continue
        fig.add_trace(go.Bar(
            x=support, y=r["empirical_pmf"],
            name=f"{label} (empirical)",
            marker_color=color_emp,
            width=0.35 if strategy == "guided" else 0.35,
            offset=-0.35 if strategy == "guided" else 0,
        ))
        fig.add_trace(go.Scatter(
            x=support, y=r["pmf_values"], mode="lines+markers",
            name=f"Geometric(p̂={r['p_hat']:.3f}), E[T]={r['mean_T']:.1f}",
            line=dict(color=color_theory, width=2, dash="dot"),
            marker=dict(size=5),
        ))

    fig.update_layout(
        height=235, margin=dict(l=40, r=10, t=15, b=40),
        paper_bgcolor="#0a0c10", plot_bgcolor="#0c1018",
        barmode="group",
        legend=dict(font=dict(family="IBM Plex Mono", size=8, color="#5a7090"),
                    bgcolor="rgba(0,0,0,0)", orientation="h",
                    yanchor="bottom", y=1.01),
        xaxis=dict(title=dict(text="T (tests until 90% confidence)",
                              font=dict(family="IBM Plex Mono", size=8)),
                   color="#2a4060", gridcolor="#0c1420",
                   tickfont=dict(family="IBM Plex Mono", size=8),
                   tickmode="linear", dtick=1),
        yaxis=dict(title=dict(text="P(T = k)",
                              font=dict(family="IBM Plex Mono", size=8)),
                   color="#2a4060", gridcolor="#0c1420",
                   tickfont=dict(family="IBM Plex Mono", size=8)),
        font=dict(family="IBM Plex Mono"),
    )
    return fig

def run_h2h_simulation(n_trials=300, n_tests=5):
    prior = compute_prior()
    results = {"guided": [], "random": []}
    np.random.seed(42)
    for _ in range(n_trials):
        true_state = sample_faulty_state(prior)
        for strategy in ["guided", "random"]:
            post = dict(prior)
            done = []
            tests_to_90 = None
            for step in range(n_tests):
                remaining = [t for t in TEST_NAMES if t not in done]
                if not remaining:
                    break
                if strategy == "guided":
                    voi  = {t: value_of_information(post, t, n_samples=60)
                            for t in remaining}
                    test = max(voi, key=voi.get)
                else:
                    test = np.random.choice(remaining)
                m = simulate_measurement(test, true_state)
                post = bayesian_update(post, test, m)
                done.append(test)
                if tests_to_90 is None and posterior_confidence(post) >= 0.90:
                    tests_to_90 = step + 1
            correct = map_fault_state(post) == true_state
            results[strategy].append({
                "correct":    correct,
                "confidence": posterior_confidence(post),
                "tests_to_90": tests_to_90,
            })
    return results

if not st.session_state["started"]:
    st.markdown("""
<div style="display:flex;flex-direction:column;align-items:center;
            justify-content:center;min-height:72vh;text-align:center;padding:40px 20px">

  <div style="font-family:'IBM Plex Mono',monospace;font-size:2rem;
              color:#2a5a8a;margin-bottom:48px;letter-spacing:6px">⬡</div>

  <div style="max-width:520px;margin-bottom:40px">
    <p style="font-family:'IBM Plex Mono',monospace;font-size:0.82rem;
              color:#5a7a9a;line-height:2.1;margin:0">
      A CPU fails validation.<br>
      Six components, any subset could be broken.<br>
      Running every diagnostic is expensive.<br>
      Running them in the wrong order is worse.
    </p>
  </div>

  <div style="font-family:'IBM Plex Mono',monospace;font-weight:600;
              font-size:5.5rem;color:#1a3a5a;line-height:1;margin-bottom:8px">32</div>
  <div style="font-family:'IBM Plex Mono',monospace;font-size:0.62rem;
              color:#2a4a6a;letter-spacing:3px;text-transform:uppercase;
              margin-bottom:48px">fault configurations</div>

  <div style="font-family:'IBM Plex Mono',monospace;font-size:0.82rem;
              color:#3a6a5a;line-height:1;margin-bottom:64px;letter-spacing:0.5px">
    Bayesian inference picks which test to run next.
  </div>

  <div style="font-family:'IBM Plex Mono',monospace;font-size:1.6rem;
              font-weight:600;color:#e8f0ff;letter-spacing:-0.5px;margin-bottom:48px">
    Bayesian Chip Debug Oracle
  </div>

</div>
""", unsafe_allow_html=True)

    col_l, col_mid, col_r = st.columns([2, 1, 2])
    with col_mid:
        if st.button("start →", use_container_width=True, key="start_btn"):
            st.session_state["started"] = True
            st.rerun()
    st.stop()

st.markdown("""
<div style="margin-bottom:18px">
  <span class="app-title">⬡ Bayesian CPU Debug Oracle</span>
  <span class="app-sub">CS109 · Stanford</span>
</div>
<hr class="hdivider">
""", unsafe_allow_html=True)

tab_single, tab_h2h = st.tabs(["◈  Single Session", "⇄  Head-to-Head"])

with tab_single:
    ent  = entropy(st.session_state.posterior)
    ent0 = st.session_state.entropy_history[0]
    conf = posterior_confidence(st.session_state.posterior)
    ef   = expected_fault_count(st.session_state.posterior)
    nd   = len(st.session_state.tests_run)
    dh   = ent - ent0
    dh_s = f"↓ {abs(dh):.3f} from start" if dh < 0 else ("— no change" if dh == 0 else f"↑ {dh:.3f}")
    dh_c = "#2a5a3a" if dh <= 0 else "#5a2a2a"

    st.markdown(f"""
    <div class="metric-row">
      <div class="metric-card">
        <div class="metric-label">entropy</div>
        <div class="metric-value">{ent:.3f}</div>
        <div class="metric-sub" style="color:{dh_c}">{dh_s}</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">confidence</div>
        <div class="metric-value">{conf:.1%}</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">E[faults]</div>
        <div class="metric-value">{ef:.2f}</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">tests</div>
        <div class="metric-value">{nd} / {len(TEST_NAMES)}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sec-hdr">Chris-ometer</div>', unsafe_allow_html=True)
    st.markdown(
        chris_reaction_html(st.session_state.posterior, nd),
        unsafe_allow_html=True,
    )

    left, _, right = st.columns([1.05, 0.04, 1])
    marg = marginal_fault_probs(st.session_state.posterior)

    with left:
        st.markdown('<div class="sec-hdr">floorplan</div>', unsafe_allow_html=True)
        st.plotly_chart(floorplan(marg), use_container_width=True,
                        config={"displayModeBar": False})
        st.markdown(comp_bars_html(marg), unsafe_allow_html=True)

        st.markdown('<div class="sec-hdr" style="margin-top:18px">fault dag</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(bn_diagram(), use_container_width=True,
                        config={"displayModeBar": False})

    with right:
        st.markdown('<div class="sec-hdr">entropy</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(entropy_chart(st.session_state.entropy_history),
                        use_container_width=True, config={"displayModeBar": False})

        done_names = [t for t, _ in st.session_state.tests_run]
        remaining  = [t for t in TEST_NAMES if t not in done_names]

        if remaining and not st.session_state.game_over:
            st.markdown('<div class="sec-hdr" style="margin-top:14px">Value of Information</div>',
                        unsafe_allow_html=True)
            with st.spinner(""):
                voi = {t: value_of_information(
                           st.session_state.posterior, t, n_samples=180)
                       for t in remaining}
            best = max(voi, key=voi.get)
            st.plotly_chart(voi_chart(voi, best), use_container_width=True,
                            config={"displayModeBar": False})

            c1, c2 = st.columns(2)
            with c1:
                if st.button(f"▶ {best}", key="single_run_best", use_container_width=True):
                    m = simulate_measurement(best, st.session_state.true_state)
                    st.session_state.posterior = bayesian_update(
                        st.session_state.posterior, best, m)
                    st.session_state.tests_run.append((best, m))
                    st.session_state.entropy_history.append(
                        entropy(st.session_state.posterior))
                    if len(st.session_state.tests_run) >= len(TEST_NAMES):
                        st.session_state.game_over = True
                    st.rerun()
            with c2:
                manual = st.selectbox("", remaining, label_visibility="collapsed")
                if st.button("▶ run", key="single_run_manual", use_container_width=True):
                    m = simulate_measurement(manual, st.session_state.true_state)
                    st.session_state.posterior = bayesian_update(
                        st.session_state.posterior, manual, m)
                    st.session_state.tests_run.append((manual, m))
                    st.session_state.entropy_history.append(
                        entropy(st.session_state.posterior))
                    if len(st.session_state.tests_run) >= len(TEST_NAMES):
                        st.session_state.game_over = True
                    st.rerun()

            st.markdown('<div class="sec-hdr" style="margin-top:18px">predictive</div>',
                        unsafe_allow_html=True)
            preview_test = st.selectbox("Preview test", remaining,
                                        key="pp_select", label_visibility="collapsed")
            st.plotly_chart(posterior_predictive_chart(preview_test,
                            st.session_state.posterior),
                            use_container_width=True, config={"displayModeBar": False})

        if st.session_state.tests_run:
            st.markdown('<div style="margin-top:14px"></div>', unsafe_allow_html=True)
            st.markdown(hist_html(st.session_state.tests_run,
                                  st.session_state.entropy_history),
                        unsafe_allow_html=True)

    st.markdown('<hr class="hdivider">', unsafe_allow_html=True)

    b1, b2, _ = st.columns([1, 1, 5])
    with b1:
        if st.button("reveal", key="single_reveal", use_container_width=True):
            st.session_state.revealed = True
    with b2:
        if st.button("↺ New chip", key="single_reset", use_container_width=True):
            init_state(); st.rerun()

    if st.session_state.revealed or st.session_state.game_over:
        ts   = st.session_state.true_state
        tf   = [COMPONENTS[i] for i, b in enumerate(ts) if b] or ["none"]
        ms   = map_fault_state(st.session_state.posterior)
        pred = [COMPONENTS[i] for i, b in enumerate(ms) if b] or ["none"]
        ok   = ms == ts
        bc   = "#1a3a1a" if ok else "#3a1a1a"
        tc   = "#76b900"  if ok else "#cc3333"
        icon = "✓ Correct" if ok else "✗ Incorrect"

        st.markdown(f"""
        <div class="result-box" style="border-color:{bc}">
          <div style="font-family:IBM Plex Mono,monospace;font-size:0.8rem;
                      font-weight:600;color:{tc};margin-bottom:10px">{icon}</div>
          <div style="font-family:IBM Plex Mono,monospace;font-size:0.73rem;
                      color:#4a6080;margin-bottom:5px">
            True faults &nbsp;&nbsp; <span style="color:#c0d0e8">{", ".join(tf)}</span></div>
          <div style="font-family:IBM Plex Mono,monospace;font-size:0.73rem;
                      color:#4a6080;margin-bottom:5px">
            MAP diagnosis <span style="color:#c0d0e8">{", ".join(pred)}</span></div>
          <div style="font-family:IBM Plex Mono,monospace;font-size:0.73rem;color:#4a6080">
            Confidence &nbsp;&nbsp;&nbsp; <span style="color:#c0d0e8">{conf:.1%}</span>
            &nbsp;&nbsp; Tests &nbsp; <span style="color:#c0d0e8">{nd}/{len(TEST_NAMES)}</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

    if nd >= 2:
        with st.expander("bootstrap · hypothesis tests"):
            with st.spinner(""):
                cis    = bootstrap_marginal_cis(
                    st.session_state.tests_run, st.session_state.prior)
                htests = component_hypothesis_tests(st.session_state.posterior)
            st.markdown(
                comp_bars_with_ci_html(marg, cis, htests), unsafe_allow_html=True)

with tab_h2h:

    def step_h2h(n_samples_voi=120):
        ts = st.session_state.h2h_true_state

        remaining_g = [t for t in TEST_NAMES
                        if t not in [x for x, _ in st.session_state.h2h_tests_guided]]
        if remaining_g:
            voi    = {t: value_of_information(
                         st.session_state.h2h_post_guided, t, n_samples=n_samples_voi)
                      for t in remaining_g}
            best_g = max(voi, key=voi.get)
            m_g    = simulate_measurement(best_g, ts)
            st.session_state.h2h_post_guided = bayesian_update(
                st.session_state.h2h_post_guided, best_g, m_g)
            st.session_state.h2h_tests_guided.append((best_g, m_g))
            st.session_state.h2h_eh_guided.append(
                entropy(st.session_state.h2h_post_guided))

        remaining_r = [t for t in TEST_NAMES
                        if t not in [x for x, _ in st.session_state.h2h_tests_random]]
        if remaining_r:
            rand_t = np.random.choice(remaining_r)
            m_r    = simulate_measurement(rand_t, ts)
            st.session_state.h2h_post_random = bayesian_update(
                st.session_state.h2h_post_random, rand_t, m_r)
            st.session_state.h2h_tests_random.append((rand_t, m_r))
            st.session_state.h2h_eh_random.append(
                entropy(st.session_state.h2h_post_random))

        if len(st.session_state.h2h_tests_guided) >= len(TEST_NAMES):
            st.session_state.h2h_done = True

    hc1, hc2, hc3, hc4 = st.columns([1, 1, 1, 4])
    with hc1:
        if st.button("▶ next", key="h2h_next", use_container_width=True,
                     disabled=st.session_state.h2h_done):
            step_h2h(); st.rerun()
    with hc2:
        if st.button("▶▶ run all", key="h2h_runall", use_container_width=True,
                     disabled=st.session_state.h2h_done):
            while not st.session_state.h2h_done:
                step_h2h(n_samples_voi=25)
            st.rerun()
    with hc3:
        if st.button("↺ new chip", key="h2h_reset", use_container_width=True):
            init_h2h(); st.rerun()

    n_rounds = len(st.session_state.h2h_tests_guided)
    st.markdown(
        f'<div style="font-family:IBM Plex Mono,monospace;font-size:0.68rem;'
        f'color:#2a4060;margin-bottom:16px">Round {n_rounds} / {len(TEST_NAMES)}</div>',
        unsafe_allow_html=True)
    st.markdown('<hr class="hdivider">', unsafe_allow_html=True)

    hcol_g, hcol_sep, hcol_r = st.columns([1, 0.03, 1])

    def side_panel(label, label_color, post, tests, eh, true_state, col):
        with col:
            marg     = marginal_fault_probs(post)
            ent_val  = entropy(post)
            conf_val = posterior_confidence(post)

            st.markdown(
                f'<div style="font-family:IBM Plex Mono,monospace;'
                f'font-size:0.85rem;font-weight:600;color:{label_color};'
                f'margin-bottom:12px;border-bottom:2px solid {label_color};'
                f'padding-bottom:6px">{label}</div>',
                unsafe_allow_html=True)

            st.markdown(f"""
<div class="metric-row" style="gap:8px">
  <div class="metric-card" style="padding:12px 14px">
    <div class="metric-label">Entropy</div>
    <div class="metric-value" style="font-size:1.3rem">{ent_val:.3f}</div>
  </div>
  <div class="metric-card" style="padding:12px 14px">
    <div class="metric-label">Confidence</div>
    <div class="metric-value" style="font-size:1.3rem">{conf_val:.1%}</div>
  </div>
</div>""", unsafe_allow_html=True)

            for comp in COMPONENTS:
                p  = marg[comp]
                w  = int(p * 100)
                cl = p_color(p)
                st.markdown(f'''<div class="comp-row">
  <div class="comp-name">{comp}</div>
  <div class="comp-bar-bg"><div class="comp-bar-fill"
       style="width:{w}%;background:{cl}"></div></div>
  <div class="comp-prob">{p:.3f}</div>
</div>''', unsafe_allow_html=True)

            if tests:
                st.markdown('<div style="margin-top:12px"></div>', unsafe_allow_html=True)
                for i, (name, val) in enumerate(tests):
                    dh = eh[i + 1] - eh[i]
                    st.markdown(
                        f'<div class="test-entry">'
                        f'<span style="color:{label_color}">{i+1}. {name}</span>'
                        f' → <span style="color:#76b900">{val:.3f}</span>'
                        f' <span style="color:#2a5a4a">ΔH={dh:+.3f}</span></div>',
                        unsafe_allow_html=True)

    side_panel("◈ Entropy-Guided", "#76b900",
               st.session_state.h2h_post_guided,
               st.session_state.h2h_tests_guided,
               st.session_state.h2h_eh_guided,
               st.session_state.h2h_true_state, hcol_g)

    with hcol_sep:
        st.markdown(
            '<div style="border-left:1px solid #121820;height:600px;margin:0 auto"></div>',
            unsafe_allow_html=True)

    side_panel("✕ Random", "#cc5533",
               st.session_state.h2h_post_random,
               st.session_state.h2h_tests_random,
               st.session_state.h2h_eh_random,
               st.session_state.h2h_true_state, hcol_r)

    st.markdown('<hr class="hdivider">', unsafe_allow_html=True)
    st.markdown('<div class="sec-hdr">entropy trajectory</div>',
                unsafe_allow_html=True)

    eh_g = st.session_state.h2h_eh_guided
    eh_r = st.session_state.h2h_eh_random

    fig_cmp = go.Figure()
    fig_cmp.add_trace(go.Scatter(
        x=list(range(len(eh_g))), y=eh_g, mode="lines+markers",
        name="Entropy-Guided",
        line=dict(color="#4a8a10", width=2.5),
        marker=dict(size=7, color="#76b900",
                    line=dict(color="#0a0c10", width=1.5)),
    ))
    fig_cmp.add_trace(go.Scatter(
        x=list(range(len(eh_r))), y=eh_r, mode="lines+markers",
        name="Random",
        line=dict(color="#8a3a20", width=2.5),
        marker=dict(size=7, color="#cc5533",
                    line=dict(color="#0a0c10", width=1.5)),
    ))
    fig_cmp.add_hline(y=0, line_dash="dot", line_color="#1a2535")
    fig_cmp.update_layout(
        height=200, margin=dict(l=40, r=20, t=10, b=40),
        paper_bgcolor="#0a0c10", plot_bgcolor="#0c1018",
        legend=dict(font=dict(family="IBM Plex Mono", size=9, color="#6a8aaa"),
                    bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(title=dict(text="tests run",
                              font=dict(family="IBM Plex Mono", size=8)),
                   color="#2a4060", gridcolor="#0c1420",
                   tickfont=dict(family="IBM Plex Mono", size=8),
                   tickmode="linear", dtick=1,
                   range=[-0.3, len(TEST_NAMES) + 0.3]),
        yaxis=dict(title=dict(text="H(posterior)",
                              font=dict(family="IBM Plex Mono", size=8)),
                   color="#2a4060", gridcolor="#0c1420",
                   tickfont=dict(family="IBM Plex Mono", size=8),
                   rangemode="tozero"),
        font=dict(family="IBM Plex Mono"),
    )
    st.plotly_chart(fig_cmp, use_container_width=True,
                    config={"displayModeBar": False})

    if st.session_state.h2h_done:
        ts     = st.session_state.h2h_true_state
        tf     = [COMPONENTS[i] for i, b in enumerate(ts) if b] or ["none"]
        map_g  = map_fault_state(st.session_state.h2h_post_guided)
        map_r  = map_fault_state(st.session_state.h2h_post_random)
        conf_g = posterior_confidence(st.session_state.h2h_post_guided)
        conf_r = posterior_confidence(st.session_state.h2h_post_random)

        rc1, rc2 = st.columns(2)
        for col, ok, conf_v, label, lc in [
            (rc1, map_g == ts, conf_g, "Entropy-Guided", "#76b900"),
            (rc2, map_r == ts, conf_r, "Random",         "#cc5533"),
        ]:
            with col:
                bc = "#1a3a1a" if ok else "#3a1a1a"
                st.markdown(f"""
<div class="result-box" style="border-color:{bc}">
  <div style="font-family:IBM Plex Mono,monospace;font-size:0.8rem;
              font-weight:600;color:{lc};margin-bottom:8px">{label}</div>
  <div style="font-family:IBM Plex Mono,monospace;font-size:0.73rem;color:#4a6080;margin-bottom:4px">
    True faults &nbsp; <span style="color:#c0d0e8">{", ".join(tf)}</span></div>
  <div style="font-family:IBM Plex Mono,monospace;font-size:1.2rem;
              color:{"#76b900" if ok else "#cc3333"};margin-top:8px">
    {"✓ Correct" if ok else "✗ Incorrect"}</div>
  <div style="font-family:IBM Plex Mono,monospace;font-size:0.73rem;
              color:#4a6080;margin-top:4px">Confidence {conf_v:.1%}</div>
</div>""", unsafe_allow_html=True)

    st.markdown('<hr class="hdivider">', unsafe_allow_html=True)
    st.markdown('<div class="sec-hdr">simulation · 300 trials</div>',
                unsafe_allow_html=True)

    if st.button("▶ simulate", key="h2h_simulate"):
        with st.spinner(""):
            st.session_state.h2h_sim_results = run_h2h_simulation(n_trials=300, n_tests=5)
        st.rerun()

    if st.session_state.h2h_sim_results is not None:
        res     = st.session_state.h2h_sim_results
        acc_g   = sum(r["correct"] for r in res["guided"]) / len(res["guided"])
        acc_r   = sum(r["correct"] for r in res["random"]) / len(res["random"])
        avg_cg  = sum(r["confidence"] for r in res["guided"]) / len(res["guided"])
        avg_cr  = sum(r["confidence"] for r in res["random"]) / len(res["random"])
        t90_g   = [r["tests_to_90"] for r in res["guided"] if r["tests_to_90"] is not None]
        t90_r   = [r["tests_to_90"] for r in res["random"] if r["tests_to_90"] is not None]
        avg_t90_g = sum(t90_g) / len(t90_g) if t90_g else None
        avg_t90_r = sum(t90_r) / len(t90_r) if t90_r else None
        improvement = (acc_g - acc_r) / max(acc_r, 0.001) * 100

        st.markdown(f"""
<div class="metric-row">
  <div class="metric-card">
    <div class="metric-label" style="color:#4a8a10">guided</div>
    <div class="metric-value" style="color:#76b900">{acc_g:.1%}</div>
  </div>
  <div class="metric-card">
    <div class="metric-label" style="color:#8a3a20">random</div>
    <div class="metric-value" style="color:#cc5533">{acc_r:.1%}</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">improvement</div>
    <div class="metric-value">+{improvement:.0f}%</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">tests to 90%</div>
    <div class="metric-value">{f"{avg_t90_g:.1f}" if avg_t90_g else "—"} vs {f"{avg_t90_r:.1f}" if avg_t90_r else "—"}</div>
  </div>
</div>
""", unsafe_allow_html=True)

        fig_acc = go.Figure(go.Bar(
            x=["Entropy-Guided", "Random"],
            y=[acc_g, acc_r],
            marker_color=["#4a8a10", "#8a3a20"],
            text=[f"{acc_g:.1%}", f"{acc_r:.1%}"],
            textposition="outside",
            textfont=dict(family="IBM Plex Mono", size=11,
                          color=["#76b900", "#cc5533"]),
            width=0.4,
        ))
        fig_acc.update_layout(
            height=240, margin=dict(l=20, r=20, t=20, b=40),
            paper_bgcolor="#0a0c10", plot_bgcolor="#0c1018",
            yaxis=dict(range=[0, 1], tickformat=".0%",
                       color="#2a4060", gridcolor="#0c1420",
                       tickfont=dict(family="IBM Plex Mono", size=9),
                       title=dict(text="accuracy (5 tests)",
                                  font=dict(family="IBM Plex Mono", size=8))),
            xaxis=dict(color="#5a7090",
                       tickfont=dict(family="IBM Plex Mono", size=10)),
            font=dict(family="IBM Plex Mono"), showlegend=False,
        )
        st.plotly_chart(fig_acc, use_container_width=True,
                        config={"displayModeBar": False})

    st.markdown('<hr class="hdivider">', unsafe_allow_html=True)
    st.markdown('<div class="sec-hdr">geometric distribution</div>',
                unsafe_allow_html=True)

    if st.button("▶ fit · 600 trials", key="h2h_geometric"):
        with st.spinner(""):
            st.session_state.h2h_geo_results = geometric_tests_to_confidence(
                target_confidence=0.90,
                n_trials=600,
                strategies=("guided", "random"),
                n_samples_voi=20,
            )
        st.rerun()

    if st.session_state.h2h_geo_results is not None:
        geo = st.session_state.h2h_geo_results
        g   = geo["guided"]
        r   = geo["random"]

        st.plotly_chart(geometric_pmf_chart(geo), use_container_width=True,
                        config={"displayModeBar": False})

        st.markdown(f"""
<div class="metric-row">
  <div class="metric-card">
    <div class="metric-label" style="color:#4a8a10">guided E[T]</div>
    <div class="metric-value" style="color:#76b900">{g["mean_T"]:.2f}</div>
    <div class="metric-sub">p̂ = {g["p_hat"]:.3f}  ·  σ = {g["std_T"]:.2f}</div>
  </div>
  <div class="metric-card">
    <div class="metric-label" style="color:#8a3a20">random E[T]</div>
    <div class="metric-value" style="color:#cc5533">{r["mean_T"]:.2f}</div>
    <div class="metric-sub">p̂ = {r["p_hat"]:.3f}  ·  σ = {r["std_T"]:.2f}</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">saved</div>
    <div class="metric-value">{r["mean_T"] - g["mean_T"]:.2f}</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">never reached</div>
    <div class="metric-value">{r["pct_never_reached"]:.1%}</div>
  </div>
</div>
""", unsafe_allow_html=True)