# app.py
import re
import json
from collections import Counter

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

# ------------------------------
# Constants & Helpers
# ------------------------------

MIS_LABELLED_SET = {
    "mis-labelled", "mis-labeled", "misbranded", "mis-branded", "mis branded", "mis labelled", "mis labeled"
}

COMPLIANT_TOKEN = "compliant as per fssr"
SUBSTANDARD_TOKEN = "sub-standard"
UNSAFE_TOKEN = "unsafe"


def split_parameters(cell_value: str) -> list:
    """Split a text cell containing one or more parameter names."""
    if cell_value is None:
        return []
    text = str(cell_value).strip()
    if not text:
        return []
    parts = re.split(r"[\n\r;|,]", text)
    return [p.strip(" ,;") for p in parts if p.strip()]


def format_node_label(title: str, count: int | None = None, pct: float | None = None) -> str:
    if count is not None and pct is not None:
        return f"{title}\\n[{count} Samples; {pct:.1f}%]"
    if count is not None:
        unit = "Sample" if count == 1 else "Samples"
        return f"{title}\\n[{count} {unit}]"
    return title


def build_tree_dot(df: pd.DataFrame, commodity: str, variant_choice: str, settings: dict) -> tuple[str, dict]:
    """Build DOT source string and return (dot_src, stats)."""
    cols = {
        "commodity": "Commodity",
        "variant2": "Variant 2",
        "overall_compliance": "Overall Compliance",
        "overall_quality": "Overall Quality Classification",
        "overall_safety": "Overall Safety Classification",
        "overall_labelling": "Overall Labelling Complaince",
        "substandard_cases": "Sub-Standard Cases",
        "unsafe_cases": "Unsafe Cases",
        "test_type": "Test Type",
        "parameter": "Parameter",
    }

    dfx = df[df[cols["commodity"]] == commodity].copy()
    dfx = dfx[dfx[cols["variant2"]].fillna("(missing)") == variant_choice]
    n_total = len(dfx)
    if n_total == 0:
        raise ValueError(f"No rows for {commodity}/{variant_choice}")

    # classification counts
    comp_series = dfx[cols["overall_compliance"]].astype(str).str.strip().str.lower()
    n_compliant = (comp_series == COMPLIANT_TOKEN).sum()
    n_noncompliant = n_total - n_compliant

    qual_series = dfx[cols["overall_quality"]].astype(str).str.strip().str.lower()
    saf_series = dfx[cols["overall_safety"]].astype(str).str.strip().str.lower()
    lab_series = dfx[cols["overall_labelling"]].astype(str).str.strip().str.lower()

    n_quality_sub = (qual_series == SUBSTANDARD_TOKEN).sum()
    n_safety_unsafe = (saf_series == UNSAFE_TOKEN).sum()
    n_lab_mis = lab_series.apply(lambda s: any(x in s for x in MIS_LABELLED_SET)).sum()

    # collect parameters (grouped by test type)
    qual_grouped = {}
    if n_quality_sub:
        sub_df = dfx[qual_series == SUBSTANDARD_TOKEN]
        for _, row in sub_df.iterrows():
            ttype = str(row.get(cols["test_type"], "Other"))
            for p in split_parameters(row.get(cols["parameter"])):
                qual_grouped.setdefault(ttype, []).append(p)

    saf_grouped = {}
    if n_safety_unsafe:
        saf_df = dfx[saf_series == UNSAFE_TOKEN]
        for _, row in saf_df.iterrows():
            ttype = str(row.get(cols["test_type"], "Other"))
            for p in split_parameters(row.get(cols["parameter"])):
                saf_grouped.setdefault(ttype, []).append(p)

    def pct(n: int) -> float:
        return (n / n_total * 100.0) if n_total else 0.0

    # Style settings
    rankdir = settings.get("rankdir", "TB")
    fontname = settings.get("fontname", "Helvetica")
    fontsize = settings.get("fontsize", 12)
    node_shape = settings.get("node_shape", "box")
    nodesep = settings.get("nodesep", 0.5)
    ranksep = settings.get("ranksep", 0.6)
    default_color = settings.get("default_color", "#ffffff")
    compliant_color = settings.get("compliant_color", "#d4edda")
    noncompliant_color = settings.get("noncompliant_color", "#f8d7da")

    # Build DOT
    dot_lines = [
        "digraph G {",
        f"  rankdir={rankdir};",
        f"  graph [splines=ortho, nodesep={nodesep}, ranksep={ranksep}];",
        ("  node [shape={shape}, style=\"rounded,filled\", color=\"#d0d0d0\", "
         "fillcolor=\"{fill}\", fontname=\"{font}\", fontsize={fs}];").format(
            shape=node_shape, fill=default_color, font=fontname, fs=fontsize
        ),
        f"  edge [fontname=\"{fontname}\", fontsize={max(8, fontsize-2)} , arrowhead=normal];",
    ]

    # Nodes
    root_id = "root"
    comp_id = "comp"
    noncomp_id = "noncomp"
    qual_id = "qual"
    saf_id = "saf"
    lab_id = "lab"

    root_label = format_node_label(f"{commodity} - {variant_choice}", n_total, 100.0)
    comp_label = format_node_label("Compliant", n_compliant, pct(n_compliant))
    noncomp_label = format_node_label("Non-Compliant", n_noncompliant, pct(n_noncompliant))

    dot_lines.append(f'  {root_id} [label="{root_label}"];')
    dot_lines.append(f'  {comp_id} [label="{comp_label}", fillcolor="{compliant_color}"];')
    dot_lines.append(f'  {noncomp_id} [label="{noncomp_label}", fillcolor="{noncompliant_color}"];')
    dot_lines.append(
        f'  {qual_id} [label="{format_node_label("Quality Parameters", n_quality_sub, pct(n_quality_sub))}"];'
    )
    dot_lines.append(
        f'  {saf_id} [label="{format_node_label("Safety Parameters", n_safety_unsafe, pct(n_safety_unsafe))}"];'
    )
    dot_lines.append(
        f'  {lab_id} [label="{format_node_label("Labelling Issues", n_lab_mis, pct(n_lab_mis))}"];'
    )

    # Edges
    dot_lines += [
        f"  {root_id} -> {comp_id};",
        f"  {root_id} -> {noncomp_id};",
        f"  {noncomp_id} -> {qual_id};",
        f"  {noncomp_id} -> {saf_id};",
        f"  {noncomp_id} -> {lab_id};",
    ]

    # Add grouped parameters
    idx = 0
    for ttype, params in qual_grouped.items():
        tnode = f"qt{idx}"
        dot_lines.append(f'  {tnode} [label="{ttype}"];')
        dot_lines.append(f"  {qual_id} -> {tnode};")
        for p in Counter(params).items():
            pname, cnt = p
            pid = f"q{idx}"
            dot_lines.append(f'  {pid} [label="{format_node_label(pname, cnt)}"];')
            dot_lines.append(f"  {tnode} -> {pid};")
            idx += 1

    idx = 0
    for ttype, params in saf_grouped.items():
        tnode = f"st{idx}"
        dot_lines.append(f'  {tnode} [label="{ttype}"];')
        dot_lines.append(f"  {saf_id} -> {tnode};")
        for p in Counter(params).items():
            pname, cnt = p
            pid = f"s{idx}"
            dot_lines.append(f'  {pid} [label="{format_node_label(pname, cnt)}"];')
            dot_lines.append(f"  {tnode} -> {pid};")
            idx += 1

    dot_lines.append("}")
    dot_src = "\n".join(dot_lines)

    stats = {
        "total": n_total,
        "variant": variant_choice,
        "compliant": n_compliant,
        "non_compliant": n_noncompliant,
        "quality_substandard": n_quality_sub,
        "safety_unsafe": n_safety_unsafe,
        "labelling_mis": n_lab_mis,
    }

    return dot_src, stats


# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="Decision Tree Chart Generator", layout="wide")
st.title("🌳 Decision Tree Chart Generator")
st.markdown("Upload Excel/CSV → choose commodity → customize tree in the sidebar → download SVG/PNG")

# Upload
uploaded = st.file_uploader("Upload dataset (Excel .xlsx or CSV)", type=["xlsx", "xls", "csv"])
if uploaded is None:
    st.info("Upload your dataset to start. Required columns: Commodity, Variant 2, Overall Compliance, Overall Quality Classification, Sub-Standard Cases, Overall Safety Classification, Unsafe Cases, Overall Labelling Complaince, Test Type, Parameter")
    st.stop()

# Read file
if uploaded.name.lower().endswith((".xlsx", ".xls")):
    df = pd.read_excel(uploaded, engine="openpyxl")
else:
    df = pd.read_csv(uploaded)

required_cols = [
    "Commodity", "Variant 2", "Overall Compliance", "Overall Safety Classification",
    "Overall Quality Classification", "Sub-Standard Cases", "Unsafe Cases", "Overall Labelling Complaince",
    "Test Type", "Parameter"
]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    st.error("Missing required columns: " + ", ".join(missing))
    st.stop()

# Commodity selection
commodities = sorted(df["Commodity"].dropna().unique().tolist())
commodity = st.selectbox("Commodity", commodities, index=0)

# Sidebar customization controls
st.sidebar.header("⚙️ Chart Appearance")
rankdir_choice = st.sidebar.radio("Orientation", ("Top → Bottom (TB)", "Left → Right (LR)"))
rankdir = "TB" if rankdir_choice.startswith("Top") else "LR"

fontname = st.sidebar.selectbox("Font family", ["Helvetica", "Arial", "Times New Roman", "Courier"])
fontsize = st.sidebar.slider("Node font size", 8, 20, 12)
node_shape = st.sidebar.selectbox("Node shape", ["box", "ellipse", "oval", "circle"])
nodesep = st.sidebar.slider("Node spacing (nodesep)", 0.1, 2.0, 0.5)
ranksep = st.sidebar.slider("Rank separation (ranksep)", 0.1, 2.0, 0.6)
default_color = st.sidebar.color_picker("Default node fill", "#ffffff")
compliant_color = st.sidebar.color_picker("Compliant node color", "#d4edda")
noncompliant_color = st.sidebar.color_picker("Non-compliant node color", "#f8d7da")
preview_height = st.sidebar.slider("Preview iframe height (px)", 300, 1400, 700)

settings = {
    "rankdir": rankdir,
    "fontname": fontname,
    "fontsize": fontsize,
    "node_shape": node_shape,
    "nodesep": nodesep,
    "ranksep": ranksep,
    "default_color": default_color,
    "compliant_color": compliant_color,
    "noncompliant_color": noncompliant_color,
}

# Variants under commodity
variants_all = df.loc[df["Commodity"] == commodity, "Variant 2"].fillna("(missing)").unique().tolist()

# ------------------------------
# Build charts per variant
# ------------------------------
for variant_choice in variants_all:
    st.subheader(f"📊 {commodity} – {variant_choice}")
    try:
        dot_src, stats = build_tree_dot(df, commodity, variant_choice, settings)
    except Exception as e:
        st.warning(str(e))
        continue

    # Stats
    st.markdown(f"**Samples:** {stats['total']} • ✅ Compliant: {stats['compliant']} • ❌ Non-compliant: {stats['non_compliant']}")

    # Download DOT
    st.download_button(
        f"⬇️ Download .dot ({variant_choice})",
        data=dot_src.encode("utf-8"),
        file_name=f"{commodity}_{variant_choice}.dot",
        mime="text/vnd.graphviz",
    )

    # Client-side Viz.js preview (SVG + PNG downloads)
    viz_html = f"""
    <!doctype html>
    <html>
    <head>
      <meta charset="utf-8"/>
      <style>body {{ margin: 0; }}</style>
    </head>
    <body>
      <div id="viz">Rendering...</div>
      <script src="https://unpkg.com/viz.js@2.1.2/viz.js"></script>
      <script src="https://unpkg.com/viz.js@2.1.2/full.render.js"></script>
      <script>
        const dot = {json.dumps(dot_src)};
        const viz = new Viz();
        viz.renderSVGElement(dot).then(function(element) {{
          document.getElementById('viz').innerHTML = '';
          document.getElementById('viz').appendChild(element);
        }});
      </script>
    </body>
    </html>
    """
    components.html(viz_html, height=preview_height, scrolling=True)































































































































# import io
# import re
# import base64
# import pandas as pd
# import streamlit as st
# from PIL import Image
# from graphviz import Digraph

# # ------------------------------
# # Helpers
# # ------------------------------

# def _norm(s: str) -> str:
#     if pd.isna(s):
#         return ""
#     return str(s).strip().lower()

# MIS_LABELLED_SET = {
#     "mis-labelled", "mis-labeled", "misbranded", "mis-branded", "mis branded", "mis labelled", "mis labeled"
# }

# COMPLIANT_TOKEN = "compliant as per fssr"
# SUBSTANDARD_TOKEN = "sub-standard"
# UNSAFE_TOKEN = "unsafe"
# PARAM_END_TOKEN = "compliance"


# def split_parameters(cell_value: str) -> list:
#     if pd.isna(cell_value) or str(cell_value).strip() == "":
#         return []

#     text = str(cell_value)
#     if PARAM_END_TOKEN.lower() in text.lower():
#         pieces = re.split(r"(?i)\bcompliance\b", text)
#         return [p.strip(",; ") for p in pieces if p.strip(",; ")]
#     parts = re.split(r"[\n\r;|]", text)
#     out = [p.strip(", ") for p in parts if p.strip()]
#     return out or [p.strip() for p in re.split(r",\s{2,}|,", text) if p.strip()]


# def format_node_label(title: str, count: int | None = None, pct: float | None = None) -> str:
#     line1 = title
#     if count is not None and pct is not None:
#         return f"{line1}\n[{count} Samples; {pct:.1f}%]"
#     elif count is not None:
#         unit = "Sample" if count == 1 else "Samples"
#         return f"{line1}\n[{count} {unit}]"
#     return line1


# def build_tree_dot(df: pd.DataFrame, commodity: str, variant_choice: str | None = None) -> tuple[str, dict]:
#     cols = {
#         "commodity": "Commodity",
#         "variant2": "Variant 2",
#         "overall_compliance": "Overall Compliance",
#         "overall_quality": "Overall Quality Classification",
#         "overall_safety": "Overall Safety Classification",
#         "overall_labelling": "Overall Labelling Complaince",
#         "substandard_cases": "Sub-Standard Cases",
#         "unsafe_cases": "Unsafe Cases",
#     }

#     dfx = df[df[cols["commodity"]] == commodity].copy()
#     if dfx.empty:
#         raise ValueError(f"No rows for commodity: {commodity}")

#     variant_counts = dfx[cols["variant2"]].fillna("(missing)").value_counts()
#     if variant_choice is None:
#         variant_choice = "Packed Samples" if "Packed Samples" in variant_counts.index else variant_counts.index[0]

#     dfx = dfx[dfx[cols["variant2"]].fillna("(missing)") == variant_choice]
#     n_total = len(dfx)

#     comp_series = dfx[cols["overall_compliance"]].astype(str).str.strip().str.lower()
#     n_compliant = (comp_series == COMPLIANT_TOKEN).sum()
#     n_noncompliant = n_total - n_compliant

#     qual_series = dfx[cols["overall_quality"]].astype(str).str.strip().str.lower()
#     saf_series = dfx[cols["overall_safety"]].astype(str).str.strip().str.lower()
#     lab_series = dfx[cols["overall_labelling"]].astype(str).str.strip().str.lower()

#     n_quality_sub = (qual_series == SUBSTANDARD_TOKEN).sum()
#     n_safety_unsafe = (saf_series == UNSAFE_TOKEN).sum()
#     n_lab_mis = lab_series.apply(lambda s: any(x in s for x in MIS_LABELLED_SET)).sum()

#     from collections import Counter
#     qual_params, saf_params = [], []
#     if n_quality_sub:
#         for _, row in dfx[qual_series == SUBSTANDARD_TOKEN].iterrows():
#             qual_params.extend(split_parameters(row.get(cols["substandard_cases"])))
#     if n_safety_unsafe:
#         for _, row in dfx[saf_series == UNSAFE_TOKEN].iterrows():
#             saf_params.extend(split_parameters(row.get(cols["unsafe_cases"])))
#     qual_param_counts = Counter([p.strip() for p in qual_params if p])
#     saf_param_counts = Counter([p.strip() for p in saf_params if p])

#     def pct(n: int) -> float:
#         return (n / n_total * 100.0) if n_total else 0.0

#     dot = [
#         "digraph G {",
#         "  rankdir=TB;",
#         "  graph [splines=ortho, nodesep=0.5, ranksep=0.6];",
#         "  node [shape=box, style=\"rounded,filled\", color=\"#d0d0d0\", fillcolor=white, fontname=Helvetica];",
#         "  edge [arrowhead=normal];",
#     ]

#     root_id, variant_id, comp_id, noncomp_id, qual_id, saf_id, lab_id = "root","variant","comp","noncomp","qual","saf","lab"
#     root_label = format_node_label(f"{commodity}", n_total, 100.0)
#     variant_label = format_node_label(f"{variant_choice}", n_total, 100.0)

#     dot += [
#         f'  {root_id} [label="{root_label}"];',
#         f'  {variant_id} [label="{variant_label}"];',
#         f'  {comp_id} [label="{format_node_label("Compliant", n_compliant, pct(n_compliant))}", fillcolor="#d4edda"];',
#         f'  {noncomp_id} [label="{format_node_label("Non-Compliant", n_noncompliant, pct(n_noncompliant))}", fillcolor="#f8d7da"];',
#         f'  {qual_id} [label="{format_node_label("Quality Parameters\\nNon-Compliance", n_quality_sub, pct(n_quality_sub))}"];',
#         f'  {saf_id} [label="{format_node_label("Safety Parameters\\nNon-Compliance", n_safety_unsafe, pct(n_safety_unsafe))}"];',
#         f'  {lab_id} [label="{format_node_label("Labelling Attributes\\nNon-Compliance", n_lab_mis, pct(n_lab_mis))}"];',
#         f"  {root_id} -> {variant_id};",
#         f"  {variant_id} -> {comp_id};",
#         f"  {variant_id} -> {noncomp_id};",
#         f"  {noncomp_id} -> {qual_id};",
#         f"  {noncomp_id} -> {saf_id};",
#         f"  {noncomp_id} -> {lab_id};",
#     ]

#     for i,(p,cnt) in enumerate(qual_param_counts.items()):
#         nid=f"q{i}"
#         dot.append(f'  {nid} [label="{format_node_label(p,cnt)}"];')
#         dot.append(f"  {qual_id} -> {nid};")

#     for i,(p,cnt) in enumerate(saf_param_counts.items()):
#         nid=f"s{i}"
#         dot.append(f'  {nid} [label="{format_node_label(p,cnt)}"];')
#         dot.append(f"  {saf_id} -> {nid};")

#     dot.append("}")
#     return "\n".join(dot), {
#         "total": n_total,
#         "variant": variant_choice,
#         "compliant": n_compliant,
#         "non_compliant": n_noncompliant,
#         "quality_substandard": n_quality_sub,
#         "safety_unsafe": n_safety_unsafe,
#         "labelling_mis": n_lab_mis,
#     }


# # ------------------------------
# # Streamlit UI
# # ------------------------------

# st.set_page_config(page_title="Decision Tree Chart Generator", layout="wide")
# st.title("🌳 Decision Tree Chart Generator")
# st.caption("Upload Excel/CSV → Select commodity → Auto-generate compliance tree")

# uploaded = st.file_uploader("Upload dataset", type=["xlsx", "xls", "csv"])

# if uploaded is None:
#     st.info("Upload your dataset to begin.")
#     st.stop()

# if uploaded.name.endswith((".xlsx", ".xls")):
#     df = pd.read_excel(uploaded, engine="openpyxl")
# else:
#     df = pd.read_csv(uploaded)

# required_cols = [
#     "Commodity", "Variant 2", "Overall Compliance", "Overall Safety Classification",
#     "Overall Quality Classification", "Sub-Standard Cases", "Unsafe Cases", "Overall Labelling Complaince"
# ]
# missing = [c for c in required_cols if c not in df.columns]
# if missing:
#     st.error("Missing required columns: " + ", ".join(missing))
#     st.stop()

# commodities = sorted(df["Commodity"].dropna().unique().tolist())
# commodity = st.selectbox("Commodity", commodities)
# variants_all = df.loc[df["Commodity"] == commodity, "Variant 2"].fillna("(missing)").unique().tolist()
# variant_choice = st.selectbox("Variant 2", variants_all)

# try:
#     dot_src, stats = build_tree_dot(df, commodity, variant_choice)
# except Exception as e:
#     st.exception(e)
#     st.stop()

# # PREVIEW (always works)
# st.subheader("Decision Tree Preview")
# st.graphviz_chart(dot_src, use_container_width=True)

# # ------------------------------
# # DOWNLOADS
# # ------------------------------

# st.download_button("⬇️ Download .dot file", data=dot_src.encode("utf-8"),
#                    file_name=f"{commodity}_decision_tree.dot", mime="text/vnd.graphviz")

# # Convert to SVG (base64)
# try:
#     g = Digraph()
#     g.source = dot_src
#     svg_bytes = g.pipe(format="svg")
#     st.download_button("⬇️ Download as SVG", data=svg_bytes,
#                        file_name=f"{commodity}_decision_tree.svg", mime="image/svg+xml")

#     # Convert SVG → PNG via Pillow
#     import cairosvg
#     png_bytes = cairosvg.svg2png(bytestring=svg_bytes)
#     st.download_button("⬇️ Download as PNG", data=png_bytes,
#                        file_name=f"{commodity}_decision_tree.png", mime="image/png")
# except Exception as e:
#     st.warning("SVG/PNG export may fail on Streamlit Cloud if Graphviz is not fully installed.")
























# import io
# import re
# import pandas as pd
# import streamlit as st
# from graphviz import Digraph

# try:
#     import graphviz
# except Exception as e:
#     graphviz = None

# # ------------------------------
# # Helpers
# # ------------------------------

# def _norm(s: str) -> str:
#     if pd.isna(s):
#         return ""
#     return str(s).strip().lower()

# MIS_LABELLED_SET = {
#     "mis-labelled", "mis-labeled", "misbranded", "mis-branded", "mis branded", "mis labelled", "mis labeled"
# }

# COMPLIANT_TOKEN = "compliant as per fssr"
# SUBSTANDARD_TOKEN = "sub-standard"
# UNSAFE_TOKEN = "unsafe"

# PARAM_END_TOKEN = "compliance"  # parameters are like "X, unit Compliance, Y, unit Compliance"


# def split_parameters(cell_value: str) -> list:
#     """Split a text cell containing one or more parameter names.
#     Heuristic: parameters end with the token 'Compliance'. We take the text
#     leading up to each 'Compliance' as one parameter and clean punctuation.
#     If the token is absent, we also try semicolon / newline / pipe splitting.
#     """
#     if pd.isna(cell_value) or str(cell_value).strip() == "":
#         return []

#     text = str(cell_value)

#     if PARAM_END_TOKEN.lower() in text.lower():
#         # Split by the end-token while keeping preceding text as the parameter name
#         pieces = re.split(r"(?i)\bcompliance\b", text)
#         params = []
#         for p in pieces:
#             p = p.strip().strip(",;")
#             if not p:
#                 continue
#             # Often looks like: "Enterobacteriaceae, CFU/g"  -> keep as-is
#             params.append(p)
#         return params

#     # Fallbacks
#     # First try newline/semicolon/pipe separated
#     parts = re.split(r"[\n\r;|]", text)
#     out = []
#     for p in parts:
#         p = p.strip().strip(",")
#         if p:
#             out.append(p)
#     if out:
#         return out

#     # Final fallback: split on ",  "  (two spaces) or just comma for safety
#     parts = [p.strip() for p in re.split(r",\s{2,}|,", text) if p.strip()]
#     return parts


# def format_node_label(title: str, count: int | None = None, pct: float | None = None) -> str:
#     line1 = title
#     line2 = None
#     if count is not None and pct is not None:
#         line2 = f"[{count} Samples; {pct:.1f}%]"
#     elif count is not None:
#         unit = "Sample" if count == 1 else "Samples"
#         line2 = f"[{count} {unit}]"
#     return f"{line1}\n{line2}" if line2 else line1


# def build_tree_dot(df: pd.DataFrame, commodity: str, variant_choice: str | None = None) -> tuple[str, dict]:
#     """Return (dot_source, stats) for the selected commodity and variant.
#     stats is a dict of computed counts useful for debug/UI.
#     """
#     cols = {
#         "commodity": "Commodity",
#         "variant2": "Variant 2",
#         "overall_compliance": "Overall Compliance",
#         "overall_quality": "Overall Quality Classification",
#         "overall_safety": "Overall Safety Classification",
#         "overall_labelling": "Overall Labelling Complaince",
#         "substandard_cases": "Sub-Standard Cases",
#         "unsafe_cases": "Unsafe Cases",
#     }

#     # Filter commodity
#     dfx = df[df[cols["commodity"]] == commodity].copy()
#     if dfx.empty:
#         raise ValueError(f"No rows for commodity: {commodity}")

#     # Variant selection
#     variant_counts = dfx[cols["variant2"]].fillna("(missing)").value_counts()
#     if variant_choice is None:
#         # Prefer 'Packed Samples' if it exists, else the mode
#         variant_choice = "Packed Samples" if "Packed Samples" in variant_counts.index else variant_counts.index[0]

#     dfx = dfx[dfx[cols["variant2"]].fillna("(missing)") == variant_choice]
#     n_total = len(dfx)
#     if n_total == 0:
#         raise ValueError("No rows after filtering Variant 2 == '" + str(variant_choice) + "'")

#     # Compliant vs Non-compliant
#     comp_series = dfx[cols["overall_compliance"]].astype(str).str.strip().str.lower()
#     n_compliant = (comp_series == COMPLIANT_TOKEN).sum()
#     n_noncompliant = n_total - n_compliant

#     # Quality, Safety, Labelling buckets (counts against the filtered subset)
#     qual_series = dfx[cols["overall_quality"]].astype(str).str.strip().str.lower()
#     saf_series = dfx[cols["overall_safety"]].astype(str).str.strip().str.lower()
#     lab_series = dfx[cols["overall_labelling"]].astype(str).str.strip().str.lower()

#     n_quality_sub = (qual_series == SUBSTANDARD_TOKEN).sum()
#     n_safety_unsafe = (saf_series == UNSAFE_TOKEN).sum()
#     n_lab_mis = lab_series.apply(lambda s: any(x in s for x in MIS_LABELLED_SET)).sum()

#     # Parameter breakdowns
#     # For Quality Sub-Standard
#     qual_params = []
#     if n_quality_sub:
#         for _, row in dfx[qual_series == SUBSTANDARD_TOKEN].iterrows():
#             qual_params.extend(split_parameters(row.get(cols["substandard_cases"])) )
#     # Count
#     from collections import Counter
#     qual_param_counts = Counter([p for p in map(lambda x: x.strip(), qual_params) if p])

#     # For Safety Unsafe
#     saf_params = []
#     if n_safety_unsafe:
#         for _, row in dfx[saf_series == UNSAFE_TOKEN].iterrows():
#             saf_params.extend(split_parameters(row.get(cols["unsafe_cases"])) )
#     saf_param_counts = Counter([p for p in map(lambda x: x.strip(), saf_params) if p])

#     # Build DOT
#     def pct(n: int) -> float:
#         return (n / n_total * 100.0) if n_total else 0.0

#     dot = [
#         "digraph G {",
#         "  rankdir=TB;",
#         "  graph [splines=ortho, nodesep=0.5, ranksep=0.6];",
#         "  node [shape=box, style=\"rounded,filled\", color=\"#d0d0d0\", fillcolor=white, fontname=Helvetica];",
#         "  edge [arrowhead=normal];",
#     ]

#     # Node ids
#     root_id = "root"
#     variant_id = "variant"
#     comp_id = "comp"
#     noncomp_id = "noncomp"
#     qual_id = "qual"
#     saf_id = "saf"
#     lab_id = "lab"

#     # Root & Variant nodes
#     root_label = format_node_label(f"{commodity}", n_total, 100.0)
#     variant_label = format_node_label(f"{variant_choice}", n_total, 100.0)

#     dot.append(f'  {root_id} [label="{root_label}"];')
#     dot.append(f'  {variant_id} [label="{variant_label}"];')

#     # Compliant / Non-compliant
#     comp_label = format_node_label("Compliant", n_compliant, pct(n_compliant))
#     noncomp_label = format_node_label("Non-Compliant", n_noncompliant, pct(n_noncompliant))

#     dot.append(f'  {comp_id} [label="{comp_label}", fillcolor="#d4edda"];')  # light green
#     dot.append(f'  {noncomp_id} [label="{noncomp_label}", fillcolor="#f8d7da"];')  # light red

#     # Buckets
#     dot.append(f'  {qual_id} [label="{format_node_label("Quality Parameters\\nNon-Compliance", n_quality_sub, pct(n_quality_sub))}"];')
#     dot.append(f'  {saf_id} [label="{format_node_label("Safety Parameters\\nNon-Compliance", n_safety_unsafe, pct(n_safety_unsafe))}"];')
#     dot.append(f'  {lab_id} [label="{format_node_label("Labelling Attributes\\nNon-Compliance", n_lab_mis, pct(n_lab_mis))}"];')

#     # Edges
#     dot += [
#         f"  {root_id} -> {variant_id};",
#         f"  {variant_id} -> {comp_id};",
#         f"  {variant_id} -> {noncomp_id};",
#         f"  {noncomp_id} -> {qual_id};",
#         f"  {noncomp_id} -> {saf_id};",
#         f"  {noncomp_id} -> {lab_id};",
#     ]

#     # Quality parameter leaves
#     if qual_param_counts:
#         dot.append("  { rank = same; ")
#         q_nodes = []
#         for i, (pname, cnt) in enumerate(sorted(qual_param_counts.items(), key=lambda x: (-x[1], x[0]))):
#             nid = f"q{i}"
#             q_nodes.append(nid)
#             label = format_node_label(pname, cnt)
#             dot.append(f'    {nid} [label="{label}"];')
#         dot.append("  }")
#         for i, (pname, cnt) in enumerate(sorted(qual_param_counts.items(), key=lambda x: (-x[1], x[0]))):
#             nid = f"q{i}"
#             dot.append(f"  {qual_id} -> {nid};")

#     # Safety parameter leaves
#     if saf_param_counts:
#         dot.append("  { rank = same; ")
#         s_nodes = []
#         for i, (pname, cnt) in enumerate(sorted(saf_param_counts.items(), key=lambda x: (-x[1], x[0]))):
#             nid = f"s{i}"
#             s_nodes.append(nid)
#             label = format_node_label(pname, cnt)
#             dot.append(f'    {nid} [label="{label}"];')
#         dot.append("  }")
#         for i, (pname, cnt) in enumerate(sorted(saf_param_counts.items(), key=lambda x: (-x[1], x[0]))):
#             nid = f"s{i}"
#             dot.append(f"  {saf_id} -> {nid};")

#     dot.append("}")
#     dot_src = "\n".join(dot)

#     stats = {
#         "total": n_total,
#         "variant": variant_choice,
#         "compliant": n_compliant,
#         "non_compliant": n_noncompliant,
#         "quality_substandard": n_quality_sub,
#         "safety_unsafe": n_safety_unsafe,
#         "labelling_mis": n_lab_mis,
#         "qual_param_counts": dict(qual_param_counts),
#         "saf_param_counts": dict(saf_param_counts),
#     }

#     return dot_src, stats


# # ------------------------------
# # Streamlit UI
# # ------------------------------

# st.set_page_config(page_title="Decision Tree Chart Generator", layout="wide")
# st.title("Decision Tree Chart Generator (Compliance • Quality • Safety • Labelling)")
# st.caption("Upload your Excel/CSV → choose commodity → auto-generate the tree like your turmeric example.")

# uploaded = st.file_uploader("Upload dataset (Excel .xlsx or CSV)", type=["xlsx", "xls", "csv"], accept_multiple_files=False)

# if uploaded is None:
#     st.info("Upload your dataset to begin. Expected columns include: Commodity, Variant 2, Overall Compliance, Overall Quality Classification, Sub-Standard Cases, Overall Safety Classification, Unsafe Cases, Overall Labelling Complaince.")
#     st.stop()

# # Read file
# if uploaded.name.lower().endswith((".xlsx", ".xls")):
#     df = pd.read_excel(uploaded, engine="openpyxl")
# else:
#     df = pd.read_csv(uploaded)

# # Basic validation
# required_cols = [
#     "Commodity", "Variant 2", "Overall Compliance", "Overall Safety Classification",
#     "Overall Quality Classification", "Sub-Standard Cases", "Unsafe Cases", "Overall Labelling Complaince"
# ]
# missing = [c for c in required_cols if c not in df.columns]
# if missing:
#     st.error("Missing required columns: " + ", ".join(missing))
#     st.stop()

# # Controls
# commodities = sorted(df["Commodity"].dropna().unique().tolist())
# commodity = st.selectbox("Commodity", commodities, index=commodities.index("Turmeric Powder") if "Turmeric Powder" in commodities else 0)

# variants_all = df.loc[df["Commodity"] == commodity, "Variant 2"].fillna("(missing)").unique().tolist()
# # Prefer 'Packed Samples' if present
# pref_index = variants_all.index("Packed Samples") if "Packed Samples" in variants_all else 0
# variant_choice = st.selectbox("Variant 2 (path node)", variants_all, index=pref_index)

# # Build graph
# try:
#     dot_src, stats = build_tree_dot(df, commodity, variant_choice)
# except Exception as e:
#     st.exception(e)
#     st.stop()

# st.subheader("Decision Tree")
# st.graphviz_chart(dot_src, use_container_width=True)

# # Download options
# st.download_button("Download .dot source", data=dot_src.encode("utf-8"), file_name=f"{commodity}_decision_tree.dot", mime="text/vnd.graphviz")

# if graphviz is not None:
#     try:
#         # g = graphviz.Source(dot_src)
#         # svg_bytes = g.pipe(format="svg")
        

#         g = Digraph(engine="dot")
#         g.source = dot_src
#         png_bytes = g.pipe(format="png")
#         st.download_button("Download PNG", data=png_bytes, file_name=f"{commodity}_decision_tree.png", mime="image/png")

#         # st.download_button("Download SVG", data=svg_bytes, file_name=f"{commodity}_decision_tree.svg", mime="image/svg+xml")
#         # # PNG is larger; still useful
#         # png_bytes = g.pipe(format="png")
#         # st.download_button("Download PNG", data=png_bytes, file_name=f"{commodity}_decision_tree.png", mime="image/png")
#     except Exception as e:
#         st.warning("SVG/PNG export requires Graphviz system binaries. If this is a hosted environment without them, the buttons may not work.")
