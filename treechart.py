# app.py
import re
import json
from collections import Counter, defaultdict

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

# ------------------------------
# Helpers
# ------------------------------
MIS_LABELLED_SET = {
    "mis-labelled", "mis-labeled", "misbranded", "mis-branded",
    "mis branded", "mis labelled", "mis labeled"
}
COMPLIANT_TOKEN = "compliant as per fssr"
SUBSTANDARD_TOKEN = "sub-standard"
UNSAFE_TOKEN = "unsafe"
PARAM_END_TOKEN = "compliance"


def split_parameters(cell_value: str) -> list:
    if cell_value is None:
        return []
    text = str(cell_value).strip()
    if not text:
        return []
    if PARAM_END_TOKEN.lower() in text.lower():
        pieces = re.split(r"(?i)\bcompliance\b", text)
        return [p.strip(" ,;") for p in pieces if p.strip(" ,;")]
    parts = re.split(r"[\n\r;|]", text)
    out = [p.strip(" ,;") for p in parts if p.strip()]
    if out:
        return out
    return [p.strip() for p in re.split(r",\s{2,}|,", text) if p.strip()]


def format_node_label(title: str, count: int | None = None, pct: float | None = None) -> str:
    if count is not None and pct is not None:
        return f"{title}\\n[{count} Samples; {pct:.1f}%]"
    if count is not None:
        unit = "Sample" if count == 1 else "Samples"
        return f"{title}\\n[{count} {unit}]"
    return title


def build_tree_dot(df: pd.DataFrame, commodity: str, variant_choice: str, settings: dict) -> tuple[str, dict, pd.DataFrame]:
    """Build DOT decision tree and also return flat dataset for CSV export."""
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
    }

    dfx = df[df[cols["commodity"]] == commodity].copy()
    dfx = dfx[dfx[cols["variant2"]].fillna("(missing)") == variant_choice]
    n_total = len(dfx)
    if n_total == 0:
        raise ValueError(f"No rows for {commodity} / {variant_choice}")

    comp_series = dfx[cols["overall_compliance"]].astype(str).str.strip().str.lower()
    n_compliant = (comp_series == COMPLIANT_TOKEN).sum()
    n_noncompliant = n_total - n_compliant

    qual_series = dfx[cols["overall_quality"]].astype(str).str.strip().str.lower()
    saf_series = dfx[cols["overall_safety"]].astype(str).str.strip().str.lower()
    lab_series = dfx[cols["overall_labelling"]].astype(str).str.strip().str.lower()

    n_quality_sub = (qual_series == SUBSTANDARD_TOKEN).sum()
    n_safety_unsafe = (saf_series == UNSAFE_TOKEN).sum()
    n_lab_mis = lab_series.apply(lambda s: any(x in s for x in MIS_LABELLED_SET)).sum()

    qual_grouped = defaultdict(list)
    if n_quality_sub:
        for _, row in dfx[qual_series == SUBSTANDARD_TOKEN].iterrows():
            ttype = str(row.get(cols["test_type"], "Other")).strip()
            for p in split_parameters(row.get(cols["substandard_cases"])):
                if p.strip():
                    qual_grouped[ttype].append(p.strip())

    saf_grouped = defaultdict(list)
    if n_safety_unsafe:
        for _, row in dfx[saf_series == UNSAFE_TOKEN].iterrows():
            ttype = str(row.get(cols["test_type"], "Other")).strip()
            for p in split_parameters(row.get(cols["unsafe_cases"])):
                if p.strip():
                    saf_grouped[ttype].append(p.strip())

    def pct(n: int) -> float:
        return (n / n_total * 100.0) if n_total else 0.0

    # -----------------------------
    # DOT
    # -----------------------------
    dot_lines = [
        "digraph G {",
        "  rankdir=TB;",
        f"  graph [splines=ortho, nodesep={settings['nodesep']}, ranksep={settings['ranksep']}];",
        ("  node [shape={shape}, style=\"rounded,filled\", color=\"#d0d0d0\", "
         "fillcolor=\"{fill}\", fontname=\"{font}\", fontsize={fs}];").format(
            shape=settings["node_shape"], fill=settings["default_color"], font=settings["fontname"], fs=settings["fontsize"]),
        f"  edge [fontname=\"{settings['fontname']}\", fontsize={max(8, settings['fontsize']-2)} , arrowhead=normal];",
    ]

    records = []  # collect CSV rows

    root_label = format_node_label(f"{commodity} - {variant_choice}", n_total, 100.0)
    comp_label = format_node_label("Compliant", n_compliant, pct(n_compliant))
    noncomp_label = format_node_label("Non-Compliant", n_noncompliant, pct(n_noncompliant))

    dot_lines.append(f'  root [label="{root_label}"];')
    dot_lines.append(f'  comp [label="{comp_label}", fillcolor="{settings["compliant_color"]}"];')
    dot_lines.append(f'  noncomp [label="{noncomp_label}", fillcolor="{settings["noncompliant_color"]}"];')
    dot_lines.append("  root -> comp; root -> noncomp;")

    dot_lines.append(f'  qual [label="{format_node_label("Quality Parameters", n_quality_sub, pct(n_quality_sub))}"];')
    dot_lines.append(f'  saf [label="{format_node_label("Safety Parameters", n_safety_unsafe, pct(n_safety_unsafe))}"];')
    dot_lines.append(f'  lab [label="{format_node_label("Labelling Parameters", n_lab_mis, pct(n_lab_mis))}"];')
    dot_lines.append("  noncomp -> qual; noncomp -> saf; noncomp -> lab;")

    # Quality
    for j, (ttype, plist) in enumerate(qual_grouped.items()):
        type_id = f"qtype{j}"
        dot_lines.append(f'  {type_id} [label="{format_node_label(ttype, len(plist))}"];')
        dot_lines.append(f"  qual -> {type_id};")
        for pname, cnt in Counter(plist).items():
            pid = f"{type_id}_{abs(hash(pname))%9999}"
            dot_lines.append(f'  {pid} [label="{format_node_label(pname, cnt)}"];')
            dot_lines.append(f"  {type_id} -> {pid};")
            records.append([commodity, variant_choice, "Quality", ttype, pname, cnt])

    # Safety
    for j, (ttype, plist) in enumerate(saf_grouped.items()):
        type_id = f"stype{j}"
        dot_lines.append(f'  {type_id} [label="{format_node_label(ttype, len(plist))}"];')
        dot_lines.append(f"  saf -> {type_id};")
        for pname, cnt in Counter(plist).items():
            pid = f"{type_id}_{abs(hash(pname))%9999}"
            dot_lines.append(f'  {pid} [label="{format_node_label(pname, cnt)}"];')
            dot_lines.append(f"  {type_id} -> {pid};")
            records.append([commodity, variant_choice, "Safety", ttype, pname, cnt])

    dot_lines.append("}")
    df_csv = pd.DataFrame(records, columns=["Commodity", "Variant", "Branch", "Test Type", "Parameter", "Count"])
    return "\n".join(dot_lines), {"total": n_total}, df_csv


def build_summary_dot(df: pd.DataFrame, commodity: str, settings: dict) -> str:
    cols = {"commodity": "Commodity", "variant2": "Variant 2"}
    dfx = df[df[cols["commodity"]] == commodity]
    total = len(dfx)
    variants = dfx[cols["variant2"]].fillna("(missing)").value_counts().to_dict()

    dot = [
        "digraph G {",
        f"  rankdir={settings['rankdir']};",
        f"  graph [splines=ortho, nodesep={settings['nodesep']}, ranksep={settings['ranksep']}];",
        f'  node [shape={settings["node_shape"]}, style="rounded,filled", fillcolor="{settings["default_color"]}", fontname="{settings["fontname"]}", fontsize={settings["fontsize"]}];'
    ]
    root = format_node_label(commodity, total, 100)
    dot.append(f'  root [label="{root}"];')
    for i, (v, cnt) in enumerate(variants.items()):
        dot.append(f'  v{i} [label="{format_node_label(v, cnt, cnt/total*100)}"];')
        dot.append(f"  root -> v{i};")
    dot.append("}")
    return "\n".join(dot)


def render_viz(dot_src: str, height: int, filename_prefix: str):
    viz_html = f"""
    <html><body>
      <div id="viz">Rendering...</div>
      <a id="download-svg" download="{filename_prefix}.svg">⬇️ SVG</a>
      <a id="download-png" download="{filename_prefix}.png">⬇️ PNG</a>
      <script src="https://unpkg.com/viz.js@2.1.2/viz.js"></script>
      <script src="https://unpkg.com/viz.js@2.1.2/full.render.js"></script>
      <script>
        const dot = {json.dumps(dot_src)};
        const viz = new Viz();
        viz.renderSVGElement(dot).then(el => {{
          document.getElementById("viz").innerHTML = "";
          document.getElementById("viz").appendChild(el);
          const svgString = new XMLSerializer().serializeToString(el);
          const svgBlob = new Blob([svgString], {{type: "image/svg+xml"}});
          const svgUrl = URL.createObjectURL(svgBlob);
          document.getElementById("download-svg").href = svgUrl;
          const img = new Image();
          img.onload = function() {{
            const c = document.createElement("canvas");
            c.width = img.width*2; c.height = img.height*2;
            const ctx = c.getContext("2d");
            ctx.fillStyle = "#fff"; ctx.fillRect(0,0,c.width,c.height);
            ctx.drawImage(img,0,0,c.width,c.height);
            document.getElementById("download-png").href = c.toDataURL("image/png");
          }};
          img.src = "data:image/svg+xml;base64,"+btoa(unescape(encodeURIComponent(svgString)));
        }});
      </script>
    </body></html>
    """
    components.html(viz_html, height=height, scrolling=True)


# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(layout="wide")
st.title("🌳 Decision Tree Chart Generator")

uploaded = st.file_uploader("Upload Excel (with Commodity, Variant 2, ...)", type=["xlsx"])
if not uploaded:
    st.stop()
df = pd.read_excel(uploaded)

# Sidebar
st.sidebar.header("⚙️ Chart Style")
rankdir_choice = st.sidebar.radio("Orientation", ["TB", "LR"])
settings = {
    "rankdir": rankdir_choice,
    "fontname": st.sidebar.selectbox("Font", ["Helvetica", "Arial", "Times New Roman"]),
    "fontsize": st.sidebar.slider("Font size", 8, 20, 12),
    "node_shape": st.sidebar.selectbox("Node shape", ["box", "ellipse", "circle"]),
    "nodesep": st.sidebar.slider("Node spacing", 0.1, 2.0, 0.5),
    "ranksep": st.sidebar.slider("Rank separation", 0.1, 2.0, 0.6),
    "default_color": st.sidebar.color_picker("Default node color", "#ffffff"),
    "compliant_color": st.sidebar.color_picker("Compliant color", "#d4edda"),
    "noncompliant_color": st.sidebar.color_picker("Non-compliant color", "#f8d7da"),
}
preview_height = st.sidebar.slider("Preview height", 300, 1200, 600)

commodities = df["Commodity"].dropna().unique().tolist()
commodity = st.selectbox("Commodity", commodities)

variants = df[df["Commodity"] == commodity]["Variant 2"].dropna().unique().tolist()

if len(variants) > 1:
    st.subheader(f"📊 Summary Chart for {commodity}")
    summary_dot = build_summary_dot(df, commodity, settings)
    render_viz(summary_dot, preview_height, f"{commodity}_summary")

for v in variants:
    st.subheader(f"📊 Detailed Chart: {commodity} - {v}")
    dot_src, stats, df_csv = build_tree_dot(df, commodity, v, settings)
    render_viz(dot_src, preview_height, f"{commodity}_{v}")

    st.download_button(
        label=f"⬇️ Download CSV for {commodity} - {v}",
        data=df_csv.to_csv(index=False).encode("utf-8"),
        file_name=f"{commodity}_{v}_tree.csv",
        mime="text/csv",
    )



















































































# # app.py
# import re
# import json
# from collections import Counter

# import pandas as pd
# import streamlit as st
# import streamlit.components.v1 as components

# # ------------------------------
# # Helpers
# # ------------------------------
# MIS_LABELLED_SET = {
#     "mis-labelled", "mis-labeled", "misbranded", "mis-branded",
#     "mis branded", "mis labelled", "mis labeled"
# }
# COMPLIANT_TOKEN = "compliant as per fssr"
# SUBSTANDARD_TOKEN = "sub-standard"
# UNSAFE_TOKEN = "unsafe"
# PARAM_END_TOKEN = "compliance"


# def split_parameters(cell_value: str) -> list:
#     """Split cell into parameter names."""
#     if cell_value is None:
#         return []
#     text = str(cell_value).strip()
#     if not text:
#         return []
#     if PARAM_END_TOKEN.lower() in text.lower():
#         pieces = re.split(r"(?i)\bcompliance\b", text)
#         return [p.strip(" ,;") for p in pieces if p.strip(" ,;")]
#     parts = re.split(r"[\n\r;|]", text)
#     out = [p.strip(" ,;") for p in parts if p.strip()]
#     if out:
#         return out
#     return [p.strip() for p in re.split(r",\s{2,}|,", text) if p.strip()]


# def format_node_label(title: str, count: int | None = None, pct: float | None = None) -> str:
#     if count is not None and pct is not None:
#         return f"{title}\\n[{count} Samples; {pct:.1f}%]"
#     if count is not None:
#         unit = "Sample" if count == 1 else "Samples"
#         return f"{title}\\n[{count} {unit}]"
#     return title
# def build_tree_dot(df: pd.DataFrame, commodity: str, variant_choice: str, settings: dict) -> tuple[str, dict]:
#     """Build DOT decision tree for a single commodity+variant with grouping by Test Type."""
#     cols = {
#         "commodity": "Commodity",
#         "variant2": "Variant 2",
#         "overall_compliance": "Overall Compliance",
#         "overall_quality": "Overall Quality Classification",
#         "overall_safety": "Overall Safety Classification",
#         "overall_labelling": "Overall Labelling Complaince",
#         "substandard_cases": "Sub-Standard Cases",
#         "unsafe_cases": "Unsafe Cases",
#         "test_type": "Test Type",   # 👈 NEW: mapping for grouping
#     }

#     dfx = df[df[cols["commodity"]] == commodity].copy()
#     dfx = dfx[dfx[cols["variant2"]].fillna("(missing)") == variant_choice]
#     n_total = len(dfx)
#     if n_total == 0:
#         raise ValueError(f"No rows for {commodity} / {variant_choice}")

#     # classifications
#     comp_series = dfx[cols["overall_compliance"]].astype(str).str.strip().str.lower()
#     n_compliant = (comp_series == COMPLIANT_TOKEN).sum()
#     n_noncompliant = n_total - n_compliant

#     qual_series = dfx[cols["overall_quality"]].astype(str).str.strip().str.lower()
#     saf_series = dfx[cols["overall_safety"]].astype(str).str.strip().str.lower()
#     lab_series = dfx[cols["overall_labelling"]].astype(str).str.strip().str.lower()

#     n_quality_sub = (qual_series == SUBSTANDARD_TOKEN).sum()
#     n_safety_unsafe = (saf_series == UNSAFE_TOKEN).sum()
#     n_lab_mis = lab_series.apply(lambda s: any(x in s for x in MIS_LABELLED_SET)).sum()

#     # -----------------------------
#     # Parameters grouped by Test Type
#     # -----------------------------
#     from collections import defaultdict

#     qual_grouped = defaultdict(list)
#     if n_quality_sub:
#         for _, row in dfx[qual_series == SUBSTANDARD_TOKEN].iterrows():
#             ttype = str(row.get(cols["test_type"], "Other")).strip()
#             for p in split_parameters(row.get(cols["substandard_cases"])):
#                 if p.strip():
#                     qual_grouped[ttype].append(p.strip())

#     saf_grouped = defaultdict(list)
#     if n_safety_unsafe:
#         for _, row in dfx[saf_series == UNSAFE_TOKEN].iterrows():
#             ttype = str(row.get(cols["test_type"], "Other")).strip()
#             for p in split_parameters(row.get(cols["unsafe_cases"])):
#                 if p.strip():
#                     saf_grouped[ttype].append(p.strip())

#     # helper
#     def pct(n: int) -> float:
#         return (n / n_total * 100.0) if n_total else 0.0

#     # -----------------------------
#     # Styles
#     # -----------------------------
#     rankdir = settings["rankdir"]
#     fontname = settings["fontname"]
#     fontsize = settings["fontsize"]
#     node_shape = settings["node_shape"]
#     nodesep = settings["nodesep"]
#     ranksep = settings["ranksep"]
#     default_color = settings["default_color"]
#     compliant_color = settings["compliant_color"]
#     noncompliant_color = settings["noncompliant_color"]

#     # dot_lines = [
#     #     "digraph G {",
#     #     f"  rankdir={rankdir};",
#     #     f"  graph [splines=ortho, nodesep={nodesep}, ranksep={ranksep}];",
#     #     ("  node [shape={shape}, style=\"rounded,filled\", color=\"#d0d0d0\", "
#     #      "fillcolor=\"{fill}\", fontname=\"{font}\", fontsize={fs}];").format(
#     #         shape=node_shape, fill=default_color, font=fontname, fs=fontsize),
#     #     f"  edge [fontname=\"{fontname}\", fontsize={max(8, fontsize-2)} , arrowhead=normal];",
#     # ]

#     dot_lines = [
        
#         "digraph G {",
#         "  rankdir=TB;",   # 👈 force top-to-bottom flow
#         f"  graph [splines=ortho, nodesep={nodesep}, ranksep={ranksep}];",
#         ("  node [shape={shape}, style=\"rounded,filled\", color=\"#d0d0d0\", "
#          "fillcolor=\"{fill}\", fontname=\"{font}\", fontsize={fs}];").format(
#             shape=node_shape, fill=default_color, font=fontname, fs=fontsize),
#         f"  edge [fontname=\"{fontname}\", fontsize={max(8, fontsize-2)} , arrowhead=normal];",
#     ]


#     # -----------------------------
#     # Main Nodes
#     # -----------------------------
#     root_label = format_node_label(f"{commodity} - {variant_choice}", n_total, 100.0)
#     comp_label = format_node_label("Compliant", n_compliant, pct(n_compliant))
#     noncomp_label = format_node_label("Non-Compliant", n_noncompliant, pct(n_noncompliant))

#     dot_lines.append(f'  root [label="{root_label}"];')
#     dot_lines.append(f'  comp [label="{comp_label}", fillcolor="{compliant_color}"];')
#     dot_lines.append(f'  noncomp [label="{noncomp_label}", fillcolor="{noncompliant_color}"];')
#     dot_lines.append("  root -> comp; root -> noncomp;")

#     # Quality / Safety / Lab
#     dot_lines.append(f'  qual [label="{format_node_label("Quality Parameters", n_quality_sub, pct(n_quality_sub))}"];')
#     dot_lines.append(f'  saf [label="{format_node_label("Safety Parameters", n_safety_unsafe, pct(n_safety_unsafe))}"];')
#     dot_lines.append(f'  lab [label="{format_node_label("Labelling Parameters", n_lab_mis, pct(n_lab_mis))}"];')
#     dot_lines.append("  noncomp -> qual; noncomp -> saf; noncomp -> lab;")

#     # -----------------------------
#     # Quality Branch with grouping
#     # -----------------------------
#     for j, (ttype, plist) in enumerate(qual_grouped.items()):
#         type_id = f"qtype{j}"
#         dot_lines.append(f'  {type_id} [label="{format_node_label(ttype, len(plist))}"];')
#         dot_lines.append(f"  qual -> {type_id};")
#         for i, pname in enumerate(sorted(set(plist))):  # unique params
#             pid = f"{type_id}_{i}"
#             dot_lines.append(f'  {pid} [label="{format_node_label(pname, plist.count(pname))}"];')
#             dot_lines.append(f"  {type_id} -> {pid};")

#     # -----------------------------
#     # Safety Branch with grouping
#     # -----------------------------
#     for j, (ttype, plist) in enumerate(saf_grouped.items()):
#         type_id = f"stype{j}"
#         dot_lines.append(f'  {type_id} [label="{format_node_label(ttype, len(plist))}"];')
#         dot_lines.append(f"  saf -> {type_id};")
#         for i, pname in enumerate(sorted(set(plist))):
#             pid = f"{type_id}_{i}"
#             dot_lines.append(f'  {pid} [label="{format_node_label(pname, plist.count(pname))}"];')
#             dot_lines.append(f"  {type_id} -> {pid};")

#     dot_lines.append("}")
#     return "\n".join(dot_lines), {"total": n_total}



# # def build_tree_dot(df: pd.DataFrame, commodity: str, variant_choice: str, settings: dict) -> tuple[str, dict]:
# #     """Build DOT decision tree for a single commodity+variant."""
# #     cols = {
# #         "commodity": "Commodity",
# #         "variant2": "Variant 2",
# #         "overall_compliance": "Overall Compliance",
# #         "overall_quality": "Overall Quality Classification",
# #         "overall_safety": "Overall Safety Classification",
# #         "overall_labelling": "Overall Labelling Complaince",
# #         "substandard_cases": "Sub-Standard Cases",
# #         "unsafe_cases": "Unsafe Cases",
# #     }

# #     dfx = df[df[cols["commodity"]] == commodity].copy()
# #     dfx = dfx[dfx[cols["variant2"]].fillna("(missing)") == variant_choice]
# #     n_total = len(dfx)
# #     if n_total == 0:
# #         raise ValueError(f"No rows for {commodity} / {variant_choice}")

# #     # classifications
# #     comp_series = dfx[cols["overall_compliance"]].astype(str).str.strip().str.lower()
# #     n_compliant = (comp_series == COMPLIANT_TOKEN).sum()
# #     n_noncompliant = n_total - n_compliant

# #     qual_series = dfx[cols["overall_quality"]].astype(str).str.strip().str.lower()
# #     saf_series = dfx[cols["overall_safety"]].astype(str).str.strip().str.lower()
# #     lab_series = dfx[cols["overall_labelling"]].astype(str).str.strip().str.lower()

# #     n_quality_sub = (qual_series == SUBSTANDARD_TOKEN).sum()
# #     n_safety_unsafe = (saf_series == UNSAFE_TOKEN).sum()
# #     n_lab_mis = lab_series.apply(lambda s: any(x in s for x in MIS_LABELLED_SET)).sum()

# #     # parameters
# #     qual_params = []
# #     if n_quality_sub:
# #         for _, row in dfx[qual_series == SUBSTANDARD_TOKEN].iterrows():
# #             qual_params.extend(split_parameters(row.get(cols["substandard_cases"])))
# #     saf_params = []
# #     if n_safety_unsafe:
# #         for _, row in dfx[saf_series == UNSAFE_TOKEN].iterrows():
# #             saf_params.extend(split_parameters(row.get(cols["unsafe_cases"])))
# #     qual_param_counts = Counter([p.strip() for p in qual_params if p.strip()])
# #     saf_param_counts = Counter([p.strip() for p in saf_params if p.strip()])

# #     def pct(n: int) -> float:
# #         return (n / n_total * 100.0) if n_total else 0.0

# #     # styles
# #     rankdir = settings["rankdir"]
# #     fontname = settings["fontname"]
# #     fontsize = settings["fontsize"]
# #     node_shape = settings["node_shape"]
# #     nodesep = settings["nodesep"]
# #     ranksep = settings["ranksep"]
# #     default_color = settings["default_color"]
# #     compliant_color = settings["compliant_color"]
# #     noncompliant_color = settings["noncompliant_color"]

# #     dot_lines = [
# #         "digraph G {",
# #         f"  rankdir={rankdir};",
# #         f"  graph [splines=ortho, nodesep={nodesep}, ranksep={ranksep}];",
# #         ("  node [shape={shape}, style=\"rounded,filled\", color=\"#d0d0d0\", "
# #          "fillcolor=\"{fill}\", fontname=\"{font}\", fontsize={fs}];").format(
# #             shape=node_shape, fill=default_color, font=fontname, fs=fontsize),
# #         f"  edge [fontname=\"{fontname}\", fontsize={max(8, fontsize-2)} , arrowhead=normal];",
# #     ]

# #     # main nodes
# #     root_label = format_node_label(f"{commodity} - {variant_choice}", n_total, 100.0)
# #     comp_label = format_node_label("Compliant", n_compliant, pct(n_compliant))
# #     noncomp_label = format_node_label("Non-Compliant", n_noncompliant, pct(n_noncompliant))
# #     dot_lines.append(f'  root [label="{root_label}"];')
# #     dot_lines.append(f'  comp [label="{comp_label}", fillcolor="{compliant_color}"];')
# #     dot_lines.append(f'  noncomp [label="{noncomp_label}", fillcolor="{noncompliant_color}"];')
# #     dot_lines.append("  root -> comp; root -> noncomp;")

# #     # quality/safety/lab
# #     dot_lines.append(f'  qual [label="{format_node_label("Quality Parameters", n_quality_sub, pct(n_quality_sub))}"];')
# #     dot_lines.append(f'  saf [label="{format_node_label("Safety Parameters", n_safety_unsafe, pct(n_safety_unsafe))}"];')
# #     dot_lines.append(f'  lab [label="{format_node_label("Labelling Parameters", n_lab_mis, pct(n_lab_mis))}"];')
# #     dot_lines.append("  noncomp -> qual; noncomp -> saf; noncomp -> lab;")

# #     # leaf params
# #     for i, (pname, cnt) in enumerate(sorted(qual_param_counts.items(), key=lambda x: -x[1])):
# #         dot_lines.append(f'  q{i} [label="{format_node_label(pname, cnt)}"];')
# #         dot_lines.append(f"  qual -> q{i};")
# #     for i, (pname, cnt) in enumerate(sorted(saf_param_counts.items(), key=lambda x: -x[1])):
# #         dot_lines.append(f'  s{i} [label="{format_node_label(pname, cnt)}"];')
# #         dot_lines.append(f"  saf -> s{i};")

# #     dot_lines.append("}")
# #     return "\n".join(dot_lines), {"total": n_total}


# def build_summary_dot(df: pd.DataFrame, commodity: str, settings: dict) -> str:
#     """Summary chart for commodity showing Loose vs Packed split."""
#     cols = {"commodity": "Commodity", "variant2": "Variant 2"}
#     dfx = df[df[cols["commodity"]] == commodity]
#     total = len(dfx)
#     variants = dfx[cols["variant2"]].fillna("(missing)").value_counts().to_dict()

#     rankdir = settings["rankdir"]
#     fontname = settings["fontname"]
#     fontsize = settings["fontsize"]
#     node_shape = settings["node_shape"]
#     nodesep = settings["nodesep"]
#     ranksep = settings["ranksep"]
#     default_color = settings["default_color"]

#     dot = [
#         "digraph G {",
#         f"  rankdir={rankdir};",
#         f"  graph [splines=ortho, nodesep={nodesep}, ranksep={ranksep}];",
#         f'  node [shape={node_shape}, style="rounded,filled", fillcolor="{default_color}", fontname="{fontname}", fontsize={fontsize}];'
#     ]
#     root = format_node_label(commodity, total, 100)
#     dot.append(f'  root [label="{root}"];')
#     for i, (v, cnt) in enumerate(variants.items()):
#         dot.append(f'  v{i} [label="{format_node_label(v, cnt, cnt/total*100)}"];')
#         dot.append(f"  root -> v{i};")
#     dot.append("}")
#     return "\n".join(dot)


# def render_viz(dot_src: str, height: int, filename_prefix: str):
#     """Render Graphviz via Viz.js in iframe with SVG/PNG download."""
#     viz_html = f"""
#     <html><body>
#       <div id="viz">Rendering...</div>
#       <a id="download-svg" download="{filename_prefix}.svg">⬇️ SVG</a>
#       <a id="download-png" download="{filename_prefix}.png">⬇️ PNG</a>
#       <script src="https://unpkg.com/viz.js@2.1.2/viz.js"></script>
#       <script src="https://unpkg.com/viz.js@2.1.2/full.render.js"></script>
#       <script>
#         const dot = {json.dumps(dot_src)};
#         const viz = new Viz();
#         viz.renderSVGElement(dot).then(el => {{
#           document.getElementById("viz").innerHTML = "";
#           document.getElementById("viz").appendChild(el);
#           const svgString = new XMLSerializer().serializeToString(el);
#           const svgBlob = new Blob([svgString], {{type: "image/svg+xml"}});
#           const svgUrl = URL.createObjectURL(svgBlob);
#           document.getElementById("download-svg").href = svgUrl;
#           const img = new Image();
#           img.onload = function() {{
#             const c = document.createElement("canvas");
#             c.width = img.width*2; c.height = img.height*2;
#             const ctx = c.getContext("2d");
#             ctx.fillStyle = "#fff"; ctx.fillRect(0,0,c.width,c.height);
#             ctx.drawImage(img,0,0,c.width,c.height);
#             document.getElementById("download-png").href = c.toDataURL("image/png");
#           }};
#           img.src = "data:image/svg+xml;base64,"+btoa(unescape(encodeURIComponent(svgString)));
#         }});
#       </script>
#     </body></html>
#     """
#     components.html(viz_html, height=height, scrolling=True)


# # ------------------------------
# # Streamlit UI
# # ------------------------------
# st.set_page_config(layout="wide")
# st.title("🌳 Decision Tree Chart Generator")

# uploaded = st.file_uploader("Upload Excel (with Commodity, Variant 2, ...)", type=["xlsx"])
# if not uploaded:
#     st.stop()
# df = pd.read_excel(uploaded)

# # Sidebar settings
# st.sidebar.header("⚙️ Chart Style")
# rankdir_choice = st.sidebar.radio("Orientation", ["TB", "LR"])
# settings = {
#     "rankdir": rankdir_choice,
#     "fontname": st.sidebar.selectbox("Font", ["Helvetica", "Arial", "Times New Roman"]),
#     "fontsize": st.sidebar.slider("Font size", 8, 20, 12),
#     "node_shape": st.sidebar.selectbox("Node shape", ["box", "ellipse", "circle"]),
#     "nodesep": st.sidebar.slider("Node spacing", 0.1, 2.0, 0.5),
#     "ranksep": st.sidebar.slider("Rank separation", 0.1, 2.0, 0.6),
#     "default_color": st.sidebar.color_picker("Default node color", "#ffffff"),
#     "compliant_color": st.sidebar.color_picker("Compliant color", "#d4edda"),
#     "noncompliant_color": st.sidebar.color_picker("Non-compliant color", "#f8d7da"),
# }
# preview_height = st.sidebar.slider("Preview height", 300, 1200, 600)

# # Select commodity
# commodities = df["Commodity"].dropna().unique().tolist()
# commodity = st.selectbox("Commodity", commodities)

# # Check variants
# variants = df[df["Commodity"] == commodity]["Variant 2"].dropna().unique().tolist()

# if len(variants) > 1:
#     st.subheader(f"📊 Summary Chart for {commodity}")
#     summary_dot = build_summary_dot(df, commodity, settings)
#     render_viz(summary_dot, preview_height, f"{commodity}_summary")

# for v in variants:
#     st.subheader(f"📊 Detailed Chart: {commodity} - {v}")
#     dot_src, stats = build_tree_dot(df, commodity, v, settings)
#     render_viz(dot_src, preview_height, f"{commodity}_{v}")



























































































































# # app.py
# import re
# import json
# from collections import Counter

# import pandas as pd
# import streamlit as st
# import streamlit.components.v1 as components

# # ------------------------------
# # Helpers (parsing & tree builder)
# # ------------------------------

# MIS_LABELLED_SET = {
#     "mis-labelled", "mis-labeled", "misbranded", "mis-branded", "mis branded", "mis labelled", "mis labeled"
# }

# COMPLIANT_TOKEN = "compliant as per fssr"
# SUBSTANDARD_TOKEN = "sub-standard"
# UNSAFE_TOKEN = "unsafe"
# PARAM_END_TOKEN = "compliance"


# def split_parameters(cell_value: str) -> list:
#     """Split a text cell containing one or more parameter names."""
#     if cell_value is None:
#         return []
#     text = str(cell_value).strip()
#     if not text:
#         return []

#     if PARAM_END_TOKEN.lower() in text.lower():
#         pieces = re.split(r"(?i)\bcompliance\b", text)
#         return [p.strip(" ,;") for p in pieces if p.strip(" ,;")]

#     parts = re.split(r"[\n\r;|]", text)
#     out = [p.strip(" ,;") for p in parts if p.strip()]
#     if out:
#         return out

#     return [p.strip() for p in re.split(r",\s{2,}|,", text) if p.strip()]


# def format_node_label(title: str, count: int | None = None, pct: float | None = None) -> str:
#     if count is not None and pct is not None:
#         return f"{title}\\n[{count} Samples; {pct:.1f}%]"
#     if count is not None:
#         unit = "Sample" if count == 1 else "Samples"
#         return f"{title}\\n[{count} {unit}]"
#     return title


# def build_tree_dot(df: pd.DataFrame, commodity: str, variant_choice: str, settings: dict) -> tuple[str, dict]:
#     """Build DOT source string and return (dot_src, stats)."""
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

#     # choose variant
#     variant_counts = dfx[cols["variant2"]].fillna("(missing)").value_counts()
#     if variant_choice is None:
#         variant_choice = "Packed Samples" if "Packed Samples" in variant_counts.index else variant_counts.index[0]

#     dfx = dfx[dfx[cols["variant2"]].fillna("(missing)") == variant_choice]
#     n_total = len(dfx)
#     if n_total == 0:
#         raise ValueError("No rows after filtering Variant 2 == '" + str(variant_choice) + "'")

#     # classification counts
#     comp_series = dfx[cols["overall_compliance"]].astype(str).str.strip().str.lower()
#     n_compliant = (comp_series == COMPLIANT_TOKEN).sum()
#     n_noncompliant = n_total - n_compliant

#     qual_series = dfx[cols["overall_quality"]].astype(str).str.strip().str.lower()
#     saf_series = dfx[cols["overall_safety"]].astype(str).str.strip().str.lower()
#     lab_series = dfx[cols["overall_labelling"]].astype(str).str.strip().str.lower()

#     n_quality_sub = (qual_series == SUBSTANDARD_TOKEN).sum()
#     n_safety_unsafe = (saf_series == UNSAFE_TOKEN).sum()
#     n_lab_mis = lab_series.apply(lambda s: any(x in s for x in MIS_LABELLED_SET)).sum()

#     # collect parameters
#     qual_params = []
#     if n_quality_sub:
#         for _, row in dfx[qual_series == SUBSTANDARD_TOKEN].iterrows():
#             qual_params.extend(split_parameters(row.get(cols["substandard_cases"])))
#     saf_params = []
#     if n_safety_unsafe:
#         for _, row in dfx[saf_series == UNSAFE_TOKEN].iterrows():
#             saf_params.extend(split_parameters(row.get(cols["unsafe_cases"])))

#     qual_param_counts = Counter([p.strip() for p in qual_params if p.strip()])
#     saf_param_counts = Counter([p.strip() for p in saf_params if p.strip()])

#     def pct(n: int) -> float:
#         return (n / n_total * 100.0) if n_total else 0.0

#     # Style settings (passed from sidebar)
#     rankdir = settings.get("rankdir", "TB")
#     fontname = settings.get("fontname", "Helvetica")
#     fontsize = settings.get("fontsize", 12)
#     node_shape = settings.get("node_shape", "box")
#     nodesep = settings.get("nodesep", 0.5)
#     ranksep = settings.get("ranksep", 0.6)
#     default_color = settings.get("default_color", "#ffffff")
#     compliant_color = settings.get("compliant_color", "#d4edda")
#     noncompliant_color = settings.get("noncompliant_color", "#f8d7da")

#     # Build DOT (quotes around color values)
#     dot_lines = [
#         "digraph G {",
#         f"  rankdir={rankdir};",
#         f"  graph [splines=ortho, nodesep={nodesep}, ranksep={ranksep}];",
#         ("  node [shape={shape}, style=\"rounded,filled\", color=\"#d0d0d0\", "
#          "fillcolor=\"{fill}\", fontname=\"{font}\", fontsize={fs}];").format(
#             shape=node_shape, fill=default_color, font=fontname, fs=fontsize
#         ),
#         f"  edge [fontname=\"{fontname}\", fontsize={max(8, fontsize-2)} , arrowhead=normal];",
#     ]

#     root_id = "root"
#     variant_id = "variant"
#     comp_id = "comp"
#     noncomp_id = "noncomp"
#     qual_id = "qual"
#     saf_id = "saf"
#     lab_id = "lab"

#     root_label = format_node_label(f"{commodity}", n_total, 100.0)
#     variant_label = format_node_label(f"{variant_choice}", n_total, 100.0)
#     comp_label = format_node_label("Compliant", n_compliant, pct(n_compliant))
#     noncomp_label = format_node_label("Non-Compliant", n_noncompliant, pct(n_noncompliant))

#     dot_lines.append(f'  {root_id} [label="{root_label}"];')
#     dot_lines.append(f'  {variant_id} [label="{variant_label}"];')
#     dot_lines.append(f'  {comp_id} [label="{comp_label}", fillcolor="{compliant_color}"];')
#     dot_lines.append(f'  {noncomp_id} [label="{noncomp_label}", fillcolor="{noncompliant_color}"];')
#     dot_lines.append(
#         f'  {qual_id} [label="{format_node_label("Quality Parameters\\nNon-Compliance", n_quality_sub, pct(n_quality_sub))}"];'
#     )
#     dot_lines.append(
#         f'  {saf_id} [label="{format_node_label("Safety Parameters\\nNon-Compliance", n_safety_unsafe, pct(n_safety_unsafe))}"];'
#     )
#     dot_lines.append(
#         f'  {lab_id} [label="{format_node_label("Labelling Attributes\\nNon-Compliance", n_lab_mis, pct(n_lab_mis))}"];'
#     )

#     dot_lines += [
#         f"  {root_id} -> {variant_id};",
#         f"  {variant_id} -> {comp_id};",
#         f"  {variant_id} -> {noncomp_id};",
#         f"  {noncomp_id} -> {qual_id};",
#         f"  {noncomp_id} -> {saf_id};",
#         f"  {noncomp_id} -> {lab_id};",
#     ]

#     # Add quality parameter leaves
#     for i, (pname, cnt) in enumerate(sorted(qual_param_counts.items(), key=lambda x: (-x[1], x[0]))):
#         nid = f"q{i}"
#         dot_lines.append(f'  {nid} [label="{format_node_label(pname, cnt)}"];')
#         dot_lines.append(f"  {qual_id} -> {nid};")

#     # Add safety parameter leaves
#     for i, (pname, cnt) in enumerate(sorted(saf_param_counts.items(), key=lambda x: (-x[1], x[0]))):
#         nid = f"s{i}"
#         dot_lines.append(f'  {nid} [label="{format_node_label(pname, cnt)}"];')
#         dot_lines.append(f"  {saf_id} -> {nid};")

#     dot_lines.append("}")
#     dot_src = "\n".join(dot_lines)

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
# st.title("🌳 Decision Tree Chart Generator")
# st.markdown("Upload Excel/CSV → choose commodity → customize tree in the sidebar → download SVG/PNG")

# # Upload
# uploaded = st.file_uploader("Upload dataset (Excel .xlsx or CSV)", type=["xlsx", "xls", "csv"])
# if uploaded is None:
#     st.info("Upload your dataset to start. Expected columns: Commodity, Variant 2, Overall Compliance, Overall Quality Classification, Sub-Standard Cases, Overall Safety Classification, Unsafe Cases, Overall Labelling Complaince")
#     st.stop()

# # Read file
# if uploaded.name.lower().endswith((".xlsx", ".xls")):
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

# # Commodity & variant selection
# commodities = sorted(df["Commodity"].dropna().unique().tolist())
# commodity = st.selectbox("Commodity", commodities, index=0)
# variants_all = df.loc[df["Commodity"] == commodity, "Variant 2"].fillna("(missing)").unique().tolist()
# variant_choice = st.selectbox("Variant 2 (path node)", variants_all, index=0)

# # Sidebar customization controls
# st.sidebar.header("⚙️ Chart Appearance")
# rankdir_choice = st.sidebar.radio("Orientation", ("Top → Bottom (TB)", "Left → Right (LR)"))
# rankdir = "TB" if rankdir_choice.startswith("Top") else "LR"

# fontname = st.sidebar.selectbox("Font family", ["Helvetica", "Arial", "Times New Roman", "Courier"])
# fontsize = st.sidebar.slider("Node font size", 8, 20, 12)
# node_shape = st.sidebar.selectbox("Node shape", ["box", "ellipse", "oval", "circle"])
# nodesep = st.sidebar.slider("Node spacing (nodesep)", 0.1, 2.0, 0.5)
# ranksep = st.sidebar.slider("Rank separation (ranksep)", 0.1, 2.0, 0.6)
# default_color = st.sidebar.color_picker("Default node fill", "#ffffff")
# compliant_color = st.sidebar.color_picker("Compliant node color", "#d4edda")
# noncompliant_color = st.sidebar.color_picker("Non-compliant node color", "#f8d7da")
# preview_height = st.sidebar.slider("Preview iframe height (px)", 300, 1400, 700)

# settings = {
#     "rankdir": rankdir,
#     "fontname": fontname,
#     "fontsize": fontsize,
#     "node_shape": node_shape,
#     "nodesep": nodesep,
#     "ranksep": ranksep,
#     "default_color": default_color,
#     "compliant_color": compliant_color,
#     "noncompliant_color": noncompliant_color,
# }

# # Generate DOT and stats
# try:
#     dot_src, stats = build_tree_dot(df, commodity, variant_choice, settings)
# except Exception as e:
#     st.exception(e)
#     st.stop()

# # Show stats small
# st.markdown(f"**Total samples ({commodity} / {variant_choice}):** {stats['total']}  •  "
#             f"Compliant: {stats['compliant']}  •  Non-compliant: {stats['non_compliant']}")

# # Provide dot download
# st.download_button("⬇️ Download .dot file", data=dot_src.encode("utf-8"),
#                    file_name=f"{commodity}_decision_tree.dot", mime="text/vnd.graphviz")

# # ------------------------------
# # Client-side preview + download using Viz.js inside an iframe
# # ------------------------------
# # We embed DOT into HTML and let Viz.js render it in the browser. This allows SVG/PNG downloads
# # without server-side Graphviz binaries.

# viz_html = f"""
# <!doctype html>
# <html>
# <head>
#   <meta charset="utf-8"/>
#   <title>Decision Tree Preview</title>
#   <style>
#     body {{ font-family: Arial, Helvetica, sans-serif; margin: 8px; }}
#     .toolbar {{ margin-bottom: 8px; }}
#     .btn {{
#       display: inline-block;
#       padding: 6px 10px;
#       margin-right: 8px;
#       background: #1976d2;
#       color: white;
#       text-decoration: none;
#       border-radius: 6px;
#       font-size: 13px;
#     }}
#     .btn:disabled {{ background: #cccccc; }}
#     #viz {{ border: 1px solid #eee; padding: 8px; overflow: auto; background:white; }}
#   </style>
# </head>
# <body>
#   <div class="toolbar">
#     <a id="download-svg" class="btn" href="#" download="chart.svg">⬇️ Download SVG</a>
#     <a id="download-png" class="btn" href="#" download="chart.png">⬇️ Download PNG</a>
#     <a id="open-svg" class="btn" href="#" target="_blank">Open SVG in new tab</a>
#   </div>
#   <div id="viz">Rendering...</div>

#   <!-- Viz.js (wasm/full render) -->
#   <script src="https://unpkg.com/viz.js@2.1.2/viz.js"></script>
#   <script src="https://unpkg.com/viz.js@2.1.2/full.render.js"></script>

#   <script>
#     const dot = {json.dumps(dot_src)};
#     const viz = new Viz();

#     function safeDownloadDataUrl(elem, dataUrl, filename) {{
#       elem.href = dataUrl;
#       elem.download = filename;
#       elem.classList.remove('disabled');
#     }}

#     viz.renderSVGElement(dot)
#       .then(function(element) {{
#           const container = document.getElementById('viz');
#           container.innerHTML = '';
#           // append SVG element
#           container.appendChild(element);

#           // Serialize SVG to string
#           const svgNode = element;
#           const serializer = new XMLSerializer();
#           const svgString = serializer.serializeToString(svgNode);

#           // SVG download (blob url)
#           const svgBlob = new Blob([svgString], {{type: 'image/svg+xml;charset=utf-8'}});
#           const svgUrl = URL.createObjectURL(svgBlob);
#           const svgA = document.getElementById('download-svg');
#           svgA.href = svgUrl;
#           svgA.download = 'decision_tree.svg';
#           document.getElementById('open-svg').href = svgUrl;

#           // Make a PNG via canvas
#           const svg64 = btoa(unescape(encodeURIComponent(svgString)));
#           const image64 = 'data:image/svg+xml;base64,' + svg64;
#           const img = new Image();
#           img.onload = function() {{
#             // scale up a bit for higher resolution
#             const scale = 2;
#             const canvas = document.createElement('canvas');
#             canvas.width = img.width * scale;
#             canvas.height = img.height * scale;
#             const ctx = canvas.getContext('2d');
#             // white background for PNG
#             ctx.fillStyle = '#ffffff';
#             ctx.fillRect(0,0,canvas.width,canvas.height);
#             ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
#             const pngUrl = canvas.toDataURL('image/png');
#             const pngA = document.getElementById('download-png');
#             pngA.href = pngUrl;
#             pngA.download = 'decision_tree.png';
#           }};
#           img.onerror = function(err) {{
#             console.error('Image load error', err);
#             document.getElementById('download-png').classList.add('disabled');
#             document.getElementById('download-png').text = 'PNG not available';
#           }};
#           img.src = image64;
#       }})
#       .catch(function(error) {{
#           console.error(error);
#           document.getElementById('viz').innerText = 'Error rendering chart: ' + error;
#       }});
#   </script>
# </body>
# </html>
# """

# components.html(viz_html, height=preview_height, scrolling=True)

# st.markdown(
#     "If the chart looks cramped, try increasing **Preview iframe height** or change **Orientation** to LR (Left→Right) in the sidebar."
# )
