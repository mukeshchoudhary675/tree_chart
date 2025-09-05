import io
import re
import pandas as pd
import streamlit as st

try:
    import graphviz
except Exception as e:
    graphviz = None

# ------------------------------
# Helpers
# ------------------------------

def _norm(s: str) -> str:
    if pd.isna(s):
        return ""
    return str(s).strip().lower()

MIS_LABELLED_SET = {
    "mis-labelled", "mis-labeled", "misbranded", "mis-branded", "mis branded", "mis labelled", "mis labeled"
}

COMPLIANT_TOKEN = "compliant as per fssr"
SUBSTANDARD_TOKEN = "sub-standard"
UNSAFE_TOKEN = "unsafe"

PARAM_END_TOKEN = "compliance"  # parameters are like "X, unit Compliance, Y, unit Compliance"


def split_parameters(cell_value: str) -> list:
    """Split a text cell containing one or more parameter names.
    Heuristic: parameters end with the token 'Compliance'. We take the text
    leading up to each 'Compliance' as one parameter and clean punctuation.
    If the token is absent, we also try semicolon / newline / pipe splitting.
    """
    if pd.isna(cell_value) or str(cell_value).strip() == "":
        return []

    text = str(cell_value)

    if PARAM_END_TOKEN.lower() in text.lower():
        # Split by the end-token while keeping preceding text as the parameter name
        pieces = re.split(r"(?i)\bcompliance\b", text)
        params = []
        for p in pieces:
            p = p.strip().strip(",;")
            if not p:
                continue
            # Often looks like: "Enterobacteriaceae, CFU/g"  -> keep as-is
            params.append(p)
        return params

    # Fallbacks
    # First try newline/semicolon/pipe separated
    parts = re.split(r"[\n\r;|]", text)
    out = []
    for p in parts:
        p = p.strip().strip(",")
        if p:
            out.append(p)
    if out:
        return out

    # Final fallback: split on ",  "  (two spaces) or just comma for safety
    parts = [p.strip() for p in re.split(r",\s{2,}|,", text) if p.strip()]
    return parts


def format_node_label(title: str, count: int | None = None, pct: float | None = None) -> str:
    line1 = title
    line2 = None
    if count is not None and pct is not None:
        line2 = f"[{count} Samples; {pct:.1f}%]"
    elif count is not None:
        unit = "Sample" if count == 1 else "Samples"
        line2 = f"[{count} {unit}]"
    return f"{line1}\n{line2}" if line2 else line1


def build_tree_dot(df: pd.DataFrame, commodity: str, variant_choice: str | None = None) -> tuple[str, dict]:
    """Return (dot_source, stats) for the selected commodity and variant.
    stats is a dict of computed counts useful for debug/UI.
    """
    cols = {
        "commodity": "Commodity",
        "variant2": "Variant 2",
        "overall_compliance": "Overall Compliance",
        "overall_quality": "Overall Quality Classification",
        "overall_safety": "Overall Safety Classification",
        "overall_labelling": "Overall Labelling Complaince",
        "substandard_cases": "Sub-Standard Cases",
        "unsafe_cases": "Unsafe Cases",
    }

    # Filter commodity
    dfx = df[df[cols["commodity"]] == commodity].copy()
    if dfx.empty:
        raise ValueError(f"No rows for commodity: {commodity}")

    # Variant selection
    variant_counts = dfx[cols["variant2"]].fillna("(missing)").value_counts()
    if variant_choice is None:
        # Prefer 'Packed Samples' if it exists, else the mode
        variant_choice = "Packed Samples" if "Packed Samples" in variant_counts.index else variant_counts.index[0]

    dfx = dfx[dfx[cols["variant2"]].fillna("(missing)") == variant_choice]
    n_total = len(dfx)
    if n_total == 0:
        raise ValueError("No rows after filtering Variant 2 == '" + str(variant_choice) + "'")

    # Compliant vs Non-compliant
    comp_series = dfx[cols["overall_compliance"]].astype(str).str.strip().str.lower()
    n_compliant = (comp_series == COMPLIANT_TOKEN).sum()
    n_noncompliant = n_total - n_compliant

    # Quality, Safety, Labelling buckets (counts against the filtered subset)
    qual_series = dfx[cols["overall_quality"]].astype(str).str.strip().str.lower()
    saf_series = dfx[cols["overall_safety"]].astype(str).str.strip().str.lower()
    lab_series = dfx[cols["overall_labelling"]].astype(str).str.strip().str.lower()

    n_quality_sub = (qual_series == SUBSTANDARD_TOKEN).sum()
    n_safety_unsafe = (saf_series == UNSAFE_TOKEN).sum()
    n_lab_mis = lab_series.apply(lambda s: any(x in s for x in MIS_LABELLED_SET)).sum()

    # Parameter breakdowns
    # For Quality Sub-Standard
    qual_params = []
    if n_quality_sub:
        for _, row in dfx[qual_series == SUBSTANDARD_TOKEN].iterrows():
            qual_params.extend(split_parameters(row.get(cols["substandard_cases"])) )
    # Count
    from collections import Counter
    qual_param_counts = Counter([p for p in map(lambda x: x.strip(), qual_params) if p])

    # For Safety Unsafe
    saf_params = []
    if n_safety_unsafe:
        for _, row in dfx[saf_series == UNSAFE_TOKEN].iterrows():
            saf_params.extend(split_parameters(row.get(cols["unsafe_cases"])) )
    saf_param_counts = Counter([p for p in map(lambda x: x.strip(), saf_params) if p])

    # Build DOT
    def pct(n: int) -> float:
        return (n / n_total * 100.0) if n_total else 0.0

    dot = [
        "digraph G {",
        "  rankdir=TB;",
        "  graph [splines=ortho, nodesep=0.5, ranksep=0.6];",
        "  node [shape=box, style=\"rounded,filled\", color=\"#d0d0d0\", fillcolor=white, fontname=Helvetica];",
        "  edge [arrowhead=normal];",
    ]

    # Node ids
    root_id = "root"
    variant_id = "variant"
    comp_id = "comp"
    noncomp_id = "noncomp"
    qual_id = "qual"
    saf_id = "saf"
    lab_id = "lab"

    # Root & Variant nodes
    root_label = format_node_label(f"{commodity}", n_total, 100.0)
    variant_label = format_node_label(f"{variant_choice}", n_total, 100.0)

    dot.append(f'  {root_id} [label="{root_label}"];')
    dot.append(f'  {variant_id} [label="{variant_label}"];')

    # Compliant / Non-compliant
    comp_label = format_node_label("Compliant", n_compliant, pct(n_compliant))
    noncomp_label = format_node_label("Non-Compliant", n_noncompliant, pct(n_noncompliant))

    dot.append(f'  {comp_id} [label="{comp_label}", fillcolor="#d4edda"];')  # light green
    dot.append(f'  {noncomp_id} [label="{noncomp_label}", fillcolor="#f8d7da"];')  # light red

    # Buckets
    dot.append(f'  {qual_id} [label="{format_node_label("Quality Parameters\\nNon-Compliance", n_quality_sub, pct(n_quality_sub))}"];')
    dot.append(f'  {saf_id} [label="{format_node_label("Safety Parameters\\nNon-Compliance", n_safety_unsafe, pct(n_safety_unsafe))}"];')
    dot.append(f'  {lab_id} [label="{format_node_label("Labelling Attributes\\nNon-Compliance", n_lab_mis, pct(n_lab_mis))}"];')

    # Edges
    dot += [
        f"  {root_id} -> {variant_id};",
        f"  {variant_id} -> {comp_id};",
        f"  {variant_id} -> {noncomp_id};",
        f"  {noncomp_id} -> {qual_id};",
        f"  {noncomp_id} -> {saf_id};",
        f"  {noncomp_id} -> {lab_id};",
    ]

    # Quality parameter leaves
    if qual_param_counts:
        dot.append("  { rank = same; ")
        q_nodes = []
        for i, (pname, cnt) in enumerate(sorted(qual_param_counts.items(), key=lambda x: (-x[1], x[0]))):
            nid = f"q{i}"
            q_nodes.append(nid)
            label = format_node_label(pname, cnt)
            dot.append(f'    {nid} [label="{label}"];')
        dot.append("  }")
        for i, (pname, cnt) in enumerate(sorted(qual_param_counts.items(), key=lambda x: (-x[1], x[0]))):
            nid = f"q{i}"
            dot.append(f"  {qual_id} -> {nid};")

    # Safety parameter leaves
    if saf_param_counts:
        dot.append("  { rank = same; ")
        s_nodes = []
        for i, (pname, cnt) in enumerate(sorted(saf_param_counts.items(), key=lambda x: (-x[1], x[0]))):
            nid = f"s{i}"
            s_nodes.append(nid)
            label = format_node_label(pname, cnt)
            dot.append(f'    {nid} [label="{label}"];')
        dot.append("  }")
        for i, (pname, cnt) in enumerate(sorted(saf_param_counts.items(), key=lambda x: (-x[1], x[0]))):
            nid = f"s{i}"
            dot.append(f"  {saf_id} -> {nid};")

    dot.append("}")
    dot_src = "\n".join(dot)

    stats = {
        "total": n_total,
        "variant": variant_choice,
        "compliant": n_compliant,
        "non_compliant": n_noncompliant,
        "quality_substandard": n_quality_sub,
        "safety_unsafe": n_safety_unsafe,
        "labelling_mis": n_lab_mis,
        "qual_param_counts": dict(qual_param_counts),
        "saf_param_counts": dict(saf_param_counts),
    }

    return dot_src, stats


# ------------------------------
# Streamlit UI
# ------------------------------

st.set_page_config(page_title="Decision Tree Chart Generator", layout="wide")
st.title("Decision Tree Chart Generator (Compliance • Quality • Safety • Labelling)")
st.caption("Upload your Excel/CSV → choose commodity → auto-generate the tree like your turmeric example.")

uploaded = st.file_uploader("Upload dataset (Excel .xlsx or CSV)", type=["xlsx", "xls", "csv"], accept_multiple_files=False)

if uploaded is None:
    st.info("Upload your dataset to begin. Expected columns include: Commodity, Variant 2, Overall Compliance, Overall Quality Classification, Sub-Standard Cases, Overall Safety Classification, Unsafe Cases, Overall Labelling Complaince.")
    st.stop()

# Read file
if uploaded.name.lower().endswith((".xlsx", ".xls")):
    df = pd.read_excel(uploaded)
else:
    df = pd.read_csv(uploaded)

# Basic validation
required_cols = [
    "Commodity", "Variant 2", "Overall Compliance", "Overall Safety Classification",
    "Overall Quality Classification", "Sub-Standard Cases", "Unsafe Cases", "Overall Labelling Complaince"
]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    st.error("Missing required columns: " + ", ".join(missing))
    st.stop()

# Controls
commodities = sorted(df["Commodity"].dropna().unique().tolist())
commodity = st.selectbox("Commodity", commodities, index=commodities.index("Turmeric Powder") if "Turmeric Powder" in commodities else 0)

variants_all = df.loc[df["Commodity"] == commodity, "Variant 2"].fillna("(missing)").unique().tolist()
# Prefer 'Packed Samples' if present
pref_index = variants_all.index("Packed Samples") if "Packed Samples" in variants_all else 0
variant_choice = st.selectbox("Variant 2 (path node)", variants_all, index=pref_index)

# Build graph
try:
    dot_src, stats = build_tree_dot(df, commodity, variant_choice)
except Exception as e:
    st.exception(e)
    st.stop()

st.subheader("Decision Tree")
st.graphviz_chart(dot_src, use_container_width=True)

# Download options
st.download_button("Download .dot source", data=dot_src.encode("utf-8"), file_name=f"{commodity}_decision_tree.dot", mime="text/vnd.graphviz")

if graphviz is not None:
    try:
        g = graphviz.Source(dot_src)
        svg_bytes = g.pipe(format="svg")
        st.download_button("Download SVG", data=svg_bytes, file_name=f"{commodity}_decision_tree.svg", mime="image/svg+xml")
        # PNG is larger; still useful
        png_bytes = g.pipe(format="png")
        st.download_button("Download PNG", data=png_bytes, file_name=f"{commodity}_decision_tree.png", mime="image/png")
    except Exception as e:
        st.warning("SVG/PNG export requires Graphviz system binaries. If this is a hosted environment without them, the buttons may not work.")
