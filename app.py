
import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objects as go
from io import StringIO

st.set_page_config(
    page_title="ATLAS | Graph Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# DESIGN SYSTEM
# ---------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;600;700;800&display=swap');

:root {
    --paper: #F5F4EF;
    --ink: #171A24;
    --muted: #777D89;
    --blue: #315CF5;
    --coral: #F16B52;
    --line: #E3E1D9;
}

.stApp {
    background: var(--paper);
    color: var(--ink);
    font-family: 'DM Sans', sans-serif;
}
[data-testid="stSidebar"] {
    background: #ECEAE3;
    border-right: 1px solid var(--line);
}
[data-testid="stSidebar"] * {
    color: var(--ink);
}
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1500px;
}
h1, h2, h3 {
    letter-spacing: -0.04em;
    color: var(--ink);
}
.eyebrow {
    font-family: 'DM Mono', monospace;
    text-transform: uppercase;
    font-size: 11px;
    letter-spacing: 2px;
    color: var(--blue);
    margin-bottom: 10px;
}
.hero-title {
    font-size: clamp(38px, 5vw, 64px);
    font-weight: 800;
    letter-spacing: -4px;
    line-height: 1;
    margin: 0;
}
.hero-sub {
    color: var(--muted);
    font-size: 15px;
    margin-top: 14px;
    max-width: 680px;
    line-height: 1.7;
}
.section-title {
    font-size: 23px;
    font-weight: 800;
    letter-spacing: -1px;
    margin-top: 12px;
}
.metric-card {
    background: #FFFFFF;
    border: 1px solid var(--line);
    border-radius: 14px;
    padding: 19px 20px;
    min-height: 112px;
}
.metric-label {
    font-family: 'DM Mono', monospace;
    color: var(--muted);
    text-transform: uppercase;
    font-size: 10px;
    letter-spacing: 1.3px;
}
.metric-value {
    font-size: 34px;
    font-weight: 800;
    letter-spacing: -2px;
    margin-top: 8px;
}
.metric-note {
    font-size: 11px;
    color: var(--muted);
    margin-top: 2px;
}
.panel {
    background: #FFFFFF;
    border: 1px solid var(--line);
    border-radius: 15px;
    padding: 20px;
}
.small-mono {
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    color: var(--muted);
}
div[data-testid="stDataFrame"] {
    border: 1px solid var(--line);
    border-radius: 10px;
}
.stButton > button {
    border-radius: 8px;
    border: 1px solid var(--blue);
    background: var(--blue);
    color: white;
    font-weight: 700;
    min-height: 42px;
}
.stButton > button:hover {
    background: #2449D5;
    border-color: #2449D5;
    color: white;
}
hr {
    border-color: var(--line);
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# DEMO NETWORK
# ---------------------------------------------------------
def generate_demo_data():
    sizes = [18, 16, 14, 12]
    probabilities = [
        [0.24, 0.018, 0.012, 0.010],
        [0.018, 0.25, 0.015, 0.012],
        [0.012, 0.015, 0.25, 0.018],
        [0.010, 0.012, 0.018, 0.27],
    ]

    graph = nx.stochastic_block_model(
        sizes, probabilities, seed=21
    )

    names = {n: f"NODE-{n + 1:03d}" for n in graph.nodes}
    graph = nx.relabel_nodes(graph, names)

    # Add deliberate connections between communities.
    bridges = [
        ("NODE-018", "NODE-019"),
        ("NODE-034", "NODE-035"),
        ("NODE-048", "NODE-049"),
    ]
    graph.add_edges_from(bridges)

    rng = np.random.default_rng(21)
    for u, v in graph.edges:
        graph[u][v]["weight"] = round(
            float(rng.uniform(0.5, 5.0)), 2
        )

    return pd.DataFrame([
        {"source": u, "target": v, "weight": data["weight"]}
        for u, v, data in graph.edges(data=True)
    ])


# ---------------------------------------------------------
# GRAPH CONSTRUCTION AND ANALYSIS
# ---------------------------------------------------------
def build_graph(data, source_col, target_col, weight_col=None):
    clean = data.copy()
    clean = clean.dropna(subset=[source_col, target_col])
    clean[source_col] = clean[source_col].astype(str).str.strip()
    clean[target_col] = clean[target_col].astype(str).str.strip()
    clean = clean[
        (clean[source_col] != "") &
        (clean[target_col] != "") &
        (clean[source_col] != clean[target_col])
    ]

    if clean.empty:
        raise ValueError("No valid connections found in this dataset.")

    if weight_col and weight_col in clean.columns:
        clean[weight_col] = pd.to_numeric(
            clean[weight_col], errors="coerce"
        ).fillna(1).clip(lower=0.01)
        graph = nx.from_pandas_edgelist(
            clean,
            source=source_col,
            target=target_col,
            edge_attr=weight_col,
            create_using=nx.Graph()
        )
        # Standardise the edge attribute name.
        for u, v, attrs in graph.edges(data=True):
            attrs["weight"] = float(attrs.get(weight_col, 1))
    else:
        graph = nx.from_pandas_edgelist(
            clean,
            source=source_col,
            target=target_col,
            create_using=nx.Graph()
        )
        nx.set_edge_attributes(graph, 1.0, "weight")

    return graph, clean


def analyze_graph(graph):
    if graph.number_of_nodes() == 0:
        raise ValueError("The graph has no nodes.")

    degree = dict(graph.degree())
    degree_centrality = nx.degree_centrality(graph)
    betweenness = nx.betweenness_centrality(
        graph, weight=None
    )

    if graph.number_of_edges() > 0:
        pagerank = nx.pagerank(graph, weight="weight")
        communities = list(
            nx.community.greedy_modularity_communities(
                graph, weight="weight"
            )
        )
    else:
        pagerank = {node: 0 for node in graph.nodes}
        communities = [{node} for node in graph.nodes]

    community_map = {}
    for index, community in enumerate(communities):
        for node in community:
            community_map[node] = index + 1

    records = []
    for node in graph.nodes:
        records.append({
            "Node": str(node),
            "Degree": degree[node],
            "Degree Centrality": degree_centrality[node],
            "Betweenness": betweenness[node],
            "PageRank": pagerank[node],
            "Community": community_map[node]
        })

    metrics = pd.DataFrame(records)
    metrics = metrics.sort_values(
        "PageRank", ascending=False
    ).reset_index(drop=True)

    return metrics, community_map, communities


# ---------------------------------------------------------
# NETWORK VISUALISATION
# ---------------------------------------------------------
def create_network_plot(
    graph,
    community_map,
    metrics,
    show_labels=True,
    layout_name="Spring",
    size_metric="Degree"
):
    if graph.number_of_nodes() > 350:
        st.warning(
            "This network contains more than 350 nodes. "
            "The visualisation may take longer to render."
        )

    if graph.number_of_nodes() == 1:
        positions = {list(graph.nodes)[0]: np.array([0.0, 0.0])}
    elif layout_name == "Circular":
        positions = nx.circular_layout(graph)
    elif layout_name == "Shell":
        positions = nx.shell_layout(graph)
    else:
        positions = nx.spring_layout(
            graph, seed=42, iterations=55, weight="weight"
        )

    metric_lookup = metrics.set_index("Node").to_dict("index")
    palette = [
        "#315CF5", "#F16B52", "#21A58B", "#A16BE8",
        "#E4A52D", "#3A9EC5", "#D45C91", "#68717F"
    ]

    edge_x, edge_y = [], []
    for u, v in graph.edges:
        x0, y0 = positions[u]
        x1, y1 = positions[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(width=0.8, color="#C9CBD2"),
        hoverinfo="none"
    )

    node_x, node_y, node_colors = [], [], []
    node_sizes, hover_text, node_labels = [], [], []

    for node in graph.nodes:
        x, y = positions[node]
        row = metric_lookup[str(node)]

        node_x.append(x)
        node_y.append(y)
        node_colors.append(
            palette[(community_map[node] - 1) % len(palette)]
        )

        if size_metric == "PageRank":
            value = row["PageRank"]
            size = 13 + min(value * 900, 22)
        elif size_metric == "Betweenness":
            value = row["Betweenness"]
            size = 13 + min(value * 100, 22)
        else:
            value = row["Degree"]
            size = 12 + min(value * 1.4, 22)

        node_sizes.append(size)
        node_labels.append(str(node))
        hover_text.append(
            f"<b>{node}</b><br>"
            f"Degree: {row['Degree']}<br>"
            f"PageRank: {row['PageRank']:.4f}<br>"
            f"Betweenness: {row['Betweenness']:.4f}<br>"
            f"Community: {row['Community']}"
        )

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text" if show_labels else "markers",
        text=node_labels if show_labels else None,
        textposition="top center",
        textfont=dict(size=9, color="#333846"),
        hovertext=hover_text,
        hoverinfo="text",
        marker=dict(
            size=node_sizes,
            color=node_colors,
            line=dict(width=1.5, color="#FFFFFF"),
            opacity=0.94
        )
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        height=650,
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=False,
        hovermode="closest",
        xaxis=dict(
            showgrid=False, zeroline=False, showticklabels=False
        ),
        yaxis=dict(
            showgrid=False, zeroline=False, showticklabels=False
        )
    )
    return fig


# ---------------------------------------------------------
# SIDEBAR / DATA INPUT
# ---------------------------------------------------------
st.sidebar.markdown("## ◈ ATLAS")
st.sidebar.markdown(
    "<div class='small-mono'>GRAPH INTELLIGENCE STUDIO · 01</div>",
    unsafe_allow_html=True
)
st.sidebar.divider()

data_source = st.sidebar.radio(
    "DATA SOURCE",
    ["Demo network", "Upload edge list"],
    index=0
)

if data_source == "Demo network":
    raw_data = generate_demo_data()
    source_col = "source"
    target_col = "target"
    weight_col = "weight"
    st.sidebar.success("Demo network loaded")
else:
    uploaded_file = st.sidebar.file_uploader(
        "Upload a CSV edge list",
        type=["csv"]
    )

    if uploaded_file is None:
        st.title("Bring your network into focus.")
        st.write(
            "Upload a CSV containing two columns that represent "
            "connections between entities."
        )
        st.info(
            "Required: source and target columns. "
            "Optional: a numeric weight column."
        )
        st.stop()

    raw_data = pd.read_csv(uploaded_file)

    if len(raw_data.columns) < 2:
        st.error("Your CSV must contain at least two columns.")
        st.stop()

    st.sidebar.caption("Choose your connection columns")
    source_col = st.sidebar.selectbox(
        "Source column", raw_data.columns, index=0
    )
    target_col = st.sidebar.selectbox(
        "Target column", raw_data.columns, index=1
    )
    weight_options = ["None"] + list(raw_data.columns)
    weight_choice = st.sidebar.selectbox(
        "Weight column (optional)", weight_options
    )
    weight_col = None if weight_choice == "None" else weight_choice

try:
    graph, clean_data = build_graph(
        raw_data, source_col, target_col, weight_col
    )
    metrics, community_map, communities = analyze_graph(graph)
except Exception as error:
    st.error(f"Could not analyse this dataset: {error}")
    st.stop()

if graph.number_of_nodes() < 2:
    st.warning("Add more connected entities to explore the network.")
    st.stop()

# ---------------------------------------------------------
# NAVIGATION
# ---------------------------------------------------------
st.sidebar.divider()
page = st.sidebar.radio(
    "WORKSPACE",
    [
        "Overview",
        "Network Explorer",
        "Node Investigator",
        "Path Finder",
        "Data Export"
    ]
)
st.sidebar.divider()
st.sidebar.markdown(
    "<div class='small-mono'>ATLAS · DAY 07 / 60<br>NETWORK ANALYSIS LAB</div>",
    unsafe_allow_html=True
)


# ---------------------------------------------------------
# REUSABLE HEADER AND METRICS
# ---------------------------------------------------------
def page_header(kicker, title, description):
    st.markdown(f"<div class='eyebrow'>{kicker}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='hero-title'>{title}</div>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='hero-sub'>{description}</div>",
        unsafe_allow_html=True
    )
    st.write("")


def metric_card(label, value, note=""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ---------------------------------------------------------
# PAGE 1: OVERVIEW
# ---------------------------------------------------------
if page == "Overview":
    page_header(
        "NETWORK OBSERVATORY / 001",
        "See the connections.",
        "ATLAS transforms relationship data into a network you can "
        "investigate. Discover communities, influential entities, "
        "and the structure hidden inside your data."
    )

    node_count = graph.number_of_nodes()
    edge_count = graph.number_of_edges()
    density = nx.density(graph)
    avg_degree = np.mean([d for _, d in graph.degree()])

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Entities", f"{node_count:,}", "Unique network nodes")
    with c2:
        metric_card("Connections", f"{edge_count:,}", "Relationships detected")
    with c3:
        metric_card("Communities", f"{len(communities):,}", "Detected groups")
    with c4:
        metric_card("Network density", f"{density:.3f}", "Connection concentration")

    st.write("")
    left, right = st.columns([1.65, 1])

    with left:
        st.markdown("<div class='section-title'>Network at a glance</div>",
                    unsafe_allow_html=True)
        st.caption("Each dot represents an entity. Each line represents a connection.")
        fig = create_network_plot(
            graph, community_map, metrics,
            show_labels=False,
            layout_name="Spring",
            size_metric="Degree"
        )
        fig.update_layout(height=510)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown("<div class='section-title'>Most influential</div>",
                    unsafe_allow_html=True)
        st.caption("Ranked by PageRank centrality.")
        top_nodes = metrics.head(8)[
            ["Node", "Degree", "PageRank", "Community"]
        ].copy()
        top_nodes["PageRank"] = top_nodes["PageRank"].round(4)
        st.dataframe(
            top_nodes,
            hide_index=True,
            use_container_width=True,
            height=350
        )

        st.markdown("<div class='section-title'>Network structure</div>",
                    unsafe_allow_html=True)
        st.write(f"**Average connections per entity:** {avg_degree:.2f}")
        st.write(f"**Connected components:** {nx.number_connected_components(graph)}")
        st.write(f"**Largest community:** {max(map(len, communities))} entities")


# ---------------------------------------------------------
# PAGE 2: NETWORK EXPLORER
# ---------------------------------------------------------
elif page == "Network Explorer":
    page_header(
        "INTERACTIVE GRAPH / 002",
        "Explore the network.",
        "Move through the relationship map. Change the layout, "
        "inspect nodes, and see how different communities connect."
    )

    controls = st.columns([1, 1, 1])
    with controls[0]:
        layout_name = st.selectbox(
            "Graph layout", ["Spring", "Circular", "Shell"]
        )
    with controls[1]:
        size_metric = st.selectbox(
            "Node size represents",
            ["Degree", "PageRank", "Betweenness"]
        )
    with controls[2]:
        show_labels = st.checkbox("Show node labels", value=False)

    fig = create_network_plot(
        graph, community_map, metrics,
        show_labels, layout_name, size_metric
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Tip: hover over a node to inspect its metrics. "
        "Node colours represent detected communities."
    )

    community_counts = (
        metrics.groupby("Community")
        .agg(Entities=("Node", "count"))
        .reset_index()
    )
    st.markdown("<div class='section-title'>Community distribution</div>",
                unsafe_allow_html=True)
    st.bar_chart(
        community_counts.set_index("Community")["Entities"],
        color="#315CF5"
    )


# ---------------------------------------------------------
# PAGE 3: NODE INVESTIGATOR
# ---------------------------------------------------------
elif page == "Node Investigator":
    page_header(
        "ENTITY ANALYSIS / 003",
        "Investigate a node.",
        "Look beyond the connections. Understand which entities "
        "are influential, which act as bridges, and who they connect to."
    )

    selected_node = st.selectbox(
        "Select an entity",
        sorted(graph.nodes, key=str)
    )

    node_row = metrics[
        metrics["Node"] == str(selected_node)
    ].iloc[0]

    a, b, c, d = st.columns(4)
    with a:
        metric_card("Direct connections", int(node_row["Degree"]))
    with b:
        metric_card("PageRank", f"{node_row['PageRank']:.4f}")
    with c:
        metric_card("Betweenness", f"{node_row['Betweenness']:.4f}")
    with d:
        metric_card("Community", int(node_row["Community"]))

    neighbors = list(graph.neighbors(selected_node))
    st.markdown("<div class='section-title'>Direct connections</div>",
                unsafe_allow_html=True)

    if neighbors:
        neighbor_rows = []
        for neighbor in neighbors:
            details = metrics[
                metrics["Node"] == str(neighbor)
            ].iloc[0]
            neighbor_rows.append({
                "Connected entity": str(neighbor),
                "Connection weight": graph[selected_node][neighbor].get("weight", 1),
                "Degree": int(details["Degree"]),
                "Community": int(details["Community"])
            })

        st.dataframe(
            pd.DataFrame(neighbor_rows).sort_values(
                "Degree", ascending=False
            ),
            hide_index=True,
            use_container_width=True
        )
    else:
        st.info("This entity has no direct connections.")

    st.markdown("<div class='section-title'>Entity interpretation</div>",
                unsafe_allow_html=True)

    if node_row["Betweenness"] > metrics["Betweenness"].quantile(0.8):
        st.success(
            "Bridge candidate: this entity has relatively high "
            "betweenness and may connect otherwise separate parts "
            "of the network."
        )
    elif node_row["Degree"] > metrics["Degree"].median():
        st.info(
            "Highly connected entity: it has more direct connections "
            "than the median node."
        )
    else:
        st.write(
            "This entity has a relatively modest connection profile "
            "compared with the rest of the network."
        )


# ---------------------------------------------------------
# PAGE 4: PATH FINDER
# ---------------------------------------------------------
elif page == "Path Finder":
    page_header(
        "RELATIONSHIP TRACE / 004",
        "Find the connection.",
        "Choose two entities and trace the shortest route between them. "
        "Useful for exploring how distant parts of a network are linked."
    )

    nodes = sorted(graph.nodes, key=str)
    left, right = st.columns(2)

    with left:
        start_node = st.selectbox(
            "Starting entity", nodes, index=0
        )
    with right:
        end_node = st.selectbox(
            "Destination entity", nodes,
            index=min(1, len(nodes) - 1)
        )

    if st.button("TRACE CONNECTION", use_container_width=True):
        if start_node == end_node:
            st.info("Choose two different entities.")
        elif nx.has_path(graph, start_node, end_node):
            path = nx.shortest_path(
                graph, start_node, end_node
            )
            st.success(
                f"Connection found · {len(path) - 1} steps"
            )

            st.markdown("<div class='section-title'>Route sequence</div>",
                        unsafe_allow_html=True)
            for index, node in enumerate(path):
                st.markdown(
                    f"**{index + 1}.** `{node}`"
                    + ("  →" if index < len(path) - 1 else "  · DESTINATION")
                )

            route_edges = list(zip(path[:-1], path[1:]))
            route_graph = nx.Graph()
            route_graph.add_edges_from(route_edges)
            st.caption(
                f"The route passes through {len(path)} entities "
                f"and {len(route_edges)} connections."
            )
        else:
            st.warning(
                "No path exists between these entities in the current network."
            )


# ---------------------------------------------------------
# PAGE 5: EXPORT
# ---------------------------------------------------------
elif page == "Data Export":
    page_header(
        "OUTPUT / 005",
        "Take your findings.",
        "Export the analysed network, node-level metrics, and "
        "community assignments for further analysis."
    )

    nodes_export = metrics.copy()
    edges_export = pd.DataFrame([
        {
            "source": u,
            "target": v,
            "weight": data.get("weight", 1)
        }
        for u, v, data in graph.edges(data=True)
    ])

    community_export = (
        metrics.groupby("Community")
        .agg(
            entity_count=("Node", "count"),
            entities=("Node", lambda x: ", ".join(map(str, x)))
        )
        .reset_index()
    )

    st.markdown("<div class='section-title'>Export preview</div>",
                unsafe_allow_html=True)
    st.write("**Node metrics**")
    st.dataframe(nodes_export.head(10), hide_index=True, use_container_width=True)

    st.write("**Connections**")
    st.dataframe(edges_export.head(10), hide_index=True, use_container_width=True)

    st.divider()
    x1, x2, x3 = st.columns(3)

    with x1:
        st.download_button(
            "Download node metrics",
            nodes_export.to_csv(index=False).encode("utf-8"),
            "atlas_node_metrics.csv",
            "text/csv",
            use_container_width=True
        )
    with x2:
        st.download_button(
            "Download connections",
            edges_export.to_csv(index=False).encode("utf-8"),
            "atlas_connections.csv",
            "text/csv",
            use_container_width=True
        )
    with x3:
        st.download_button(
            "Download communities",
            community_export.to_csv(index=False).encode("utf-8"),
            "atlas_communities.csv",
            "text/csv",
            use_container_width=True
        )

    st.info(
        "Exports contain calculated network metrics and community "
        "assignments. They do not include the interactive graph itself."
    )
