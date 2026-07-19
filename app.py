import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import warnings

warnings.filterwarnings("ignore")


# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="PMAF Dashboard",
    page_icon="chart_with_upwards_trend",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
    [data-testid="stMetricValue"] {
        font-size: 32px;
        font-weight: bold;
    }
    [data-testid="stMetricLabel"] {
        font-size: 14px;
    }
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# =============================================================================
# CONSTANTS
# =============================================================================

DATA_PATH = Path("data/final/district_rankings.csv")

PILLAR_SCORES = [
    "Market_Size_Score",
    "Chronic_Disease_Score",
    "Acute_Disease_Score",
    "Healthcare_Score",
    "Economic_Score",
    "Development_Score"
]

PMAF_SCORES = [
    "Overall_PMAF",
    "Chronic_PMAF",
    "Acute_PMAF"
]

RANK_COLUMNS = [
    "Overall_Rank",
    "Chronic_Rank",
    "Acute_Rank"
]

PILLAR_DISPLAY_NAMES = {
    "Market_Size_Score": "Market Size",
    "Chronic_Disease_Score": "Chronic Disease",
    "Acute_Disease_Score": "Acute Disease",
    "Healthcare_Score": "Healthcare",
    "Economic_Score": "Economic",
    "Development_Score": "Development"
}

PMAF_DISPLAY_NAMES = {
    "Overall_PMAF": "Overall PMAF",
    "Chronic_PMAF": "Chronic PMAF",
    "Acute_PMAF": "Acute PMAF"
}

PILLAR_DESCRIPTIONS = {
    "Market_Size_Score": "Population and household potential for pharmaceutical products.",
    "Chronic_Disease_Score": "Burden of chronic diseases (anaemia, diabetes, hypertension, tobacco, alcohol, overweight).",
    "Acute_Disease_Score": "Burden of acute diseases (ARI, diarrhoea, malnutrition).",
    "Healthcare_Score": "Healthcare infrastructure, access, and maternal care coverage.",
    "Economic_Score": "Economic capacity, purchasing power, and banking penetration.",
    "Development_Score": "Education, infrastructure, and healthcare spending capacity."
}


# =============================================================================
# DATA LOADING
# =============================================================================

@st.cache_data
def load_data() -> pd.DataFrame:
    """Load district rankings dataset."""
    if not DATA_PATH.exists():
        st.error(f"Data file not found: {DATA_PATH}")
        st.stop()
    
    df = pd.read_csv(DATA_PATH)
    return df


@st.cache_data
def get_states(df: pd.DataFrame) -> List[str]:
    """Get sorted list of states."""
    return sorted(df["State"].unique())


@st.cache_data
def get_districts_by_state(df: pd.DataFrame, state: str) -> List[str]:
    """Get districts for a specific state."""
    return sorted(df[df["State"] == state]["District"].unique())


@st.cache_data
def get_top_states(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Get top states by average Overall PMAF."""
    top_states = df.groupby("State")[PMAF_SCORES].mean().reset_index()
    top_states["District_Count"] = df.groupby("State").size().values
    top_states = top_states.nlargest(n, "Overall_PMAF")
    return top_states


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def format_score(score: float) -> str:
    """Format score for display."""
    return f"{score:.2f}"


def get_district_data(df: pd.DataFrame, state: str, district: str) -> pd.Series:
    """Get data for a specific district."""
    return df[(df["State"] == state) & (df["District"] == district)].iloc[0]


def create_metric_card(label: str, value: float, suffix: str = "") -> None:
    """Create a metric card."""
    st.metric(label=label, value=f"{value:.2f}{suffix}")


def district_search(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    """Allow user to search for district."""
    search_term = st.text_input(
        "Search for a district:",
        placeholder="e.g., Pune, Mumbai, Delhi...",
        key=f"search_{id(df)}"
    )
    
    if search_term and len(search_term) > 0:
        matches = df[
            (df["District"].str.contains(search_term, case=False, na=False)) |
            (df["State"].str.contains(search_term, case=False, na=False))
        ]
        
        if len(matches) > 0:
            match_labels = matches["District"] + ", " + matches["State"]
            selected_label = st.selectbox(
                "Select from matches:",
                match_labels.values,
                key=f"select_{id(df)}"
            )
            
            if selected_label:
                parts = selected_label.rsplit(", ", 1)
                district = parts[0]
                state = parts[1]
                return state, district
    
    return None, None


# =============================================================================
# HEADER AND FOOTER FUNCTIONS
# =============================================================================

def render_header() -> None:
    """Render branded header."""
    st.markdown("=" * 70)
    st.markdown("**PMAF DASHBOARD**")
    st.markdown("District Pharmaceutical Market Attractiveness Framework")
    st.markdown("")
    st.markdown("Sun Pharma  |  Trilytics 2026")
    st.markdown("")
    st.markdown("851 Districts Analyzed  |  Overall + Chronic + Acute")
    st.markdown("=" * 70)
    st.markdown("")


def render_footer() -> None:
    """Render footer."""
    st.markdown("")
    st.markdown("=" * 70)
    st.markdown("PMAF Dashboard  |  Sun Pharma  |  Trilytics 2026")
    st.caption("Production-Ready Data Pipeline  |  IIT ISM Dhanbad")
    st.markdown("=" * 70)


def render_pipeline():
    """Render PMAF data pipeline."""

    st.subheader("PMAF Data Pipeline")

    stages = [
        "📥 Raw Data",
        "🧹 Data Cleaning",
        "⚙️ Feature Engineering",
        "📊 Normalization",
        "🧮 PMAF Scoring",
        "🏆 District Ranking"
    ]

    descriptions = [
        "Census | NFHS | RHS | SECC | RBI",
        "Cleaning, validation and preprocessing",
        "Feature creation into 6 PMAF pillars",
        "Min-Max normalization",
        "Weighted PMAF score computation",
        "851 districts ranked"
    ]

    cols = st.columns(6)

    for col, stage, desc in zip(cols, stages, descriptions):
        with col:
            st.markdown(f"### {stage}")
            st.caption(desc)


# =============================================================================
# CHART FUNCTIONS
# =============================================================================

def create_histogram(df: pd.DataFrame, column: str, title: str) -> go.Figure:
    """Create histogram for PMAF scores."""
    fig = px.histogram(
        df,
        x=column,
        nbins=50,
        title=title,
        labels={column: "PMAF Score"},
        color_discrete_sequence=["#4F46E5"]
    )
    fig.update_layout(
        showlegend=False,
        hovermode="x unified",
        height=400,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    return fig


def create_top_10_bar(df: pd.DataFrame, score_col: str, title: str) -> go.Figure:
    """Create horizontal bar chart for top 10 districts."""
    top_10 = df.nlargest(10, score_col).sort_values(score_col)
    top_10["District_Label"] = top_10["District"] + ", " + top_10["State"]
    
    fig = px.bar(
        top_10,
        x=score_col,
        y="District_Label",
        orientation="h",
        title=title,
        labels={score_col: "PMAF Score", "District_Label": ""},
        color=score_col,
        color_continuous_scale="Blues"
    )
    fig.update_layout(
        showlegend=False,
        height=400,
        margin=dict(l=0, r=0, t=40, b=0),
        xaxis_range=[0, 100]
    )
    return fig


def create_radar_chart(pillars: Dict[str, float], title: str) -> go.Figure:
    """Create radar chart for pillar scores."""
    categories = list(pillars.keys())
    values = list(pillars.values())
    
    fig = go.Figure(
        data=go.Scatterpolar(
            r=values,
            theta=categories,
            fill="toself",
            name="Scores",
            marker=dict(color="#4F46E5")
        )
    )
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=False,
        title=title,
        height=500,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    return fig


def create_comparison_bars(
    district1_pillars: Dict[str, float],
    district2_pillars: Dict[str, float],
    district1_name: str,
    district2_name: str
) -> go.Figure:
    """Create grouped bar chart for district comparison."""
    categories = list(district1_pillars.keys())
    
    fig = go.Figure(
        data=[
            go.Bar(name=district1_name, x=categories, y=list(district1_pillars.values())),
            go.Bar(name=district2_name, x=categories, y=list(district2_pillars.values()))
        ]
    )
    fig.update_layout(
        barmode="group",
        title="Pillar Score Comparison",
        yaxis_range=[0, 1],
        height=400,
        margin=dict(l=0, r=0, t=40, b=0),
        hovermode="x unified"
    )
    return fig


def create_ranking_chart(df: pd.DataFrame, score_col: str, n: int = 20) -> go.Figure:
    """Create horizontal bar chart for rankings."""
    top_n = df.nlargest(n, score_col).sort_values(score_col)
    top_n["District_Label"] = top_n["District"] + ", " + top_n["State"]
    
    fig = px.bar(
        top_n,
        x=score_col,
        y="District_Label",
        orientation="h",
        labels={score_col: "PMAF Score", "District_Label": ""},
        color=score_col,
        color_continuous_scale="Viridis"
    )
    fig.update_layout(
        showlegend=False,
        height=600,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis_range=[0, 100]
    )
    return fig


def create_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    """Create correlation heatmap."""
    corr_cols = PILLAR_SCORES + PMAF_SCORES
    corr_matrix = df[corr_cols].corr()
    
    labels = [
        PILLAR_DISPLAY_NAMES.get(col, col) if col in PILLAR_DISPLAY_NAMES
        else PMAF_DISPLAY_NAMES.get(col, col)
        for col in corr_cols
    ]
    
    fig = go.Figure(
        data=go.Heatmap(
            z=corr_matrix.values,
            x=labels,
            y=labels,
            colorscale="RdBu",
            zmid=0,
            zmin=-1,
            zmax=1
        )
    )
    fig.update_layout(
        title="Correlation Heatmap: Pillars and PMAF Scores",
        height=600,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    return fig


def create_scatter(df: pd.DataFrame, x_col: str, y_col: str, title: str) -> go.Figure:
    """Create scatter plot."""
    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        hover_data=["State", "District"],
        title=title,
        labels={
            x_col: PILLAR_DISPLAY_NAMES.get(x_col, x_col),
            y_col: PMAF_DISPLAY_NAMES.get(y_col, y_col)
        },
        color=y_col,
        color_continuous_scale="Viridis",
        size_max=15
    )
    fig.update_layout(
        height=500,
        margin=dict(l=0, r=0, t=40, b=0),
        hovermode="closest"
    )
    return fig


# =============================================================================
# PAGE SECTIONS
# =============================================================================

def page_executive_overview(df: pd.DataFrame) -> None:
    """Executive Overview page."""
    render_header()
    st.title("Executive Overview")
    st.markdown("KPIs and market insights at a glance.")
    
    st.markdown("")
    
    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        create_metric_card("Total Districts", len(df))
    with col2:
        create_metric_card("Avg Overall PMAF", df["Overall_PMAF"].mean())
    with col3:
        create_metric_card("Max Overall PMAF", df["Overall_PMAF"].max())
    with col4:
        create_metric_card("Min Overall PMAF", df["Overall_PMAF"].min())
    
    st.markdown("---")
    
    # Top States Summary
    st.subheader("Top 10 States by Average Overall PMAF")
    top_states = get_top_states(df, n=10)
    top_states_display = top_states[["State", "Overall_PMAF", "District_Count"]].copy()
    top_states_display.columns = ["State", "Avg Overall PMAF", "Districts"]
    top_states_display["Avg Overall PMAF"] = top_states_display["Avg Overall PMAF"].round(2)
    
    st.dataframe(top_states_display, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # Histograms
    st.subheader("Market Distribution Analysis")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.plotly_chart(
            create_histogram(df, "Overall_PMAF", "Overall PMAF"),
            use_container_width=True
        )
    with col2:
        st.plotly_chart(
            create_histogram(df, "Chronic_PMAF", "Chronic PMAF"),
            use_container_width=True
        )
    with col3:
        st.plotly_chart(
            create_histogram(df, "Acute_PMAF", "Acute PMAF"),
            use_container_width=True
        )
    
    st.markdown("---")
    
    # Top 10 Charts
    st.subheader("Top 10 Districts by Market Type")
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(
            create_top_10_bar(df, "Overall_PMAF", "Overall PMAF"),
            use_container_width=True
        )
    with col2:
        st.plotly_chart(
            create_top_10_bar(df, "Chronic_PMAF", "Chronic PMAF"),
            use_container_width=True
        )
    
    st.plotly_chart(
        create_top_10_bar(df, "Acute_PMAF", "Acute PMAF"),
        use_container_width=True
    )
    
    st.markdown("---")
    
    # Executive Insight
    with st.expander("Executive Insight"):
        top_overall = df.loc[df["Overall_PMAF"].idxmax()]
        top_chronic = df.loc[df["Chronic_PMAF"].idxmax()]
        top_acute = df.loc[df["Acute_PMAF"].idxmax()]
        
        insight = f"""
**Key Findings:**

- **Highest Overall Attractiveness:** {top_overall['District']}, {top_overall['State']} (PMAF: {top_overall['Overall_PMAF']:.2f})
- **Highest Chronic Market:** {top_chronic['District']}, {top_chronic['State']} (PMAF: {top_chronic['Chronic_PMAF']:.2f})
- **Highest Acute Market:** {top_acute['District']}, {top_acute['State']} (PMAF: {top_acute['Acute_PMAF']:.2f})

**Market Composition:**
- Districts with Overall PMAF > 70: {len(df[df['Overall_PMAF'] > 70])}
- Districts with Chronic PMAF > 70: {len(df[df['Chronic_PMAF'] > 70])}
- Districts with Acute PMAF > 70: {len(df[df['Acute_PMAF'] > 70])}
- Average Overall PMAF across all districts: {df['Overall_PMAF'].mean():.2f}
"""
        st.markdown(insight)
    
    render_footer()


def page_district_explorer(df: pd.DataFrame) -> None:
    """District Explorer page."""
    render_header()
    st.title("District Explorer")
    st.markdown("Detailed analysis of individual districts.")
    
    st.markdown("")
    
    # Search option
    st.subheader("Find a District")
    search_col1, search_col2 = st.columns([2, 1])
    
    with search_col1:
        search_state, search_district = district_search(df)
    
    with search_col2:
        st.write("")
        st.write("")
        use_search = search_state is not None and search_district is not None
    
    # Fallback to dropdown
    if not use_search:
        st.subheader("Or Browse by State and District")
        states = get_states(df)
        selected_state = st.selectbox("Select State:", states, key="explorer_state")
        
        districts = get_districts_by_state(df, selected_state)
        selected_district = st.selectbox("Select District:", districts, key="explorer_district")
    else:
        selected_state = search_state
        selected_district = search_district
    
    if selected_state and selected_district:
        district_data = get_district_data(df, selected_state, selected_district)
        
        st.markdown("---")
        st.subheader(f"{selected_district}, {selected_state}")
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            create_metric_card("Overall PMAF", district_data["Overall_PMAF"])
        with col2:
            create_metric_card("Chronic PMAF", district_data["Chronic_PMAF"])
        with col3:
            create_metric_card("Acute PMAF", district_data["Acute_PMAF"])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Overall Rank", int(district_data["Overall_Rank"]))
        with col2:
            st.metric("Chronic Rank", int(district_data["Chronic_Rank"]))
        with col3:
            st.metric("Acute Rank", int(district_data["Acute_Rank"]))
        
        st.markdown("---")
        
        # Radar Chart
        st.subheader("Pillar Scores Profile")
        pillars_dict = {
            PILLAR_DISPLAY_NAMES[col]: district_data[col]
            for col in PILLAR_SCORES
        }
        st.plotly_chart(
            create_radar_chart(pillars_dict, f"District Profile: {selected_district}"),
            use_container_width=True
        )
        
        # Dataframe
        st.markdown("---")
        with st.expander("View All Metrics"):
            display_cols = ["State", "District"] + PILLAR_SCORES + PMAF_SCORES + RANK_COLUMNS
            st.dataframe(district_data[display_cols].to_frame().T, use_container_width=True)
    
    render_footer()


def page_compare_districts(df: pd.DataFrame) -> None:
    """Compare Districts page."""
    render_header()
    st.title("Compare Districts")
    st.markdown("Side-by-side comparison of two districts.")
    
    st.markdown("")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("District 1")
        state1 = st.selectbox("Select State:", get_states(df), key="comp_state1")
        districts1 = get_districts_by_state(df, state1)
        district1 = st.selectbox("Select District:", districts1, key="comp_dist1")
    
    with col2:
        st.subheader("District 2")
        state2 = st.selectbox(
            "Select State:",
            get_states(df),
            key="comp_state2",
            index=1 if len(get_states(df)) > 1 else 0
        )
        districts2 = get_districts_by_state(df, state2)
        district2 = st.selectbox("Select District:", districts2, key="comp_dist2")
    
    if district1 and district2:
        data1 = get_district_data(df, state1, district1)
        data2 = get_district_data(df, state2, district2)
        
        st.markdown("---")
        
        # Metric Cards
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(f"{district1}, {state1}")
            create_metric_card("Overall PMAF", data1["Overall_PMAF"])
            create_metric_card("Chronic PMAF", data1["Chronic_PMAF"])
            create_metric_card("Acute PMAF", data1["Acute_PMAF"])
        
        with col2:
            st.subheader(f"{district2}, {state2}")
            create_metric_card("Overall PMAF", data2["Overall_PMAF"])
            create_metric_card("Chronic PMAF", data2["Chronic_PMAF"])
            create_metric_card("Acute PMAF", data2["Acute_PMAF"])
        
        st.markdown("---")
        
        # Comparison Chart
        st.subheader("Pillar Score Comparison")
        pillars1 = {PILLAR_DISPLAY_NAMES[col]: data1[col] for col in PILLAR_SCORES}
        pillars2 = {PILLAR_DISPLAY_NAMES[col]: data2[col] for col in PILLAR_SCORES}
        
        st.plotly_chart(
            create_comparison_bars(pillars1, pillars2, f"{district1}, {state1}", f"{district2}, {state2}"),
            use_container_width=True
        )
        
        # Winner highlighting
        st.markdown("---")
        with st.expander("Detailed Pillar Performance"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write("**Pillar**")
            with col2:
                st.write(f"**{district1}**")
            with col3:
                st.write(f"**{district2}**")
            
            for pillar in PILLAR_SCORES:
                display_name = PILLAR_DISPLAY_NAMES[pillar]
                val1 = data1[pillar]
                val2 = data2[pillar]
                winner = district1 if val1 > val2 else district2 if val2 > val1 else "Tied"
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write(f"{display_name}")
                with col2:
                    st.write(f"{val1:.3f}")
                with col3:
                    st.write(f"{val2:.3f}")
    
    render_footer()


def page_rankings(df: pd.DataFrame) -> None:
    """Rankings page."""
    render_header()
    st.title("District Rankings")
    st.markdown("View rankings and filter by market type and scope.")
    
    st.markdown("")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        rank_type = st.radio("Select Ranking Type:", ["Overall", "Chronic", "Acute"], horizontal=True)
    
    with col2:
        top_n_options = [10, 25, 50, 100, 851]
        top_n = st.selectbox("Top N Districts:", top_n_options, index=2)
    
    score_map = {"Overall": "Overall_PMAF", "Chronic": "Chronic_PMAF", "Acute": "Acute_PMAF"}
    score_col = score_map[rank_type]
    
    if top_n == 851:
        df_ranked = df.sort_values(score_col, ascending=False)
    else:
        df_ranked = df.nlargest(top_n, score_col)
    
    st.markdown("---")
    st.subheader(f"Top {top_n if top_n < 851 else 'All'} Districts: {rank_type} PMAF")
    
    st.plotly_chart(
        create_ranking_chart(df_ranked, score_col, n=min(top_n, 30)),
        use_container_width=True
    )
    
    st.markdown("---")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.dataframe(
            df_ranked[["State", "District"] + PMAF_SCORES + RANK_COLUMNS].head(top_n),
            use_container_width=True,
            hide_index=True
        )
    
    with col2:
        st.write("")
        st.write("")
        csv = df_ranked[["State", "District"] + PMAF_SCORES + RANK_COLUMNS].to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"{rank_type.lower()}_rankings.csv",
            mime="text/csv"
        )
    
    render_footer()


def page_analytics(df: pd.DataFrame) -> None:
    """Analytics page."""
    render_header()
    st.title("Analytics")
    st.markdown("Detailed analysis and statistical insights.")
    
    st.markdown("")
    
    # Distribution Analysis
    st.subheader("1. Distribution Analysis")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.plotly_chart(
            create_histogram(df, "Overall_PMAF", "Overall PMAF"),
            use_container_width=True
        )
    with col2:
        st.plotly_chart(
            create_histogram(df, "Chronic_PMAF", "Chronic PMAF"),
            use_container_width=True
        )
    with col3:
        st.plotly_chart(
            create_histogram(df, "Acute_PMAF", "Acute PMAF"),
            use_container_width=True
        )
    
    st.markdown("---")
    
    # Correlation Heatmap
    st.subheader("2. Pillar Correlation Analysis")
    st.plotly_chart(
        create_correlation_heatmap(df),
        use_container_width=True
    )
    
    st.markdown("---")
    
    # Scatter Plots
    st.subheader("3. Score Relationships")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(
            create_scatter(df, "Economic_Score", "Overall_PMAF", "Economic Capacity vs Overall PMAF"),
            use_container_width=True
        )
    
    with col2:
        st.plotly_chart(
            create_scatter(df, "Healthcare_Score", "Overall_PMAF", "Healthcare Access vs Overall PMAF"),
            use_container_width=True
        )
    
    st.plotly_chart(
        create_scatter(df, "Market_Size_Score", "Overall_PMAF", "Market Size vs Overall PMAF"),
        use_container_width=True
    )
    
    render_footer()


def page_about_pmaf(df: pd.DataFrame) -> None:
    """About PMAF page."""
    render_header()
    st.title("About PMAF")
    st.markdown("Understand the framework, methodology, and data sources.")
    
    st.markdown("")
    
    # PMAF Definition
    st.subheader("What is PMAF?")
    st.markdown("""
The **Pharmaceutical Market Attractiveness Framework (PMAF)** is a data-driven
district-level scoring system designed to identify and rank pharmaceutical market
opportunities across 851 Indian districts. Higher PMAF scores indicate greater
strategic business potential.
""")
    
    st.markdown("---")
    
    # Three Indices
    st.subheader("Three Market Indices")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Overall PMAF**")
        st.caption("Comprehensive market potential across all therapy areas")
    
    with col2:
        st.markdown("**Chronic PMAF**")
        st.caption("Attractiveness for chronic disease therapeutics")
    
    with col3:
        st.markdown("**Acute PMAF**")
        st.caption("Attractiveness for acute disease therapeutics")
    
    st.markdown("---")
    
    # Six Pillars
    st.subheader("Six Pillars")
    
    col1, col2 = st.columns(2)
    for idx, (pillar, description) in enumerate(PILLAR_DESCRIPTIONS.items()):
        if idx % 2 == 0:
            col = col1
        else:
            col = col2
        
        with col:
            with st.expander(PILLAR_DISPLAY_NAMES[pillar]):
                st.write(description)
    
    st.markdown("---")
    
    # Data Sources
    st.subheader("Data Sources")
    source_cols = st.columns(5)
    sources = [
        ("Census", "Demographic & Economic"),
        ("NFHS-5", "Health & Nutrition"),
        ("RHS", "Healthcare Infrastructure"),
        ("SECC", "Economic & Social"),
        ("RBI", "Banking & Finance")
    ]
    
    for col, (source_name, description) in zip(source_cols, sources):
        with col:
            st.markdown(f"**{source_name}**")
            st.caption(description)
    
    st.markdown("---")
    
    # Pipeline
    render_pipeline()
    
    st.markdown("---")
    
    # Use Cases
    st.subheader("Business Applications")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Sales Deployment**")
        st.caption("Optimize sales force allocation and territory planning")
    
    with col2:
        st.markdown("**Market Prioritization**")
        st.caption("Identify high-potential districts for expansion")
    
    with col3:
        st.markdown("**Resource Allocation**")
        st.caption("Allocate marketing and commercial resources strategically")
    
    st.markdown("---")
    
    # Methodology Note
    with st.expander("Methodology Notes"):
        st.markdown("""
**Normalization:**
- All features normalized to [0, 1] using Min-Max normalization
- Landless Manual Labour reversed to reflect inverse relationship with attractiveness

**Pillar Scoring:**
- Each pillar calculated as arithmetic mean of constituent features

**PMAF Scoring:**
- Weighted composition of six pillars
- Scaled to 0-100 range
- Three variants with different weights for Overall, Chronic, and Acute markets

**Data Quality:**
- Missing values imputed using median strategy
- All features validated for completeness and bounds
- No NaN or infinite values in final outputs
""")
    
    render_footer()


# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """Main application."""
    df = load_data()
    
    # Sidebar
    st.sidebar.title("PMAF Dashboard")
    st.sidebar.markdown("District Pharmaceutical Market Attractiveness Framework")
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio(
        "Navigation",
        [
            "Executive Overview",
            "District Explorer",
            "Compare Districts",
            "Rankings",
            "Analytics",
            "About PMAF"
        ]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.caption("Built with Streamlit & Plotly")
    st.sidebar.caption("Data Pipeline: Production-Ready")
    
    # Route to pages
    if page == "Executive Overview":
        page_executive_overview(df)
    elif page == "District Explorer":
        page_district_explorer(df)
    elif page == "Compare Districts":
        page_compare_districts(df)
    elif page == "Rankings":
        page_rankings(df)
    elif page == "Analytics":
        page_analytics(df)
    elif page == "About PMAF":
        page_about_pmaf(df)


if __name__ == "__main__":
    main()