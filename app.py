"""Streamlit prototype of the case-based PM tool recommender."""

import pandas as pd
import streamlit as st

from recommender import (
    CAPABILITIES, DEPENDENCY, DISTRIBUTIONS, K, METHODOLOGIES, TEAM_SIZES,
    VOLATILITY, load_cases, recommend,
)

st.set_page_config(page_title="PM Tool Recommender", page_icon="🧭", layout="wide")

cases = load_cases()
tool_counts = cases["tool"].value_counts()

st.title("PM Tool Recommender")
st.write(
    "Describe your IT project and the capabilities it needs. The system compares it with "
    f"{len(cases)} completed projects reported by project management professionals in the "
    "Nigerian IT sector, and grades each tool by how well it fitted the most similar of those projects."
)
st.info(
    "Tools covered: " + ", ".join(f"**{t}** ({n} projects)" for t, n in tool_counts.items())
    + ". Only tools reported by at least 20 respondents are included."
)

FIT_LABELS = {1: "Very poor", 2: "Poor", 3: "Moderate", 4: "Good", 5: "Excellent"}
REV_TEAM = {v: k for k, v in TEAM_SIZES.items()}
REV_DEP = {v: k for k, v in DEPENDENCY.items()}
REV_VOL = {v: k for k, v in VOLATILITY.items()}
SHORT_DIST = {
    "Primarily co-located": "Co-located",
    "Hybrid": "Hybrid",
    "Fully remote/distributed": "Fully remote",
    "Distributed across multiple countries or time zones": "Multi-country",
}

with st.form("project_form"):
    left, right = st.columns(2)
    with left:
        st.subheader("1. Project characteristics")
        methodology = st.selectbox("Delivery methodology", METHODOLOGIES)
        team_size = st.selectbox("Team size", list(TEAM_SIZES), index=1)
        distribution = st.selectbox("Team distribution", DISTRIBUTIONS, index=2)
        dependency = st.select_slider("Dependency complexity", list(DEPENDENCY), value="Moderate")
        volatility = st.select_slider("How often requirements change", list(VOLATILITY), value="Occasionally")
    with right:
        st.subheader("2. Capabilities the project needs")
        st.caption("1 = Not needed · 5 = Critical")
        caps = {key: st.slider(label, 1, 5, 4) for key, label in CAPABILITIES.items()}
    submitted = st.form_submit_button("Recommend tools", type="primary", width="stretch")

if submitted:
    st.session_state["profile"] = {
        "methodology": methodology,
        "team_size": TEAM_SIZES[team_size],
        "distribution": distribution,
        "dependency": DEPENDENCY[dependency],
        "volatility": VOLATILITY[volatility],
        **caps,
    }

if "profile" in st.session_state:
    ranking, evidence = recommend(st.session_state["profile"], cases)
    st.divider()
    st.subheader("3. Recommended tools")
    best = ranking.iloc[0]
    st.success(f"Best match: **{best['tool']}**, with an alignment score of **{best['alignment_score']:.0%}**.")

    cols = st.columns(len(ranking))
    for col, (_, row) in zip(cols, ranking.iterrows()):
        with col:
            st.metric(f"#{row['rank']}  {row['tool']}", f"{row['alignment_score']:.0%}")
            st.progress(float(row["alignment_score"]))
            st.caption(
                f"Based on the {K} most similar of {row['cases_for_tool']} {row['tool']} projects "
                f"(average similarity {row['mean_similarity']:.0%})."
            )

    st.subheader("4. Why these scores?")
    st.write(
        "Each score is the average fit reported on the most similar past projects that used the tool, "
        "with more similar projects counting more. These are the projects behind each score."
    )
    for tool in ranking["tool"]:
        ev = evidence[tool]
        table = pd.DataFrame({
            "Similarity": ev["similarity"].map("{:.0%}".format),
            "Reported fit": ev["fit"].map(lambda f: f"{f} – {FIT_LABELS[f]}"),
            "Methodology": ev["methodology"],
            "Team size": ev["team_size"].map(REV_TEAM),
            "Distribution": ev["distribution"].map(SHORT_DIST),
            "Dependencies": ev["dependency"].map(REV_DEP),
            "Requirement change": ev["volatility"].map(REV_VOL),
        })
        with st.expander(f"{tool}: the {K} most similar projects", expanded=tool == best["tool"]):
            st.dataframe(table, hide_index=True, width="stretch")

with st.expander("How the recommender works"):
    st.markdown(
        f"""
1. **Case base.** {len(cases)} completed IT projects from the study's survey. Each records the project's
   characteristics, the capabilities it needed, the tool used, and how well that tool fitted (1–5).
2. **Similarity.** Your project is compared with every past project using Gower's coefficient, which handles
   the mix of categories (methodology, distribution) and rating scales (everything else) in a project profile.
3. **Retrieval.** For each tool, the {K} most similar past projects that used it are retrieved.
4. **Scoring.** The tool's alignment score is the similarity-weighted average of those projects' fit ratings,
   rescaled from 0% (very poor fit) to 100% (excellent fit).
5. **Ranking.** Tools are ranked by alignment score.
"""
    )
st.caption("Prototype developed for a master's thesis in Project Management Technology. For research use.")
