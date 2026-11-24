"""Case-based recommender for project management (PM) tool selection.

Implements the model specified in Section 3.10.2 of the thesis:

1. Each questionnaire response is a case: a project profile, the tool used,
   and the fit the respondent reported (1-5).
2. The similarity between a new project and each past case is computed with
   Gower's (1971) coefficient over the project profile (Q4-Q9).
3. For each tool, the K most similar past cases that used that tool are
   retrieved (K = 5).
4. The tool's alignment score is the similarity-weighted average of those
   cases' fit ratings, rescaled to 0-1:  A(p, t) = sum(s_i * f_i) / sum(s_i),
   where f_i = (FIT_i - 1) / 4.
5. Tools are ranked by alignment score.
"""

from pathlib import Path

import numpy as np
import pandas as pd

K = 5  # number of similar cases retrieved per tool

# Answer options, in questionnaire order.
METHODOLOGIES = ["Agile / Scrum", "Kanban / Lean", "Predictive / Waterfall", "Hybrid"]
TEAM_SIZES = {"1–5": 1, "6–15": 2, "16–50": 3, "51–150": 4, "More than 150": 5}
DISTRIBUTIONS = [
    "Primarily co-located",
    "Hybrid",
    "Fully remote/distributed",
    "Distributed across multiple countries or time zones",
]
DEPENDENCY = {"Low": 1, "Moderate": 2, "High": 3}
VOLATILITY = {"Rarely": 1, "Occasionally": 2, "Frequently": 3, "Very frequently": 4}
CAPABILITIES = {
    "task_management": "Task management and workflows",
    "scheduling": "Timeline, Gantt and milestone scheduling",
    "dependency_management": "Dependency management",
    "resource_allocation": "Resource allocation and workload management",
    "agile_backlog": "Agile sprints and backlog management",
    "risk_issues": "Risk and issue management",
    "budget_cost": "Budget and cost management",
    "reporting": "Reporting and dashboards",
    "automation": "Workflow automation",
    "collaboration": "Team collaboration",
}

# Gower variable types and the range of each ordinal scale.
NOMINAL = ["methodology", "distribution"]
ORDINAL_RANGES = {"team_size": 4, "dependency": 2, "volatility": 3}
ORDINAL_RANGES.update({c: 4 for c in CAPABILITIES})


def load_cases(path=None):
    path = Path(path) if path else Path(__file__).with_name("cases.csv")
    return pd.read_csv(path)


def gower_similarity(profile, cases):
    """Gower similarity (0-1) between one project profile and every case.

    Nominal variables score 1 for a match and 0 otherwise. Ordinal variables
    score 1 - |difference| / range of the scale. Variables missing in either
    the profile or a case are left out of that case's average.
    """
    parts, present = [], []
    for v in NOMINAL:
        if profile.get(v) is None:
            continue
        col = cases[v]
        ok = col.notna()
        parts.append(np.where(ok, (col == profile[v]).astype(float), 0.0))
        present.append(ok.astype(float).values)
    for v, rng in ORDINAL_RANGES.items():
        if profile.get(v) is None:
            continue
        col = cases[v]
        ok = col.notna()
        parts.append(np.where(ok, 1 - (col - profile[v]).abs() / rng, 0.0))
        present.append(ok.astype(float).values)
    parts, present = np.vstack(parts), np.vstack(present)
    return parts.sum(axis=0) / present.sum(axis=0)


def recommend(profile, cases=None, k=K):
    """Rank the tools in the case base for a project profile.

    Returns (ranking, evidence): ranking is a DataFrame with one row per tool;
    evidence maps each tool to the similar cases behind its score.
    """
    cases = load_cases() if cases is None else cases
    cases = cases.copy()
    cases["similarity"] = gower_similarity(profile, cases)
    cases["fit_scaled"] = (cases["fit"] - 1) / 4

    rows, evidence = [], {}
    for tool, group in cases.groupby("tool"):
        top = group.nlargest(k, "similarity")
        score = (top["similarity"] * top["fit_scaled"]).sum() / top["similarity"].sum()
        rows.append({
            "tool": tool,
            "alignment_score": score,
            "cases_for_tool": len(group),
            "mean_similarity": top["similarity"].mean(),
        })
        evidence[tool] = top
    ranking = (pd.DataFrame(rows)
               .sort_values(["alignment_score", "mean_similarity"], ascending=False)
               .reset_index(drop=True))
    ranking.insert(0, "rank", range(1, len(ranking) + 1))
    return ranking, evidence
