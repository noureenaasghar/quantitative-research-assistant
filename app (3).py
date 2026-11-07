import os
import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from groq import Groq

# ============================================================
# CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="AI Quantitative Research Assistant",
    page_icon="🔬",
    layout="wide"
)

MODEL = "openai/gpt-oss-20b"
TOPIC_DEFAULT = (
    "Sex-Based Differences in Kidney-Related Healthcare Utilization, "
    "Monitoring, and Clinical Outcomes Among Patients with "
    "Tuberous Sclerosis Complex (TSC)"
)
AIM_DEFAULT = (
    "To examine whether males and females with TSC differ in "
    "kidney-related healthcare utilization and monitoring."
)

MODULES = [
    ("1. Research Questions", "Develop the primary question, aim, objectives, exposure and outcomes."),
    ("2. Study Design & Methodology", "Define population, eligibility, index date, follow-up and design."),
    ("3. Variables & Data Collection", "Build a variable dictionary and data collection framework."),
    ("4. Statistical Analysis Planner", "Develop a cautious, outcome-led statistical analysis plan."),
    ("5. Advanced Statistical Methods", "Understand candidate models, assumptions and interpretation."),
    ("6. Research Quality Reviewer", "Review feasibility, bias, validity and methodological consistency."),
    ("7. Research Gap & Literature Review", "Structure the literature search and provisional research gap."),
    ("8. Proposal Builder", "Assemble the research proposal from the decisions made so far."),
    ("9. Scientific Validation", "Audit a previous AI output against its original prompt."),
    ("10. Statistical Analysis & Visualization", "Upload data, calculate statistics, create graphs and export results.")
]

MODULE_PROMPTS = {
"1. Research Questions": """Refine the research question for the stated topic. Provide a primary research question, aim, specific objectives, exposure, primary outcome, secondary outcomes, target population, and unit of analysis. Distinguish descriptive, associational, predictive and causal aims. Do not invent data or follow-up periods. Identify decisions still needed.""",
"2. Study Design & Methodology": """Develop a study-design framework. Discuss suitable observational designs, target population, data source, eligibility criteria, index date, baseline period, follow-up, comparison groups, potential confounding, selection bias and information bias. Separate decisions supported by supplied information from conditional decisions. Do not invent database capabilities or dates.""",
"3. Variables & Data Collection": """Create a variable dictionary with construct, operational definition, data type, possible source, coding considerations and limitations. Organize exposure, outcomes, demographics, clinical covariates, follow-up and data-quality fields. Mark variables that may not exist in claims data (such as laboratory values, imaging detail or disease severity).""",
"4. Statistical Analysis Planner": """Create an outcome-led statistical analysis plan. Start with descriptive statistics. For each outcome specify its type, unit of analysis, denominator, follow-up and candidate analysis only when justified. Discuss effect estimates, confidence intervals, assumptions, diagnostics, missingness and sensitivity analyses. Do not prescribe methods before clarifying data structure.""",
"5. Advanced Statistical Methods": """Explain only methods relevant to the supplied research question and data structure. For each candidate method explain purpose, required data, assumptions, diagnostics, effect estimate and limitations. Distinguish logistic, modified Poisson, Poisson, negative binomial, Cox, Fine-Gray, GEE, mixed-effects and propensity-score approaches. Do not recommend all methods automatically.""",
"6. Research Quality Reviewer": """Act as a critical research reviewer. Assess question clarity, population, eligibility, design, outcome validity, exposure measurement, confounding, bias, feasibility, statistical suitability, missing data, generalizability and interpretation. Prioritize material concerns and provide actionable corrections. Do not invent flaws or require every possible analysis.""",
"7. Research Gap & Literature Review": """Create a literature-review framework for the stated topic. Identify themes, search concepts, synonyms, databases to consider, inclusion boundaries and a provisional gap statement. Clearly distinguish established knowledge from hypotheses requiring literature verification. Never invent publications, authors, citations or findings.""",
"8. Proposal Builder": """Build a structured proposal using the information supplied and previous module outputs: title, background outline, rationale, problem statement, question, aim, objectives, design, population, eligibility, variables, outcomes, data collection, statistical plan, ethics, limitations and contribution. Mark unknown items as TO BE CONFIRMED; do not fabricate references, sample size or data availability.""",
"9. Scientific Validation": """Audit the supplied previous AI output against the original prompt. Verify quotations. Separate actual errors, important omissions and additional considerations. Do not automatically claim selection/collider bias. Do not invent follow-up periods, data fields, clinical definitions or verified evidence. Do not recommend models before outcome and data structure are clear. Give concise corrections and a coverage/limitations statement."""
}

SYSTEM_PROMPT = """You are an AI-assisted quantitative health research support system.
Use clear academic English. Explain technical terms accessibly.
Never invent data, findings, references, citations or database capabilities.
Distinguish verified facts from assumptions and conditional recommendations.
Do not present simulated data as real patient data.
Do not replace a statistician, epidemiologist, clinician or supervisor.
Be concise, structured and specific to the user's research question."""

# ============================================================
# API
# ============================================================
def get_client():
    key = os.getenv("GROQ_API_KEY")
    if not key:
        try:
            key = st.secrets["GROQ_API_KEY"]
        except Exception:
            key = None
    return Groq(api_key=key) if key else None

def ask_groq(module_name, topic, aim, context):
    client = get_client()
    if client is None:
        raise RuntimeError("GROQ_API_KEY is missing. Add it in Streamlit Secrets.")
    prompt = f"""MODULE: {module_name}
RESEARCH TOPIC: {topic}
AIM: {aim}
PREVIOUS WORK / USER CONTEXT:
{context[:12000] if context else "None supplied."}

TASK:
{MODULE_PROMPTS[module_name]}

Return a useful, structured response. Keep it focused and avoid repeating the prompt."""
    result = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=2200
    )
    return result.choices[0].message.content

# ============================================================
# SESSION STATE
# ============================================================
if "module_outputs" not in st.session_state:
    st.session_state.module_outputs = {}
if "analysis_df" not in st.session_state:
    st.session_state.analysis_df = None
if "analysis_is_synthetic" not in st.session_state:
    st.session_state.analysis_is_synthetic = False

# ============================================================
# SYNTHETIC DEMO DATA
# ============================================================
def make_demo_data(n=500, seed=42):
    rng = np.random.default_rng(seed)
    sex = rng.choice(["Female", "Male"], size=n)
    age = np.clip(rng.normal(31, 13, n), 2, 75).round(1)
    comorbidity = rng.binomial(1, 0.24, n)
    followup = rng.uniform(0.5, 5.0, n).round(2)
    # Simulated pattern solely for demonstrating analysis workflow.
    p_use = np.where(sex == "Female", 0.48, 0.43) + 0.10 * comorbidity
    any_use = rng.binomial(1, np.clip(p_use, 0.05, 0.95))
    rate = np.where(sex == "Female", 1.35, 1.20) + 0.55 * comorbidity
    encounters = rng.poisson(rate * followup) * any_use
    imaging = rng.poisson(0.35 * followup) * any_use
    nephrology = rng.poisson(0.28 * followup) * any_use
    return pd.DataFrame({
        "patient_id": np.arange(1, n + 1),
        "sex": sex,
        "age_years": age,
        "baseline_comorbidity": comorbidity,
        "followup_years": followup,
        "any_kidney_use": any_use,
        "kidney_encounters": encounters,
        "kidney_imaging_count": imaging,
        "nephrology_visit_count": nephrology
    })

# ============================================================
# ANALYSIS HELPERS
# ============================================================
def pformat(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"

def run_analysis(df, sex_col, outcome_col, followup_col=None):
    work = df.copy()
    work[sex_col] = work[sex_col].astype(str)
    groups = [g for g in work[sex_col].dropna().unique()]
    if len(groups) != 2:
        raise ValueError("For this demonstration, select a grouping variable with exactly two groups.")
    a, b = groups[0], groups[1]
    x = pd.to_numeric(work.loc[work[sex_col] == a, outcome_col], errors="coerce").dropna()
    y = pd.to_numeric(work.loc[work[sex_col] == b, outcome_col], errors="coerce").dropna()
    if len(x) < 2 or len(y) < 2:
        raise ValueError("Each group needs at least two non-missing observations.")
    desc = work.groupby(sex_col)[outcome_col].agg(["count", "mean", "std", "median", "min", "max"])
    is_binary = set(work[outcome_col].dropna().unique()).issubset({0, 1})
    if is_binary:
        tab = pd.crosstab(work[sex_col], work[outcome_col])
        if tab.shape == (2, 2):
            chi2, p, dof, expected = stats.chi2_contingency(tab)
            test_name = "Chi-square test"
            if (expected < 5).any():
                _, p = stats.fisher_exact(tab)
                test_name = "Fisher's exact test (expected cell count <5)"
            test_stat = chi2
        else:
            p, test_name, test_stat = np.nan, "Not calculated", np.nan
        results = {
            "test": test_name, "statistic": test_stat, "p": p,
            "group_a": a, "group_b": b,
            "n_a": int((work[sex_col] == a).sum()),
            "n_b": int((work[sex_col] == b).sum()),
            "outcome_type": "Binary"
        }
    else:
        # Welch's t-test is used as a transparent default for a mean comparison.
        tstat, p = stats.ttest_ind(x, y, equal_var=False, nan_policy="omit")
        results = {
            "test": "Welch two-sample t-test",
            "statistic": tstat, "p": p,
            "group_a": a, "group_b": b,
            "n_a": len(x), "n_b": len(y),
            "outcome_type": "Numeric"
        }
    return desc, results

def make_plot(df, sex_col, outcome_col, plot_type):
    fig, ax = plt.subplots(figsize=(7, 4))
    data = df[[sex_col, outcome_col]].dropna().copy()
    data[outcome_col] = pd.to_numeric(data[outcome_col], errors="coerce")
    data = data.dropna()
    if plot_type == "Box plot":
        data.boxplot(column=outcome_col, by=sex_col, ax=ax)
        fig.suptitle("")
        ax.set_title(f"{outcome_col} by {sex_col}")
        ax.set_xlabel(sex_col)
    elif plot_type == "Histogram":
        for label, group in data.groupby(sex_col):
            ax.hist(group[outcome_col], bins=20, alpha=0.5, label=str(label))
        ax.legend()
        ax.set_title(f"Distribution of {outcome_col}")
    elif plot_type == "Bar chart (mean)":
        means = data.groupby(sex_col)[outcome_col].mean()
        ax.bar(means.index.astype(str), means.values)
        ax.set_title(f"Mean {outcome_col} by {sex_col}")
        ax.set_ylabel("Mean")
    else:
        proportions = data.groupby(sex_col)[outcome_col].mean() * 100
        ax.bar(proportions.index.astype(str), proportions.values)
        ax.set_title(f"Percentage with {outcome_col}=1")
        ax.set_ylabel("Percent")
    fig.tight_layout()
    return fig

# ============================================================
# INTERFACE
# ============================================================
st.title("AI-Powered Quantitative Research Assistant")
st.caption("Step-by-step research planning, statistical analysis and visualization prototype")
st.info("Research demonstration only. AI guidance and statistical outputs require expert review.")

with st.sidebar:
    st.title("Research Workflow")
    st.caption("Complete modules in sequence; outputs remain available during this session.")
    selected = st.radio(
        "Choose a step",
        [m[0] for m in MODULES],
        index=0
    )
    st.markdown("---")
    st.caption("Demonstration topic")
    st.write(TOPIC_DEFAULT)

topic = st.text_input("Research Topic", TOPIC_DEFAULT)
aim = st.text_area("Research Aim / Objective", AIM_DEFAULT, height=85)

# ============================================================
# MODULES 1-9: AI-ASSISTED RESEARCH WORKFLOW
# ============================================================
if selected != "10. Statistical Analysis & Visualization":
    st.header(selected)
    description = dict(MODULES)[selected]
    st.write(description)

    if selected == "9. Scientific Validation":
        original_prompt = st.text_area("Original user prompt", height=150)
        previous_output = st.text_area("Previous AI output to audit", height=250)
        extra_context = st.text_area("Optional protocol / SAP / additional context", height=100)
        context = (
            f"ORIGINAL PROMPT:\n{original_prompt}\n\n"
            f"PREVIOUS AI OUTPUT:\n{previous_output}\n\n"
            f"ADDITIONAL CONTEXT:\n{extra_context}"
        )
        can_run = bool(previous_output.strip())
    else:
        previous = []
        for name, _ in MODULES:
            if name == selected:
                break
            if name in st.session_state.module_outputs:
                previous.append(f"\n\n--- {name} ---\n{st.session_state.module_outputs[name]}")
        earlier_context = "".join(previous)
        user_context = st.text_area(
            "Your notes / protocol / data dictionary / earlier decisions",
            height=130,
            placeholder="Add relevant information. Earlier completed module outputs will be included automatically."
        )
        context = (earlier_context + "\n\nUSER NOTES:\n" + user_context).strip()
        can_run = True

    if st.button("Generate this research step", type="primary", disabled=not can_run):
        with st.spinner("Preparing your research step..."):
            try:
                output = ask_groq(selected, topic, aim, context)
                st.session_state.module_outputs[selected] = output
                st.success("Step completed. Your output is saved in this session.")
            except Exception as e:
                st.error(f"AI request failed: {e}")

    if selected in st.session_state.module_outputs:
        st.markdown("---")
        st.subheader("Research Assistant Output")
        st.markdown(st.session_state.module_outputs[selected])
        st.download_button(
            "Download this module",
            st.session_state.module_outputs[selected],
            file_name=f"{selected.split('. ',1)[0]}_research_output.md",
            mime="text/markdown"
        )

    completed = [m[0] for m in MODULES if m[0] in st.session_state.module_outputs]
    st.caption(f"Completed AI planning/review steps: {len(completed)} of 9")

# ============================================================
# MODULE 10: REAL DATA CALCULATIONS AND GRAPHS
# ============================================================
else:
    st.header("10. Statistical Analysis & Visualization")
    st.write("Upload a CSV file or generate a clearly labelled synthetic TSC demonstration dataset.")
    st.warning("Synthetic data are simulated for software demonstration only. They are not real TSC patient records and must never be presented as clinical evidence.")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Generate synthetic TSC demo dataset", type="primary"):
            st.session_state.analysis_df = make_demo_data()
            st.session_state.analysis_is_synthetic = True
    with c2:
        uploaded = st.file_uploader("Or upload your own CSV dataset", type=["csv"])
        if uploaded is not None:
            try:
                st.session_state.analysis_df = pd.read_csv(uploaded)
                st.session_state.analysis_is_synthetic = False
            except Exception as e:
                st.error(f"Could not read CSV: {e}")

    df = st.session_state.analysis_df
    if df is not None:
        if st.session_state.analysis_is_synthetic:
            st.info("ACTIVE DATA: Synthetic simulated dataset (n={:,}).".format(len(df)))
        else:
            st.success("ACTIVE DATA: User-uploaded CSV. Check data definitions before interpreting results.")

        st.subheader("Step 1 — Dataset overview")
        m1, m2, m3 = st.columns(3)
        m1.metric("Rows / records", f"{len(df):,}")
        m2.metric("Variables", f"{df.shape[1]:,}")
        m3.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")
        st.dataframe(df.head(10), use_container_width=True)

        overview = pd.DataFrame({
            "Variable": df.columns,
            "Data type": [str(x) for x in df.dtypes],
            "Missing n": [int(df[c].isna().sum()) for c in df.columns],
            "Missing %": [round(df[c].isna().mean()*100, 1) for c in df.columns],
            "Unique values": [int(df[c].nunique(dropna=True)) for c in df.columns]
        })
        st.markdown("**Variable and missing-data profile**")
        st.dataframe(overview, use_container_width=True)
        st.download_button("Download data profile CSV", overview.to_csv(index=False), "data_profile.csv", "text/csv")

        st.markdown("---")
        st.subheader("Step 2 — Choose comparison")
        categorical = [c for c in df.columns if df[c].dtype == "object" or df[c].nunique(dropna=True) <= 10]
        default_group = "sex" if "sex" in df.columns else (categorical[0] if categorical else df.columns[0])
        group_col = st.selectbox("Grouping variable (e.g., sex)", df.columns.tolist(), index=df.columns.get_loc(default_group))
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if not numeric_cols:
            st.error("No numeric variables found in the dataset.")
            st.stop()
        default_outcome = "kidney_encounters" if "kidney_encounters" in numeric_cols else numeric_cols[0]
        outcome_col = st.selectbox("Numeric outcome", numeric_cols, index=numeric_cols.index(default_outcome))
        plot_type = st.selectbox("Graph type", ["Box plot", "Bar chart (mean)", "Histogram"])
        if st.button("Run descriptive and group comparison", type="primary"):
            try:
                desc, result = run_analysis(df, group_col, outcome_col)
                st.session_state["latest_desc"] = desc
                st.session_state["latest_test"] = result
            except Exception as e:
                st.error(f"Analysis could not be completed: {e}")

        if "latest_desc" in st.session_state:
            st.markdown("---")
            st.subheader("Step 3 — Descriptive statistics")
            st.dataframe(st.session_state.latest_desc, use_container_width=True)
            st.download_button("Download descriptive statistics", st.session_state.latest_desc.to_csv().encode(), "descriptive_statistics.csv", "text/csv")

            st.subheader("Step 4 — Statistical comparison")
            r = st.session_state.latest_test
            st.write(f"**Test used:** {r['test']}")
            st.write(f"**Test statistic:** {r['statistic']:.3f}" if pd.notna(r["statistic"]) else "**Test statistic:** Not available")
            st.write(f"**P-value:** {pformat(r['p'])}" if pd.notna(r["p"]) else "**P-value:** Not calculated")
            st.caption("A p-value does not measure clinical importance. Interpret it alongside effect size, confidence intervals, study design and data quality.")
            if r["outcome_type"] == "Binary":
                st.write("Binary outcome comparison. Review group denominators and cell counts; Fisher's exact test is used when expected cell counts are small.")
            else:
                st.write("Numeric outcome comparison using Welch's t-test. This compares group means without assuming equal variances. For highly skewed counts, consider a count-data model after checking assumptions.")

            st.subheader("Step 5 — Visualization")
            fig = make_plot(df, group_col, outcome_col, plot_type)
            st.pyplot(fig)
            image_buffer = io.BytesIO()
            fig.savefig(image_buffer, format="png", dpi=160, bbox_inches="tight")
            st.download_button("Download graph (PNG)", image_buffer.getvalue(), "group_comparison.png", "image/png")
            plt.close(fig)

            st.subheader("Step 6 — Optional count-rate summary")
            if "followup_years" in df.columns and "kidney_encounters" in df.columns and "sex" in df.columns:
                rate_table = df.groupby("sex").agg(
                    patients=("patient_id", "count"),
                    encounters=("kidney_encounters", "sum"),
                    person_years=("followup_years", "sum")
                )
                rate_table["encounters_per_person_year"] = rate_table["encounters"] / rate_table["person_years"]
                st.dataframe(rate_table, use_container_width=True)
                st.caption("Crude descriptive rates only. They are not adjusted comparisons and require appropriate interpretation of eligibility and follow-up.")
                st.download_button("Download crude rate summary", rate_table.to_csv().encode(), "crude_rates.csv", "text/csv")

            st.subheader("Step 7 — Export analysis dataset")
            st.download_button("Download active dataset CSV", df.to_csv(index=False).encode(), "analysis_dataset.csv", "text/csv")

    else:
        st.info("Generate the synthetic demo dataset or upload a CSV to begin.")
        st.markdown("**Expected example columns:** `sex`, `age_years`, `followup_years`, `any_kidney_use`, `kidney_encounters`. Your own dataset can use different names; select the relevant columns after upload.")

st.markdown("---")
st.caption("AI-assisted research prototype | Statistical calculations are performed on the uploaded or simulated dataset. Expert review is required before research use.")
