
import os
import io
import warnings

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from groq import Groq

warnings.filterwarnings("ignore")

# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI-Assisted Quantitative Research Assistant",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 AI-Assisted Quantitative Research Assistant")

st.markdown(
    """
    A step-by-step research development and quantitative
    analysis tool for health and pharmacy research.

    **Research planning → Methodology → Variables → Analysis plan
    → Statistical analysis → Interpretation → Research proposal**
    """
)

st.caption(
    "Educational research-support prototype. "
    "AI-generated suggestions and statistical outputs require "
    "researcher and supervisor review."
)

# ============================================================
# 2. DEFAULT RESEARCH TOPIC
# ============================================================

DEFAULT_TOPIC = (
    "Sex-Based Differences in Kidney-Related Healthcare Utilization, "
    "Monitoring, and Clinical Outcomes Among Patients with "
    "Tuberous Sclerosis Complex (TSC)"
)

DEFAULT_OBJECTIVE = (
    "To explore whether healthcare utilization, kidney-related "
    "monitoring, and selected clinical outcomes differ by sex "
    "among patients with Tuberous Sclerosis Complex."
)

# ============================================================
# 3. SESSION STATE
# ============================================================

defaults = {
    "research_outputs": {},
    "research_topic": DEFAULT_TOPIC,
    "research_objective": DEFAULT_OBJECTIVE,
    "analysis_results": {},
    "analysis_data": None,
    "data_source": "Synthetic demonstration dataset"
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# ============================================================
# 4. GROQ CONNECTION
# ============================================================

MODEL_NAME = "openai/gpt-oss-20b"


def get_groq_client():

    try:
        api_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        return None

    try:
        return Groq(api_key=api_key)
    except Exception:
        return None


SYSTEM_PROMPT = """
You are an AI-assisted quantitative health research assistant.

Your role is to support quantitative health and pharmacy research,
especially pharmacy practice, healthcare utilization, health
services, clinical outcomes, and population health.

Instructions:

1. Explain research and statistical concepts in accessible
   academic language.
2. Explain the reasoning behind methodological decisions.
3. Never invent research findings, references, DOIs, or citations.
4. Distinguish observed data, assumptions, and suggestions.
5. Match statistical methods to the research question and
   outcome variable.
6. Consider confounding, selection bias, information bias,
   missing data, sample size, and limitations.
7. Do not automatically recommend advanced statistical methods.
8. Explain Poisson and Negative Binomial regression for
   count outcomes where appropriate.
9. Explain binary outcome methods where appropriate.
10. Never interpret synthetic data as real patient evidence.
11. Explain technical terminology briefly.
12. Identify points requiring supervisor or statistician review.
13. Use clear headings and structured explanations.
"""


def ask_ai(prompt, max_tokens=1800):

    client = get_groq_client()

    if client is None:
        return (
            "Groq API connection unavailable. Please check "
            "GROQ_API_KEY in Streamlit Secrets."
        )

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_completion_tokens=max_tokens
        )

        return response.choices[0].message.content or (
            "The AI returned an empty response."
        )

    except Exception as error:
        return f"AI request failed: {str(error)}"


# ============================================================
# 5. RESEARCH CONTEXT
# ============================================================

st.header("1. Research Foundation")

st.write(
    "Define the research topic and overall objective. "
    "These will guide the subsequent research-development stages."
)

topic = st.text_area(
    "Research topic",
    value=st.session_state.research_topic,
    height=90,
    key="research_topic_input"
)

objective = st.text_area(
    "Overall research objective",
    value=st.session_state.research_objective,
    height=90,
    key="research_objective_input"
)

st.session_state.research_topic = topic
st.session_state.research_objective = objective

st.divider()


# ============================================================
# 6. RESEARCH WORKFLOW SUPPORT FUNCTIONS
# ============================================================

def get_previous_work():

    collected = []

    stage_order = [
        "Research Questions and Objectives",
        "Study Design and Methodology",
        "Variables and Data Collection",
        "Statistical Analysis Plan",
        "Statistical Analysis Interpretation",
        "Interpretation, Bias and Research Quality",
        "Research Proposal Development"
    ]

    for stage_name in stage_order:

        output = st.session_state.research_outputs.get(stage_name)

        if output:
            collected.append(
                f"\n{stage_name}\n{output}"
            )

    return "\n".join(collected)


def generate_stage(stage_number, title, description, instructions):

    stage_key = f"stage_{stage_number}"

    st.subheader(f"Step {stage_number}. {title}")
    st.write(description)

    user_input = st.text_area(
        "Additional information or instructions (optional)",
        key=f"{stage_key}_input",
        height=85,
        placeholder="Add research details or questions for this stage..."
    )

    if st.button(
        f"Generate {title}",
        key=f"{stage_key}_button",
        type="primary"
    ):

        if not topic.strip() or not objective.strip():

            st.error(
                "Please provide the research topic and objective first."
            )

        else:

            prompt = f"""
RESEARCH TOPIC:
{topic}

OVERALL OBJECTIVE:
{objective}

CURRENT STAGE:
{title}

TASK:
{instructions}

ADDITIONAL RESEARCHER INPUT:
{user_input}

PREVIOUS RESEARCH WORK:
{get_previous_work()[:4500]}

Develop this stage in a structured way.

Explain the reasoning behind recommendations.
Define technical terms briefly.
Identify assumptions and limitations.
Do not invent evidence or citations.
"""

            with st.spinner(f"Developing Step {stage_number}..."):
                result = ask_ai(prompt)

            st.session_state.research_outputs[title] = result

    if title in st.session_state.research_outputs:

        st.markdown("#### AI Research Output")
        st.markdown(st.session_state.research_outputs[title])

        st.download_button(
            f"Download {title}",
            data=st.session_state.research_outputs[title],
            file_name=f"research_step_{stage_number}.txt",
            mime="text/plain",
            key=f"{stage_key}_download"
        )


# ============================================================
# 7. STEP 1 — RESEARCH QUESTIONS AND OBJECTIVES
# ============================================================

st.header("2. Research Development Workflow")

st.markdown(
    "Develop the research in sequence. Each stage builds on "
    "the decisions made in the preceding stages."
)

generate_stage(
    1,
    "Research Questions and Objectives",
    "Develop focused and answerable research questions from "
    "the overall research objective.",
    """
1. Identify the central research problem.
2. Develop one primary research question.
3. Suggest relevant secondary research questions.
4. Refine the primary and secondary objectives.
5. Explain how each question can be answered quantitatively.
6. Check whether the questions are specific and measurable.

Explain the difference between a research question,
research aim, and research objective.
"""
)

st.divider()


# ============================================================
# 8. STEP 2 — STUDY DESIGN AND METHODOLOGY
# ============================================================

generate_stage(
    2,
    "Study Design and Methodology",
    "Determine how the research questions can be investigated.",
    """
1. Recommend a suitable quantitative study design.
2. Explain why this design fits the research questions.
3. Define the target population.
4. Suggest inclusion and exclusion criteria.
5. Discuss possible study settings and data sources.
6. Explain possible sampling approaches.
7. Discuss sample size considerations.
8. Identify feasibility and ethical considerations.

Compare alternative study designs briefly where relevant.
Do not claim that a particular dataset or data source is available.
"""
)

st.divider()


# ============================================================
# 9. STEP 3 — VARIABLES AND DATA COLLECTION
# ============================================================

generate_stage(
    3,
    "Variables and Data Collection",
    "Translate the research questions into measurable variables.",
    """
1. Identify the primary outcome variable.
2. Identify relevant secondary outcome variables.
3. Identify the main exposure or grouping variable.
4. Identify potential confounding variables.
5. Suggest operational definitions.
6. Identify measurement scales and data types.
7. Explain how variables relate to the research questions.
8. Present a structured variable table.

For each variable, explain what it measures and why it is relevant.
Do not assume that all suggested variables will be available
in the final dataset.
"""
)

st.divider()


# ============================================================
# 10. STEP 4 — STATISTICAL ANALYSIS PLAN
# ============================================================

generate_stage(
    4,
    "Statistical Analysis Plan",
    "Match statistical methods to research questions and outcome types.",
    """
Prepare a structured statistical analysis plan.

1. Explain the descriptive analysis required.
2. Match each research question with an appropriate method.
3. Explain methods for continuous outcomes.
4. Explain methods for binary outcomes.
5. Explain methods for count outcomes.
6. Discuss regression and adjustment for confounders.
7. Explain assumptions and diagnostic checks.
8. Discuss missing data.
9. Explain how results could be presented.

Where relevant, discuss:
- Welch's t-test
- Chi-square test
- Fisher's exact test
- Logistic regression
- Poisson regression
- Negative Binomial regression

Do not recommend every method automatically.
Explain which methods are relevant and what information
is needed before choosing them.
"""
)

st.divider()


# ============================================================
# 11. DATASET AND STATISTICAL ANALYSIS
# ============================================================

st.header("3. Dataset and Statistical Analysis")

st.write(
    "Use a synthetic demonstration dataset or upload a CSV file. "
    "The analysis workspace provides descriptive statistics, "
    "group comparisons, regression models, and visualizations."
)

st.warning(
    "The built-in dataset is synthetic and is only for demonstrating "
    "the analysis workflow. Its results are not clinical evidence."
)


# ============================================================
# 12. SYNTHETIC DATA GENERATION
# ============================================================

@st.cache_data
def generate_synthetic_data(n=500, seed=42):

    rng = np.random.default_rng(seed)

    sex = rng.choice(
        ["Female", "Male"],
        size=n,
        p=[0.52, 0.48]
    )

    age = rng.integers(5, 70, size=n)

    disease_duration = np.maximum(
        0,
        rng.normal(15, 8, n)
    ).round(1)

    male_effect = (sex == "Male").astype(int)

    kidney_monitoring_probability = (
        0.55 + 0.08 * male_effect
    )

    kidney_monitoring_probability = np.clip(
        kidney_monitoring_probability,
        0.05,
        0.95
    )

    kidney_monitoring = rng.binomial(
        1,
        kidney_monitoring_probability
    )

    specialist_visits = rng.poisson(
        np.exp(
            1.0
            + 0.012 * age
            + 0.018 * disease_duration
            + 0.10 * male_effect
        )
    )

    imaging_count = rng.poisson(
        np.exp(
            0.35
            + 0.008 * age
            + 0.025 * disease_duration
            + 0.12 * male_effect
        )
    )

    annual_healthcare_visits = rng.poisson(
        np.exp(
            1.15
            + 0.009 * age
            + 0.025 * disease_duration
            + 0.08 * male_effect
        )
    )

    hospitalization_probability = 1 / (
        1 + np.exp(
            -(
                -2.0
                + 0.025 * age
                + 0.035 * disease_duration
                + 0.15 * male_effect
            )
        )
    )

    hospitalization = rng.binomial(
        1,
        hospitalization_probability
    )

    complication_probability = 1 / (
        1 + np.exp(
            -(
                -2.2
                + 0.018 * age
                + 0.045 * disease_duration
                + 0.12 * male_effect
            )
        )
    )

    kidney_complication = rng.binomial(
        1,
        complication_probability
    )

    egfr = (
        105
        - 0.35 * age
        - 0.25 * disease_duration
        - 5 * kidney_complication
        + rng.normal(0, 12, n)
    )

    egfr = np.clip(egfr, 15, 150).round(1)

    df = pd.DataFrame({
        "patient_id": range(1, n + 1),
        "sex": sex,
        "age": age,
        "disease_duration": disease_duration,
        "kidney_monitoring": kidney_monitoring,
        "specialist_visits": specialist_visits,
        "imaging_count": imaging_count,
        "hospitalization": hospitalization,
        "kidney_complication": kidney_complication,
        "eGFR": egfr,
        "annual_healthcare_visits": annual_healthcare_visits
    })

    return df


# ============================================================
# 13. DATASET LOADING
# ============================================================

data_option = st.radio(
    "Select dataset source",
    [
        "Synthetic demonstration dataset",
        "Upload my own CSV dataset"
    ],
    horizontal=True,
    key="dataset_source_option"
)

uploaded_file = None

if data_option == "Synthetic demonstration dataset":

    sample_size = st.slider(
        "Number of synthetic records",
        min_value=100,
        max_value=2000,
        value=500,
        step=100
    )

    data = generate_synthetic_data(
        n=sample_size,
        seed=42
    )

    st.session_state.data_source = (
        "Synthetic demonstration dataset"
    )

else:

    uploaded_file = st.file_uploader(
        "Upload CSV dataset",
        type=["csv"],
        help=(
            "The CSV should contain the study variables "
            "required for the selected analysis."
        )
    )

    if uploaded_file is not None:

        try:
            data = pd.read_csv(uploaded_file)
            st.session_state.data_source = "Uploaded CSV dataset"

        except Exception as error:

            st.error(f"Could not read CSV file: {error}")
            data = None

    else:
        data = None


# ============================================================
# 14. DATA VALIDATION
# ============================================================

REQUIRED_COLUMNS = [
    "sex",
    "age",
    "disease_duration",
    "kidney_monitoring",
    "specialist_visits",
    "imaging_count",
    "hospitalization",
    "kidney_complication",
    "eGFR",
    "annual_healthcare_visits"
]

COUNT_VARIABLES = [
    "specialist_visits",
    "imaging_count",
    "annual_healthcare_visits"
]

BINARY_VARIABLES = [
    "kidney_monitoring",
    "hospitalization",
    "kidney_complication"
]

CONTINUOUS_VARIABLES = [
    "age",
    "disease_duration",
    "eGFR"
]


def normalize_binary(series):

    mapping = {
        "yes": 1,
        "no": 0,
        "true": 1,
        "false": 0,
        "1": 1,
        "0": 0,
        "present": 1,
        "absent": 0
    }

    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")

    return series.astype(str).str.strip().str.lower().map(mapping)


def validate_dataset(df):

    issues = []

    missing_columns = [
        col for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing_columns:
        issues.append(
            "Missing required columns: "
            + ", ".join(missing_columns)
        )

        return None, issues

    df = df.copy()

    df["sex"] = (
        df["sex"]
        .astype(str)
        .str.strip()
        .str.title()
    )

    allowed_sex = ["Female", "Male"]

    unexpected_sex = (
        set(df["sex"].dropna().unique())
        - set(allowed_sex)
    )

    if unexpected_sex:
        issues.append(
            "Sex must use Female and Male categories. "
            "Unexpected values: "
            + ", ".join(map(str, unexpected_sex))
        )

    numeric_columns = (
        CONTINUOUS_VARIABLES
        + COUNT_VARIABLES
        + BINARY_VARIABLES
    )

    for col in numeric_columns:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    for col in BINARY_VARIABLES:
        df[col] = normalize_binary(df[col])

        invalid = (
            df[col].notna()
            & ~df[col].isin([0, 1])
        )

        if invalid.any():
            issues.append(
                f"{col} must contain only 0 and 1 "
                "(or Yes/No)."
            )

    for col in COUNT_VARIABLES:

        invalid = (
            df[col].notna()
            & (
                (df[col] < 0)
                | (df[col] % 1 != 0)
            )
        )

        if invalid.any():
            issues.append(
                f"{col} must contain non-negative whole numbers."
            )

    for col in CONTINUOUS_VARIABLES:

        if (df[col].dropna() < 0).any():
            issues.append(
                f"{col} contains negative values."
            )

    if df.empty:
        issues.append("Dataset is empty.")

    if len(df) < 10:
        issues.append(
            "Very small dataset: statistical results may be unstable."
        )

    return df, issues


validated_data = None

if data is not None:

    validated_data, validation_issues = validate_dataset(data)

    st.subheader("Dataset Validation")

    if validation_issues:

        for issue in validation_issues:
            st.error(issue)

    else:

        st.success(
            "Dataset structure and required variables validated."
        )

        st.session_state.analysis_data = validated_data

        st.write(
            f"Records: **{len(validated_data):,}** | "
            f"Variables: **{len(validated_data.columns)}**"
        )

        with st.expander("Preview dataset"):
            st.dataframe(
                validated_data.head(20),
                use_container_width=True
            )

        with st.expander("Dataset information and missing values"):

            info_df = pd.DataFrame({
                "Variable": validated_data.columns,
                "Data type": [
                    str(validated_data[col].dtype)
                    for col in validated_data.columns
                ],
                "Missing values": [
                    int(validated_data[col].isna().sum())
                    for col in validated_data.columns
                ],
                "Missing (%)": [
                    round(
                        validated_data[col].isna().mean() * 100,
                        2
                    )
                    for col in validated_data.columns
                ]
            })

            st.dataframe(
                info_df,
                use_container_width=True
            )

else:
    st.info("Select a dataset source to begin statistical analysis.")


# ============================================================
# 15. STATISTICAL ANALYSIS FUNCTIONS
# ============================================================

def get_analysis_data():

    df = st.session_state.analysis_data

    if df is None:
        return None

    return df.copy()


def safe_group_data(df, variable):

    subset = df[["sex", variable]].dropna()

    female = subset.loc[
        subset["sex"] == "Female",
        variable
    ]

    male = subset.loc[
        subset["sex"] == "Male",
        variable
    ]

    return female, male


def create_group_plot(df, variable, title):

    plot_data = df[["sex", variable]].dropna()

    if plot_data.empty:
        return None

    fig, ax = plt.subplots(figsize=(7, 4))

    groups = [
        plot_data.loc[
            plot_data["sex"] == "Female",
            variable
        ],
        plot_data.loc[
            plot_data["sex"] == "Male",
            variable
        ]
    ]

   ax.boxplot(
    groups,
    tick_labels=group_names,
    patch_artist=True
)

    ax.set_title(title)
    ax.set_ylabel(variable)
    ax.set_xlabel("Sex")

    fig.tight_layout()

    return fig


def create_distribution_plot(df, variable, title):

    values = df[variable].dropna()

    if values.empty:
        return None

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.hist(
        values,
        bins=20,
        edgecolor="black",
        alpha=0.8
    )

    ax.set_title(title)
    ax.set_xlabel(variable)
    ax.set_ylabel("Frequency")

    fig.tight_layout()

    return fig


def descriptive_statistics(df):

    numeric_columns = (
        CONTINUOUS_VARIABLES
        + COUNT_VARIABLES
    )

    overall = df[numeric_columns].describe().T

    by_sex = df.groupby("sex")[numeric_columns].agg(
        ["count", "mean", "std", "median"]
    )

    return overall, by_sex


def welch_test(df, variable):

    female, male = safe_group_data(df, variable)

    if len(female) < 2 or len(male) < 2:
        raise ValueError(
            "At least two observations in each group are required."
        )

    result = stats.ttest_ind(
        female,
        male,
        equal_var=False,
        nan_policy="omit"
    )

    mean_difference = male.mean() - female.mean()

    se = np.sqrt(
        male.var(ddof=1) / len(male)
        + female.var(ddof=1) / len(female)
    )

    df_num = (
        male.var(ddof=1) / len(male)
        + female.var(ddof=1) / len(female)
    ) ** 2

    df_den = (
        (male.var(ddof=1) / len(male)) ** 2
        / (len(male) - 1)
        +
        (female.var(ddof=1) / len(female)) ** 2
        / (len(female) - 1)
    )

    degrees_freedom = df_num / df_den

    critical_value = stats.t.ppf(
        0.975,
        degrees_freedom
    )

    ci_low = mean_difference - critical_value * se
    ci_high = mean_difference + critical_value * se

    return {
        "Variable": variable,
        "Female mean": female.mean(),
        "Male mean": male.mean(),
        "Mean difference (Male - Female)": mean_difference,
        "t statistic": result.statistic,
        "Degrees of freedom": degrees_freedom,
        "p value": result.pvalue,
        "95% CI lower": ci_low,
        "95% CI upper": ci_high
    }


def binary_comparison(df, variable):

    subset = df[["sex", variable]].dropna()

    table = pd.crosstab(
        subset["sex"],
        subset[variable]
    )

    table = table.reindex(
        index=["Female", "Male"],
        columns=[0, 1],
        fill_value=0
    )

    if table.shape != (2, 2):
        raise ValueError(
            "A valid 2 x 2 table could not be created."
        )

    chi2, p_chi, dof, expected = stats.chi2_contingency(
        table,
        correction=False
    )

    if (expected < 5).any():

        odds_ratio, p_value = stats.fisher_exact(table)
        test_used = "Fisher's exact test"

    else:

        odds_ratio = np.nan
        p_value = p_chi
        test_used = "Chi-square test"

    return {
        "table": table,
        "test": test_used,
        "chi_square": chi2,
        "p_value": p_value,
        "odds_ratio": odds_ratio,
        "expected": expected
    }


def poisson_regression(df, outcome):

    model_data = df[
        [
            outcome,
            "sex",
            "age",
            "disease_duration"
        ]
    ].dropna().copy()

    if len(model_data) < 20:
        raise ValueError(
            "At least 20 complete observations are recommended "
            "for this demonstration model."
        )

    if (model_data[outcome] < 0).any():
        raise ValueError("Count outcome cannot contain negative values.")

    if model_data[outcome].sum() == 0:
        raise ValueError("Count outcome contains only zero values.")

    formula = (
        f"{outcome} ~ C(sex) + age + disease_duration"
    )

    model = smf.glm(
        formula=formula,
        data=model_data,
        family=sm.families.Poisson()
    ).fit()

    result_table = pd.DataFrame({
        "Coefficient": model.params,
        "Standard Error": model.bse,
        "z statistic": model.tvalues,
        "p value": model.pvalues,
        "IRR": np.exp(model.params),
        "95% CI lower": np.exp(model.conf_int()[0]),
        "95% CI upper": np.exp(model.conf_int()[1])
    })

    dispersion = (
        model.pearson_chi2 / model.df_resid
        if model.df_resid > 0
        else np.nan
    )

    return model, result_table, dispersion


def negative_binomial_regression(df, outcome):

    model_data = df[
        [
            outcome,
            "sex",
            "age",
            "disease_duration"
        ]
    ].dropna().copy()

    if len(model_data) < 20:
        raise ValueError(
            "At least 20 complete observations are recommended "
            "for this demonstration model."
        )

    if (model_data[outcome] < 0).any():
        raise ValueError("Count outcome cannot contain negative values.")

    if model_data[outcome].sum() == 0:
        raise ValueError("Count outcome contains only zero values.")

    formula = (
        f"{outcome} ~ C(sex) + age + disease_duration"
    )

    model = smf.negativebinomial(
        formula=formula,
        data=model_data
    ).fit(
        disp=False,
        maxiter=200
    )

    result_table = pd.DataFrame({
        "Coefficient": model.params,
        "Standard Error": model.bse,
        "z statistic": model.tvalues,
        "p value": model.pvalues,
        "IRR": np.exp(model.params),
        "95% CI lower": np.exp(model.conf_int()[0]),
        "95% CI upper": np.exp(model.conf_int()[1])
    })

    return model, result_table


# ============================================================
# 16. DESCRIPTIVE STATISTICS
# ============================================================

df = get_analysis_data()

if df is not None:

    st.subheader("A. Descriptive Statistics")

    st.write(
        "Summarize the dataset before conducting comparisons "
        "or regression analyses."
    )

    if st.button(
        "Calculate Descriptive Statistics",
        key="descriptive_button"
    ):

        try:

            overall, by_sex = descriptive_statistics(df)

            st.session_state.analysis_results[
                "descriptive_overall"
            ] = overall

            st.session_state.analysis_results[
                "descriptive_by_sex"
            ] = by_sex

        except Exception as error:
            st.error(f"Descriptive analysis failed: {error}")

    if "descriptive_overall" in st.session_state.analysis_results:

        st.markdown("**Overall descriptive statistics**")

        st.dataframe(
            st.session_state.analysis_results[
                "descriptive_overall"
            ].round(3),
            use_container_width=True
        )

        st.markdown("**Descriptive statistics by sex**")

        st.dataframe(
            st.session_state.analysis_results[
                "descriptive_by_sex"
            ].round(3),
            use_container_width=True
        )

    st.divider()


    # ========================================================
    # 17. WELCH'S T-TEST
    # ========================================================

    st.subheader("B. Continuous Outcome Comparison")

    st.write(
        "**Welch's two-sample t-test** compares the mean of a "
        "continuous variable between two independent groups. "
        "It does not assume equal group variances."
    )

    continuous_choice = st.selectbox(
        "Select continuous outcome",
        CONTINUOUS_VARIABLES,
        key="welch_variable"
    )

    if st.button(
        "Run Welch's t-test",
        key="welch_button"
    ):

        try:

            result = welch_test(
                df,
                continuous_choice
            )

            st.session_state.analysis_results[
                "welch"
            ] = result

        except Exception as error:
            st.error(f"Welch's t-test failed: {error}")

    if "welch" in st.session_state.analysis_results:

        result = st.session_state.analysis_results["welch"]

        st.markdown("**Welch's t-test results**")

        st.dataframe(
            pd.DataFrame([result]).round(4),
            use_container_width=True
        )

        if result["p value"] < 0.05:
            st.info(
                "The observed group difference is statistically "
                "significant at the 0.05 level."
            )
        else:
            st.info(
                "The observed group difference is not statistically "
                "significant at the 0.05 level."
            )

        st.caption(
            "Interpret the mean difference and confidence interval "
            "alongside the p-value."
        )

    st.divider()


    # ========================================================
    # 18. BINARY OUTCOME COMPARISON
    # ========================================================

    st.subheader("C. Binary Outcome Comparison")

    st.write(
        "Compare the proportion of participants with an outcome "
        "between Female and Male groups."
    )

    binary_choice = st.selectbox(
        "Select binary outcome",
        BINARY_VARIABLES,
        key="binary_variable"
    )

    if st.button(
        "Run Binary Outcome Comparison",
        key="binary_button"
    ):

        try:

            result = binary_comparison(
                df,
                binary_choice
            )

            st.session_state.analysis_results[
                "binary"
            ] = result

            st.session_state.analysis_results[
                "binary_variable"
            ] = binary_choice

        except Exception as error:
            st.error(f"Binary comparison failed: {error}")

    if "binary" in st.session_state.analysis_results:

        result = st.session_state.analysis_results["binary"]

        st.markdown("**Contingency table**")

        st.dataframe(
            result["table"],
            use_container_width=True
        )

        col1, col2, col3 = st.columns(3)

        col1.metric("Test used", result["test"])
        col2.metric(
            "p-value",
            f"{result['p_value']:.4f}"
        )

        if np.isfinite(result["odds_ratio"]):
            col3.metric(
                "Odds ratio",
                f"{result['odds_ratio']:.3f}"
            )
        else:
            col3.metric("Odds ratio", "Not calculated")

        st.caption(
            "For Fisher's exact test, the odds ratio is calculated "
            "from the 2 x 2 contingency table."
        )

    st.divider()


    # ========================================================
    # 19. POISSON REGRESSION
    # ========================================================

    st.subheader("D. Poisson Regression")

    st.write(
        "Poisson regression is used for count outcomes. "
        "It models the expected count as a function of predictors. "
        "The exponentiated coefficients are incidence rate ratios "
        "(IRRs)."
    )

    poisson_choice = st.selectbox(
        "Select count outcome for Poisson regression",
        COUNT_VARIABLES,
        key="poisson_variable"
    )

    if st.button(
        "Run Poisson Regression",
        key="poisson_button"
    ):

        try:

            model, result_table, dispersion = poisson_regression(
                df,
                poisson_choice
            )

            st.session_state.analysis_results[
                "poisson_model"
            ] = model

            st.session_state.analysis_results[
                "poisson_table"
            ] = result_table

            st.session_state.analysis_results[
                "poisson_dispersion"
            ] = dispersion

            st.session_state.analysis_results[
                "poisson_outcome"
            ] = poisson_choice

        except Exception as error:
            st.error(f"Poisson regression failed: {error}")

    if "poisson_table" in st.session_state.analysis_results:

        st.markdown("**Poisson regression results**")

        st.dataframe(
            st.session_state.analysis_results[
                "poisson_table"
            ].round(4),
            use_container_width=True
        )

        dispersion = st.session_state.analysis_results[
            "poisson_dispersion"
        ]

        st.metric(
            "Pearson dispersion statistic",
            f"{dispersion:.3f}"
            if np.isfinite(dispersion)
            else "Not available"
        )

        st.caption(
            "A dispersion statistic substantially greater than 1 "
            "may indicate overdispersion. This is a diagnostic clue, "
            "not a decision rule by itself."
        )

        with st.expander("Poisson model summary"):
            st.text(
                st.session_state.analysis_results[
                    "poisson_model"
                ].summary().as_text()
            )

    st.divider()


    # ========================================================
    # 20. NEGATIVE BINOMIAL REGRESSION
    # ========================================================

    st.subheader("E. Negative Binomial Regression")

    st.write(
        "Negative Binomial regression is another model for count "
        "outcomes. It allows for overdispersion, where the variance "
        "is greater than the mean."
    )

    nb_choice = st.selectbox(
        "Select count outcome for Negative Binomial regression",
        COUNT_VARIABLES,
        key="nb_variable"
    )

    if st.button(
        "Run Negative Binomial Regression",
        key="nb_button"
    ):

        try:

            model, result_table = negative_binomial_regression(
                df,
                nb_choice
            )

            st.session_state.analysis_results[
                "nb_model"
            ] = model

            st.session_state.analysis_results[
                "nb_table"
            ] = result_table

            st.session_state.analysis_results[
                "nb_outcome"
            ] = nb_choice

        except Exception as error:
            st.error(
                f"Negative Binomial regression failed: {error}"
            )

    if "nb_table" in st.session_state.analysis_results:

        st.markdown("**Negative Binomial regression results**")

        st.dataframe(
            st.session_state.analysis_results[
                "nb_table"
            ].round(4),
            use_container_width=True
        )

        with st.expander("Negative Binomial model summary"):
            st.text(
                st.session_state.analysis_results[
                    "nb_model"
                ].summary().as_text()
            )

    st.divider()


    # ========================================================
    # 21. DATA VISUALIZATIONS
    # ========================================================

    st.subheader("F. Data Visualizations")

    st.write(
        "Visualize distributions and compare selected variables "
        "between Female and Male groups."
    )

    plot_variable = st.selectbox(
        "Select variable for visualization",
        CONTINUOUS_VARIABLES + COUNT_VARIABLES,
        key="plot_variable"
    )

    plot_type = st.radio(
        "Select plot type",
        [
            "Group comparison (Boxplot)",
            "Overall distribution (Histogram)"
        ],
        horizontal=True,
        key="plot_type"
    )

    if st.button(
        "Generate Graph",
        key="graph_button"
    ):

        try:

            if plot_type == "Group comparison (Boxplot)":

                fig = create_group_plot(
                    df,
                    plot_variable,
                    f"{plot_variable} by sex"
                )

            else:

                fig = create_distribution_plot(
                    df,
                    plot_variable,
                    f"Distribution of {plot_variable}"
                )

            if fig is None:
                st.warning("No data available for this graph.")
            else:
                st.session_state.analysis_results[
                    "current_graph"
                ] = fig

        except Exception as error:
            st.error(f"Graph generation failed: {error}")

    if "current_graph" in st.session_state.analysis_results:

        st.pyplot(
            st.session_state.analysis_results["current_graph"],
            use_container_width=True
        )

    st.divider()


    # ========================================================
    # 22. AI INTERPRETATION OF STATISTICAL RESULTS
    # ========================================================

    st.subheader("G. AI Interpretation of Statistical Results")

    st.write(
        "Use AI to help explain statistical outputs in plain "
        "academic language. The researcher remains responsible "
        "for checking the interpretation."
    )

    if st.button(
        "Interpret Available Statistical Results",
        key="interpret_button",
        type="primary"
    ):

        interpretation_parts = []

        if "descriptive_overall" in st.session_state.analysis_results:

            interpretation_parts.append(
                "DESCRIPTIVE STATISTICS:\n"
                + st.session_state.analysis_results[
                    "descriptive_overall"
                ].round(3).to_string()
            )

        if "welch" in st.session_state.analysis_results:

            interpretation_parts.append(
                "WELCH T-TEST:\n"
                + str(
                    st.session_state.analysis_results["welch"]
                )
            )

        if "binary" in st.session_state.analysis_results:

            result = st.session_state.analysis_results["binary"]

            interpretation_parts.append(
                "BINARY COMPARISON:\n"
                + str(result["table"])
                + "\nTest: "
                + str(result["test"])
                + "\np-value: "
                + str(result["p_value"])
            )

        if "poisson_table" in st.session_state.analysis_results:

            interpretation_parts.append(
                "POISSON REGRESSION:\n"
                + st.session_state.analysis_results[
                    "poisson_table"
                ].round(4).to_string()
            )

        if "nb_table" in st.session_state.analysis_results:

            interpretation_parts.append(
                "NEGATIVE BINOMIAL REGRESSION:\n"
                + st.session_state.analysis_results[
                    "nb_table"
                ].round(4).to_string()
            )

        if not interpretation_parts:

            st.warning(
                "Run at least one statistical analysis before "
                "requesting AI interpretation."
            )

        else:

            results_text = "\n\n".join(interpretation_parts)

            interpretation_prompt = f"""
RESEARCH TOPIC:
{topic}

RESEARCH OBJECTIVE:
{objective}

DATA SOURCE:
{st.session_state.data_source}

STATISTICAL OUTPUTS:
{results_text[:4500]}

Please explain the available statistical results.

Organize the explanation into:

1. What analysis was conducted?
2. What does the method measure?
3. What do the reported estimates mean?
4. What do the p-values and confidence intervals indicate?
5. What are the important limitations?
6. What should the researcher check next?
7. How could the findings be described academically?

IMPORTANT:
- If the dataset is synthetic, clearly state that the results
  are hypothetical demonstrations only.
- Do not describe synthetic results as real patient findings.
- Do not claim causation from observational comparisons.
- Do not invent missing results.
- Explain IRR as an incidence rate ratio where relevant.
- Distinguish statistical significance from clinical importance.
"""

            with st.spinner("Interpreting statistical outputs..."):

                interpretation = ask_ai(
                    interpretation_prompt,
                    max_tokens=1800
                )

            st.session_state.analysis_results[
                "ai_interpretation"
            ] = interpretation

    if "ai_interpretation" in st.session_state.analysis_results:

        st.markdown("#### AI Interpretation")

        st.markdown(
            st.session_state.analysis_results[
                "ai_interpretation"
            ]
        )

        st.download_button(
            "Download AI Interpretation",
            data=st.session_state.analysis_results[
                "ai_interpretation"
            ],
            file_name="statistical_interpretation.txt",
            mime="text/plain",
            key="interpretation_download"
        )


    # ========================================================
    # 23. DOWNLOAD ANALYSIS DATA
    # ========================================================

    st.divider()

    st.subheader("H. Download Analysis Dataset")

    csv_data = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download Dataset as CSV",
        data=csv_data,
        file_name="research_analysis_dataset.csv",
        mime="text/csv",
        key="dataset_download"
    )

else:

    st.info(
        "Statistical analysis will become available after "
        "a valid dataset is loaded."
    )


# ============================================================
# 24. STEP 5 — INTERPRETATION, BIAS AND RESEARCH QUALITY
# ============================================================

st.divider()

st.header("4. Research Interpretation and Quality")

generate_stage(
    5,
    "Interpretation, Bias and Research Quality",
    "Consider how results should be interpreted and what "
    "methodological limitations may affect the findings.",
    """
1. Explain how findings should be interpreted.
2. Distinguish statistical significance from clinical
   or practical significance.
3. Discuss effect sizes and confidence intervals.
4. Identify possible confounding.
5. Discuss selection bias and information bias.
6. Consider missing data and measurement limitations.
7. Identify possible sources of uncertainty.
8. Discuss generalizability.
9. Suggest sensitivity or robustness checks where appropriate.
10. Identify what requires supervisor or statistician review.

If statistical outputs are available, use them as context.
Do not invent findings or claim that hypothetical results
have already been observed.
"""
)

st.divider()


# ============================================================
# 25. STEP 6 — RESEARCH PROPOSAL DEVELOPMENT
# ============================================================

st.header("5. Research Proposal Development")

generate_stage(
    6,
    "Research Proposal Development",
    "Bring the previous research-development stages together "
    "into a coherent quantitative research proposal.",
    """
Prepare a structured research proposal using the research
topic, objective, and work from previous stages.

Include:

1. Proposed title
2. Background and research rationale
3. Research problem
4. Research question
5. Primary and secondary objectives
6. Proposed study design
7. Study population and setting
8. Eligibility criteria
9. Variables and operational definitions
10. Data collection approach
11. Statistical analysis plan
12. Ethical considerations
13. Potential limitations
14. Expected contribution to pharmacy or health services research

Maintain consistency across all sections.

Do not invent literature citations, study findings,
or evidence of a confirmed research gap.

Clearly label points that need literature review,
supervisor input, or further verification.
"""
)

st.divider()


# ============================================================
# 26. FINAL CONSOLIDATED RESEARCH PROPOSAL
# ============================================================

st.header("6. Final Research Proposal")

st.write(
    "Generate one consolidated proposal using the research "
    "work developed throughout the workflow."
)

if st.button(
    "Generate Consolidated Research Proposal",
    type="primary",
    key="final_proposal_button"
):

    previous_work = get_previous_work()

    if not previous_work.strip():

        st.warning(
            "Please complete at least one research stage first."
        )

    else:

        final_prompt = f"""
RESEARCH TOPIC:
{topic}

OVERALL OBJECTIVE:
{objective}

RESEARCH WORK DEVELOPED:
{previous_work[:8500]}

STATISTICAL ANALYSIS CONTEXT:
{st.session_state.data_source}

Prepare a coherent consolidated quantitative research proposal.

Organize it into:

1. Title
2. Background and rationale
3. Research problem
4. Research question
5. Research objectives
6. Study design and methodology
7. Study population and setting
8. Variables and operational definitions
9. Data collection
10. Statistical analysis plan
11. Research quality and potential bias
12. Ethical considerations
13. Limitations
14. Potential contribution

Identify inconsistencies between earlier stages.
Do not silently invent missing information.
Clearly mark sections that need further development,
literature verification, or supervisor input.

Do not insert synthetic analysis results as real research findings.
"""

        with st.spinner(
            "Preparing the consolidated research proposal..."
        ):

            final_result = ask_ai(
                final_prompt,
                max_tokens=2500
            )

        st.session_state.research_outputs[
            "Final Research Proposal"
        ] = final_result

if "Final Research Proposal" in st.session_state.research_outputs:

    st.markdown("### Consolidated Proposal")

    st.markdown(
        st.session_state.research_outputs[
            "Final Research Proposal"
        ]
    )

    st.download_button(
        "Download Complete Research Proposal",
        data=st.session_state.research_outputs[
            "Final Research Proposal"
        ],
        file_name="quantitative_research_proposal.txt",
        mime="text/plain",
        key="final_proposal_download"
    )


# ============================================================
# 27. RESEARCH PROGRESS
# ============================================================

st.divider()

st.header("Research Progress")

progress_stages = [
    "Research Questions and Objectives",
    "Study Design and Methodology",
    "Variables and Data Collection",
    "Statistical Analysis Plan",
    "Interpretation, Bias and Research Quality",
    "Research Proposal Development"
]

completed_stages = sum(
    1 for stage in progress_stages
    if stage in st.session_state.research_outputs
)

st.progress(
    completed_stages / len(progress_stages)
)

st.write(
    f"Research stages completed: "
    f"{completed_stages} / {len(progress_stages)}"
)

for stage in progress_stages:

    if stage in st.session_state.research_outputs:
        st.success(f"Completed: {stage}")
    else:
        st.info(f"Pending: {stage}")


# ============================================================
# 28. RESEARCH GUIDELINES
# ============================================================

st.divider()

st.subheader("Important Research Guidelines")

st.markdown(
    """
    - AI suggestions are a starting point, not final research decisions.
    - Study design must match the research question.
    - Variables must be clearly defined before analysis.
    - Statistical methods must match the outcome type and study design.
    - Welch's t-test compares means between two independent groups.
    - Binary comparisons examine differences in proportions.
    - Poisson regression models count outcomes.
    - Negative Binomial regression can account for overdispersion.
    - IRR means incidence rate ratio.
    - Confounding, missing data, and bias must be considered.
    - Synthetic data must never be presented as real patient evidence.
    - Literature claims and references must be independently verified.
    - Final methods and interpretations require supervisor review.
    """
)


# ============================================================
# 29. FOOTER
# ============================================================

st.divider()

st.caption(
    "AI-Assisted Quantitative Research Assistant | "
    "Research planning, statistical analysis, and proposal development"
)
