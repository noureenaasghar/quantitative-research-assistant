# ============================================================
# AI-ASSISTED QUANTITATIVE RESEARCH ASSISTANT
# Complete Streamlit Application
# ============================================================

import os
import io
import textwrap
import warnings
from datetime import datetime

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats
import statsmodels.api as sm

from groq import Groq

warnings.filterwarnings("ignore")

# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Quantitative Research Assistant",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 AI-Assisted Quantitative Research Assistant")

st.markdown(
    """
    An educational research-support tool for quantitative
    health and pharmacy research.

    It supports dataset exploration, statistical analysis,
    research planning, and AI-assisted interpretation.
    """
)

st.caption(
    "Demonstration prototype | Statistical outputs require "
    "researcher verification before scientific use."
)

# ============================================================
# 2. DEMONSTRATION RESEARCH TOPIC
# ============================================================

DEFAULT_TOPIC = (
    "Sex-Based Differences in Kidney-Related Healthcare "
    "Utilization, Monitoring, and Clinical Outcomes Among "
    "Patients with Tuberous Sclerosis Complex (TSC)"
)

DEFAULT_OBJECTIVE = (
    "To explore whether healthcare utilization, kidney-related "
    "monitoring, and selected clinical outcomes differ by sex "
    "among patients with Tuberous Sclerosis Complex."
)

# ============================================================
# 3. APPLICATION MODULES
# ============================================================

MODULES = {
    "Research Questions": (
        "Develop focused, answerable quantitative research "
        "questions and objectives."
    ),
    "Study Design & Methodology": (
        "Explore suitable observational study designs, "
        "populations, eligibility criteria, and data sources."
    ),
    "Variables & Data Collection": (
        "Identify outcomes, exposures, covariates, operational "
        "definitions, and data collection requirements."
    ),
    "Statistical Analysis Planner": (
        "Select statistical methods based on outcome type, "
        "study design, assumptions, and available data."
    ),
    "Advanced Methods Guidance": (
        "Explore methods such as regression, survival analysis, "
        "propensity scores, longitudinal analysis, and "
        "competing-risk approaches when appropriate."
    ),
    "Research Quality Reviewer": (
        "Review methodological clarity, bias, confounding, "
        "missing data, limitations, and reproducibility."
    ),
    "Literature Gap & Rationale": (
        "Develop a cautious literature-review framework and "
        "identify possible research gaps requiring verification."
    ),
    "Research Proposal Builder": (
        "Organize the research rationale, objectives, methods, "
        "analysis plan, and limitations into a proposal draft."
    ),
    "Research Integrity & Validation": (
        "Check consistency, unsupported claims, fabricated "
        "results, ethical considerations, and validation needs."
    )
}

# ============================================================
# 4. GROQ API CONNECTION
# ============================================================

def get_groq_client():

    api_key = None

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


MODEL_NAME = "openai/gpt-oss-20b"

# ============================================================
# 5. AI SYSTEM INSTRUCTIONS
# ============================================================

SYSTEM_PROMPT = """
You are an AI-assisted quantitative health research support
assistant.

Your role is to help researchers plan, understand, and review
quantitative health and pharmacy research.

IMPORTANT RULES:

1. Never invent study findings, numerical results, datasets,
   citations, journal articles, DOIs, or references.

2. Clearly distinguish between:
   - Information provided by the user
   - General methodological guidance
   - Assumptions
   - Suggestions requiring verification

3. Do not claim that a statistical method is appropriate
   without considering the outcome type, study design,
   distribution, sample size, and assumptions.

4. For count outcomes, discuss Poisson regression and
   negative binomial regression where appropriate. Mention
   overdispersion assessment and exposure/follow-up time.

5. For binary outcomes, consider logistic regression or
   other suitable models depending on the study design.

6. For time-to-event outcomes, discuss survival analysis
   and relevant assumptions.

7. Consider confounding, missing data, selection bias,
   measurement bias, and limitations.

8. Do not interpret synthetic demonstration data as real
   patient evidence.

9. Explain statistical concepts in clear, accessible language.

10. State when a statistician, supervisor, or subject expert
    should verify the analysis.

11. Use structured headings and practical recommendations.

12. Do not present a possible literature gap as established
    fact unless supporting sources have been provided.
"""

# ============================================================
# 6. AI RESPONSE FUNCTION
# ============================================================

def generate_ai_response(
    client,
    module,
    topic,
    objective,
    context,
    uploaded_text=""
):

    if client is None:
        return (
            "Groq API connection is not available.\n\n"
            "Please check that GROQ_API_KEY is correctly added "
            "to Streamlit Secrets."
        )

    module_instruction = MODULES.get(
        module,
        "Provide relevant quantitative research guidance."
    )

    user_prompt = f"""
RESEARCH TOPIC:
{topic}

RESEARCH OBJECTIVE:
{objective}

SELECTED MODULE:
{module}

MODULE TASK:
{module_instruction}

ADDITIONAL CONTEXT PROVIDED BY THE USER:
{context}

UPLOADED TEXT:
{uploaded_text}

Please complete the selected module.

Use clear headings. Explain technical terms briefly.
Do not invent evidence, references, data, or results.
Identify assumptions, limitations, and points requiring
expert verification.
"""

    try:

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            temperature=0.2,
            max_completion_tokens=3500
        )

        return response.choices[0].message.content

    except Exception as error:

        return (
            "AI request could not be completed.\n\n"
            f"Error details: {str(error)}"
        )

# ============================================================
# END OF PART 1
# Continue directly with Part 2 below this line.
# ============================================================
# ============================================================
# 7. SYNTHETIC TSC DEMONSTRATION DATASET
# ============================================================

def generate_synthetic_data(n=500, seed=42):

    rng = np.random.default_rng(seed)

    sex = rng.choice(
        ["Female", "Male"],
        size=n,
        p=[0.52, 0.48]
    )

    age = rng.integers(18, 75, size=n)

    disease_duration = np.maximum(
        1,
        age - rng.integers(10, 25, size=n)
    )

    kidney_monitoring = rng.binomial(
        1,
        np.where(sex == "Female", 0.72, 0.65)
    )

    specialist_visits = rng.poisson(
        np.where(sex == "Female", 3.8, 3.3)
    )

    imaging_count = rng.poisson(
        np.where(sex == "Female", 2.4, 2.1)
    )

    hospitalization = rng.binomial(
        1,
        np.where(sex == "Female", 0.24, 0.29)
    )

    kidney_complication = rng.binomial(
        1,
        np.where(sex == "Female", 0.18, 0.22)
    )

    eGFR = np.clip(
        rng.normal(
            np.where(sex == "Female", 91, 88),
            14,
            n
        ),
        25,
        125
    )

    annual_healthcare_visits = (
        specialist_visits
        + rng.poisson(2.0, size=n)
    )

    data = pd.DataFrame({
        "patient_id": [
            f"TSC-{i:04d}" for i in range(1, n + 1)
        ],
        "sex": sex,
        "age": age,
        "disease_duration": disease_duration,
        "kidney_monitoring": kidney_monitoring,
        "specialist_visits": specialist_visits,
        "imaging_count": imaging_count,
        "hospitalization": hospitalization,
        "kidney_complication": kidney_complication,
        "eGFR": eGFR.round(2),
        "annual_healthcare_visits": annual_healthcare_visits
    })

    return data


# ============================================================
# 8. SESSION STATE INITIALIZATION
# ============================================================

if "data" not in st.session_state:
    st.session_state.data = generate_synthetic_data()

if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = {}

if "ai_history" not in st.session_state:
    st.session_state.ai_history = []

if "selected_topic" not in st.session_state:
    st.session_state.selected_topic = DEFAULT_TOPIC

if "selected_objective" not in st.session_state:
    st.session_state.selected_objective = DEFAULT_OBJECTIVE


# ============================================================
# 9. DATA UPLOAD
# ============================================================

st.sidebar.header("📁 Dataset")

data_source = st.sidebar.radio(
    "Choose dataset source",
    [
        "Synthetic demonstration dataset",
        "Upload my CSV dataset"
    ]
)

if data_source == "Synthetic demonstration dataset":

    if st.sidebar.button("Generate new synthetic dataset"):

        st.session_state.data = generate_synthetic_data(
            seed=int(
                np.random.default_rng().integers(1, 100000)
            )
        )

        st.session_state.analysis_results = {}

        st.rerun()

else:

    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV file",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:

            uploaded_data = pd.read_csv(uploaded_file)

            if uploaded_data.empty:
                st.sidebar.error("Uploaded CSV is empty.")

            else:

                st.session_state.data = uploaded_data

                st.session_state.analysis_results = {}

                st.sidebar.success(
                    "CSV dataset uploaded successfully."
                )

        except Exception as error:

            st.sidebar.error(
                f"Unable to read CSV: {error}"
            )


# ============================================================
# 10. DATA VALIDATION FUNCTION
# ============================================================

def validate_dataset(df):

    results = {}

    results["Number of records"] = len(df)

    results["Number of variables"] = len(df.columns)

    results["Missing cells"] = int(
        df.isnull().sum().sum()
    )

    results["Duplicate rows"] = int(
        df.duplicated().sum()
    )

    results["Duplicate column names"] = int(
        df.columns.duplicated().sum()
    )

    if "patient_id" in df.columns:

        results["Unique patient IDs"] = int(
            df["patient_id"].nunique()
        )

        results["Duplicate patient IDs"] = int(
            df["patient_id"].duplicated().sum()
        )

    else:

        results["Unique patient IDs"] = "Not available"

        results["Duplicate patient IDs"] = "Not available"

    results["Completely empty columns"] = int(
        df.isnull().all().sum()
    )

    return results


# ============================================================
# 11. DATASET OVERVIEW
# ============================================================

st.header("📊 Dataset Overview")

df = st.session_state.data

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Total records",
        f"{len(df):,}"
    )

with col2:
    st.metric(
        "Total variables",
        f"{len(df.columns):,}"
    )

with col3:
    st.metric(
        "Missing cells",
        f"{df.isnull().sum().sum():,}"
    )

with st.expander("View dataset", expanded=True):

    st.dataframe(
        df,
        use_container_width=True,
        height=300
    )

with st.expander("View variable information"):

    variable_info = pd.DataFrame({
        "Variable": df.columns,
        "Data type": [
            str(df[column].dtype)
            for column in df.columns
        ],
        "Non-missing values": [
            int(df[column].notna().sum())
            for column in df.columns
        ],
        "Missing values": [
            int(df[column].isna().sum())
            for column in df.columns
        ],
        "Unique values": [
            int(df[column].nunique())
            for column in df.columns
        ]
    })

    st.dataframe(
        variable_info,
        use_container_width=True
    )


# ============================================================
# 12. DATA QUALITY VALIDATION
# ============================================================

st.header("✅ Data Quality Validation")

validation_results = validate_dataset(df)

validation_table = pd.DataFrame(
    list(validation_results.items()),
    columns=["Validation check", "Result"]
)

st.dataframe(
    validation_table,
    use_container_width=True,
    hide_index=True
)

if (
    validation_results["Missing cells"] == 0
    and validation_results["Duplicate rows"] == 0
    and validation_results["Duplicate column names"] == 0
):

    st.success(
        "Basic data quality checks passed."
    )

else:

    st.warning(
        "Potential data quality issues detected. "
        "Review missing values, duplicate records, "
        "and column names before analysis."
    )

# ============================================================
# END OF PART 2
# Continue directly with Part 3 below this line.
# ============================================================
# ============================================================
# 13. STATISTICAL ANALYSIS FUNCTIONS
# ============================================================

def descriptive_statistics(df):

    numeric_df = df.select_dtypes(
        include=np.number
    )

    if numeric_df.empty:
        return pd.DataFrame()

    summary = numeric_df.describe().T

    summary["missing"] = numeric_df.isnull().sum()

    summary["median"] = numeric_df.median()

    summary["variance"] = numeric_df.var()

    summary["skewness"] = numeric_df.skew()

    return summary.round(3)


# ============================================================
# 14. WELCH TWO-SAMPLE T-TEST
# ============================================================

def welch_test(df, outcome, group="sex"):

    if outcome not in df.columns or group not in df.columns:
        return {"error": "Selected variables were not found."}

    data = df[[outcome, group]].copy()

    data[outcome] = pd.to_numeric(
        data[outcome],
        errors="coerce"
    )

    data = data.dropna()

    groups = data[group].unique()

    if len(groups) != 2:
        return {
            "error": "Welch t-test requires exactly two groups."
        }

    group1 = data.loc[
        data[group] == groups[0], outcome
    ]

    group2 = data.loc[
        data[group] == groups[1], outcome
    ]

    if len(group1) < 2 or len(group2) < 2:
        return {
            "error": "Each group must contain at least two observations."
        }

    statistic, p_value = stats.ttest_ind(
        group1,
        group2,
        equal_var=False
    )

    mean1 = group1.mean()
    mean2 = group2.mean()

    difference = mean1 - mean2

    variance1 = group1.var(ddof=1)
    variance2 = group2.var(ddof=1)

    n1 = len(group1)
    n2 = len(group2)

    standard_error = np.sqrt(
        variance1 / n1 + variance2 / n2
    )

    numerator = (
        variance1 / n1 + variance2 / n2
    ) ** 2

    denominator = (
        (variance1 / n1) ** 2 / (n1 - 1)
        + (variance2 / n2) ** 2 / (n2 - 1)
    )

    degrees_freedom = numerator / denominator

    critical_value = stats.t.ppf(
        0.975,
        degrees_freedom
    )

    ci_lower = difference - critical_value * standard_error

    ci_upper = difference + critical_value * standard_error

    pooled_sd = np.sqrt(
        (
            (n1 - 1) * variance1
            + (n2 - 1) * variance2
        ) / (n1 + n2 - 2)
    )

    cohens_d = (
        difference / pooled_sd
        if pooled_sd > 0 else np.nan
    )

    correction = (
        1 - (3 / (4 * (n1 + n2) - 9))
    )

    hedges_g = cohens_d * correction

    return {
        "group_1": str(groups[0]),
        "group_2": str(groups[1]),
        "n_1": n1,
        "n_2": n2,
        "mean_1": mean1,
        "mean_2": mean2,
        "mean_difference": difference,
        "standard_error": standard_error,
        "t_statistic": statistic,
        "degrees_freedom": degrees_freedom,
        "p_value": p_value,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "hedges_g": hedges_g
    }


# ============================================================
# 15. BINARY OUTCOME COMPARISON
# ============================================================

def binary_comparison(df, outcome, group="sex"):

    if outcome not in df.columns or group not in df.columns:
        return {"error": "Selected variables were not found."}

    data = df[[outcome, group]].dropna().copy()

    groups = data[group].unique()

    if len(groups) != 2:
        return {
            "error": "Comparison requires exactly two groups."
        }

    table = pd.crosstab(
        data[group],
        data[outcome]
    )

    if table.shape != (2, 2):
        return {
            "error": "Both variables must have exactly two categories."
        }

    chi2, p_value, dof, expected = stats.chi2_contingency(
        table,
        correction=False
    )

    fisher_odds, fisher_p = stats.fisher_exact(table)

    percentages = (
        pd.crosstab(
            data[group],
            data[outcome],
            normalize="index"
        ) * 100
    )

    return {
        "table": table,
        "percentages": percentages.round(2),
        "chi_square": chi2,
        "chi_square_p": p_value,
        "fisher_odds_ratio": fisher_odds,
        "fisher_p": fisher_p,
        "expected_frequencies": expected
    }


# ============================================================
# 16. COUNT MODEL DATA PREPARATION
# ============================================================

def prepare_count_model_data(
    df,
    outcome,
    group="sex",
    exposure=None
):

    required = [outcome, group]

    if exposure:
        required.append(exposure)

    missing_columns = [
        column for column in required
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    data = df[required].copy()

    data[outcome] = pd.to_numeric(
        data[outcome],
        errors="coerce"
    )

    if exposure:
        data[exposure] = pd.to_numeric(
            data[exposure],
            errors="coerce"
        )

    data = data.dropna()

    if len(data) < 10:
        raise ValueError(
            "At least 10 complete observations are required."
        )

    if (data[outcome] < 0).any():
        raise ValueError(
            "Count outcome cannot contain negative values."
        )

    if not np.allclose(
        data[outcome],
        np.round(data[outcome])
    ):
        raise ValueError(
            "Outcome must contain whole-number counts."
        )

    if data[group].nunique() != 2:
        raise ValueError(
            "Group variable must contain exactly two categories."
        )

    if exposure:

        if (data[exposure] <= 0).any():
            raise ValueError(
                "Exposure values must be greater than zero."
            )

    categories = sorted(
        data[group].astype(str).unique()
    )

    data[group] = pd.Categorical(
        data[group].astype(str),
        categories=categories
    )

    reference_group = categories[0]

    group_dummy = (
        data[group] == categories[1]
    ).astype(float)

    X = pd.DataFrame({
        "Intercept": 1.0,
        f"{group}[T.{categories[1]}]": group_dummy
    })

    y = data[outcome].astype(float)

    if exposure:
        exposure_values = data[exposure].astype(float)
    else:
        exposure_values = None

    return (
        y,
        X,
        exposure_values,
        reference_group,
        categories[1]
    )


# ============================================================
# 17. FORMAT REGRESSION RESULTS
# ============================================================

def format_count_model_results(model, reference_group, comparison_group):

    confidence_intervals = model.conf_int()

    rows = []

    for parameter in model.params.index:

        if parameter.lower() == "alpha":
            continue

        coefficient = float(model.params[parameter])

        standard_error = float(model.bse[parameter])

        p_value = float(model.pvalues[parameter])

        lower = float(
            confidence_intervals.loc[parameter, 0]
        )

        upper = float(
            confidence_intervals.loc[parameter, 1]
        )

        rows.append({
            "Parameter": parameter,
            "Coefficient": coefficient,
            "Standard Error": standard_error,
            "IRR": np.exp(coefficient),
            "IRR 95% CI Lower": np.exp(lower),
            "IRR 95% CI Upper": np.exp(upper),
            "P-value": p_value
        })

    results = pd.DataFrame(rows)

    results.attrs["reference_group"] = reference_group
    results.attrs["comparison_group"] = comparison_group

    return results


# ============================================================
# 18. POISSON REGRESSION
# ============================================================

def poisson_regression(
    df,
    outcome,
    group="sex",
    exposure=None
):

    try:

        (
            y,
            X,
            exposure_values,
            reference_group,
            comparison_group
        ) = prepare_count_model_data(
            df,
            outcome,
            group,
            exposure
        )

        if exposure_values is not None:

            model = sm.GLM(
                y,
                X,
                family=sm.families.Poisson(),
                exposure=exposure_values
            )

        else:

            model = sm.GLM(
                y,
                X,
                family=sm.families.Poisson()
            )

        fitted = model.fit()

        results = format_count_model_results(
            fitted,
            reference_group,
            comparison_group
        )

        dispersion = (
            fitted.pearson_chi2 / fitted.df_resid
            if fitted.df_resid > 0 else np.nan
        )

        results.attrs["pearson_dispersion"] = float(dispersion)
        results.attrs["deviance"] = float(fitted.deviance)
        results.attrs["degrees_of_freedom"] = int(fitted.df_resid)

        return results

    except Exception as error:

        return {
            "error": str(error)
        }


# ============================================================
# 19. NEGATIVE BINOMIAL REGRESSION
# ============================================================

def negative_binomial_regression(
    df,
    outcome,
    group="sex",
    exposure=None
):

    try:

        (
            y,
            X,
            exposure_values,
            reference_group,
            comparison_group
        ) = prepare_count_model_data(
            df,
            outcome,
            group,
            exposure
        )

        if exposure_values is not None:

            model = sm.NegativeBinomial(
                y,
                X,
                exposure=exposure_values,
                loglike_method="nb2"
            )

        else:

            model = sm.NegativeBinomial(
                y,
                X,
                loglike_method="nb2"
            )

        fitted = model.fit(
            disp=False,
            maxiter=200
        )

        results = format_count_model_results(
            fitted,
            reference_group,
            comparison_group
        )

        if "alpha" in fitted.params.index:

            alpha_index = fitted.params.index.get_loc("alpha")

            alpha = float(fitted.params.iloc[alpha_index])

            alpha_se = float(fitted.bse.iloc[alpha_index])

            results = pd.concat(
                [
                    results,
                    pd.DataFrame([{
                        "Parameter": "Alpha (dispersion)",
                        "Coefficient": alpha,
                        "Standard Error": alpha_se,
                        "IRR": np.nan,
                        "IRR 95% CI Lower": np.nan,
                        "IRR 95% CI Upper": np.nan,
                        "P-value": float(
                            fitted.pvalues.iloc[alpha_index]
                        )
                    }])
                ],
                ignore_index=True
            )

        results.attrs["reference_group"] = reference_group
        results.attrs["comparison_group"] = comparison_group

        return results

    except Exception as error:

        return {
            "error": str(error)
        }


# ============================================================
# END OF PART 3
# Continue directly with Part 4 below this line.
# ============================================================
# ============================================================
# 20. VISUALIZATION FUNCTIONS
# ============================================================

def create_group_plot(df, outcome, group="sex"):

    data = df[[outcome, group]].copy()

    data[outcome] = pd.to_numeric(
        data[outcome],
        errors="coerce"
    )

    data = data.dropna()

    if data.empty:
        return None

    summary = data.groupby(
        group,
        observed=True
    )[outcome].mean()

    fig, ax = plt.subplots(figsize=(8, 4))

    summary.plot(
        kind="bar",
        ax=ax
    )

    ax.set_title(
        f"Mean {outcome} by {group}"
    )

    ax.set_xlabel(group)
    ax.set_ylabel(f"Mean {outcome}")

    ax.tick_params(axis="x", rotation=0)

    fig.tight_layout()

    return fig


def create_distribution_plot(df, outcome):

    if outcome not in df.columns:
        return None

    values = pd.to_numeric(
        df[outcome],
        errors="coerce"
    ).dropna()

    if values.empty:
        return None

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.hist(
        values,
        bins=20,
        edgecolor="black"
    )

    ax.set_title(
        f"Distribution of {outcome}"
    )

    ax.set_xlabel(outcome)
    ax.set_ylabel("Frequency")

    fig.tight_layout()

    return fig


# ============================================================
# 21. ANALYSIS SUMMARY FUNCTION
# ============================================================

def create_analysis_summary(df):

    summary = {
        "Total observations": len(df),
        "Total variables": len(df.columns),
        "Missing cells": int(df.isnull().sum().sum()),
        "Duplicate rows": int(df.duplicated().sum())
    }

    if "sex" in df.columns:

        summary["Group distribution"] = (
            df["sex"]
            .value_counts(dropna=False)
            .to_dict()
        )

    return summary


# ============================================================
# 22. DOWNLOAD FUNCTIONS
# ============================================================

def dataframe_to_csv(df):

    return df.to_csv(
        index=False
    ).encode("utf-8")


def create_excel_report(df, descriptive=None):

    output = io.BytesIO()

    with pd.ExcelWriter(
        output,
        engine="openpyxl"
    ) as writer:

        df.to_excel(
            writer,
            sheet_name="Dataset",
            index=False
        )

        if descriptive is not None:

            descriptive.to_excel(
                writer,
                sheet_name="Descriptive Statistics"
            )

        validation = pd.DataFrame(
            list(validate_dataset(df).items()),
            columns=["Validation Check", "Result"]
        )

        validation.to_excel(
            writer,
            sheet_name="Data Validation",
            index=False
        )

    output.seek(0)

    return output.getvalue()


# ============================================================
# 23. DOWNLOAD SECTION
# ============================================================

st.header("📥 Download Dataset and Reports")

download_col1, download_col2 = st.columns(2)

with download_col1:

    st.download_button(
        label="Download dataset (CSV)",
        data=dataframe_to_csv(st.session_state.data),
        file_name="TSC_research_dataset.csv",
        mime="text/csv",
        use_container_width=True
    )

with download_col2:

    try:

        descriptive_for_export = descriptive_statistics(
            st.session_state.data
        )

        excel_data = create_excel_report(
            st.session_state.data,
            descriptive_for_export
        )

        st.download_button(
            label="Download research report (Excel)",
            data=excel_data,
            file_name="TSC_research_report.xlsx",
            mime=(
                "application/vnd.openxmlformats-officedocument."
                "spreadsheetml.sheet"
            ),
            use_container_width=True
        )

    except Exception as error:

        st.warning(
            f"Excel report could not be prepared: {error}"
        )


# ============================================================
# 24. VISUALIZATION SECTION
# ============================================================

st.header("📈 Data Visualizations")

current_df = st.session_state.data

numeric_columns = (
    current_df.select_dtypes(include=np.number)
    .columns.tolist()
)

if numeric_columns:

    selected_plot_variable = st.selectbox(
        "Select a variable for distribution",
        numeric_columns,
        key="distribution_variable"
    )

    distribution_fig = create_distribution_plot(
        current_df,
        selected_plot_variable
    )

    if distribution_fig is not None:

        st.pyplot(distribution_fig)

        plt.close(distribution_fig)

    if "sex" in current_df.columns:

        selected_group_variable = st.selectbox(
            "Select an outcome for group comparison",
            numeric_columns,
            key="group_plot_variable"
        )

        group_fig = create_group_plot(
            current_df,
            selected_group_variable
        )

        if group_fig is not None:

            st.pyplot(group_fig)

            plt.close(group_fig)

else:

    st.info(
        "No numeric variables are available for visualization."
    )


# ============================================================
# END OF PART 4
# Continue directly with Part 5 below this line.
# ============================================================
# ============================================================
# 25. MAIN APPLICATION INTERFACE
# ============================================================

st.divider()

st.header("🧪 Quantitative Research Analysis")

current_df = st.session_state.data

numeric_columns = (
    current_df.select_dtypes(include=np.number)
    .columns.tolist()
)

# ------------------------------------------------------------
# DESCRIPTIVE STATISTICS
# ------------------------------------------------------------

st.subheader("1. Descriptive Statistics")

if st.button(
    "Generate Descriptive Statistics",
    key="run_descriptive"
):

    descriptive_result = descriptive_statistics(
        current_df
    )

    st.session_state.analysis_results[
        "descriptive"
    ] = descriptive_result

if "descriptive" in st.session_state.analysis_results:

    st.dataframe(
        st.session_state.analysis_results["descriptive"],
        use_container_width=True
    )


# ------------------------------------------------------------
# WELCH T-TEST
# ------------------------------------------------------------

st.divider()

st.subheader("2. Group Comparison — Welch's t-test")

if "sex" in current_df.columns and numeric_columns:

    ttest_outcome = st.selectbox(
        "Select numeric outcome",
        numeric_columns,
        key="ttest_outcome"
    )

    if st.button(
        "Run Welch's t-test",
        key="run_ttest"
    ):

        ttest_result = welch_test(
            current_df,
            ttest_outcome,
            "sex"
        )

        st.session_state.analysis_results[
            "welch"
        ] = ttest_result

    if "welch" in st.session_state.analysis_results:

        result = st.session_state.analysis_results["welch"]

        if isinstance(result, dict) and "error" in result:

            st.error(result["error"])

        else:

            st.write("Welch's t-test results")

            st.dataframe(
                pd.DataFrame([
                    {
                        "Measure": "Group 1",
                        "Result": result["group_1"]
                    },
                    {
                        "Measure": "Group 2",
                        "Result": result["group_2"]
                    },
                    {
                        "Measure": "Group 1 sample size",
                        "Result": result["n_1"]
                    },
                    {
                        "Measure": "Group 2 sample size",
                        "Result": result["n_2"]
                    },
                    {
                        "Measure": "Group 1 mean",
                        "Result": round(result["mean_1"], 3)
                    },
                    {
                        "Measure": "Group 2 mean",
                        "Result": round(result["mean_2"], 3)
                    },
                    {
                        "Measure": "Mean difference",
                        "Result": round(
                            result["mean_difference"], 3
                        )
                    },
                    {
                        "Measure": "95% CI lower",
                        "Result": round(result["ci_lower"], 3)
                    },
                    {
                        "Measure": "95% CI upper",
                        "Result": round(result["ci_upper"], 3)
                    },
                    {
                        "Measure": "t-statistic",
                        "Result": round(result["t_statistic"], 3)
                    },
                    {
                        "Measure": "Degrees of freedom",
                        "Result": round(
                            result["degrees_freedom"], 3
                        )
                    },
                    {
                        "Measure": "P-value",
                        "Result": round(result["p_value"], 5)
                    },
                    {
                        "Measure": "Hedges' g",
                        "Result": round(result["hedges_g"], 3)
                    }
                ]),
                use_container_width=True,
                hide_index=True
            )

            st.caption(
                "Welch's t-test compares group means without "
                "assuming equal population variances. "
                "Interpret the confidence interval, effect size, "
                "and p-value together."
            )

else:

    st.info(
        "Welch's t-test requires a numeric outcome and a "
        "two-category group variable named sex."
    )


# ------------------------------------------------------------
# BINARY OUTCOME COMPARISON
# ------------------------------------------------------------

st.divider()

st.subheader("3. Binary Outcome Comparison")

binary_candidates = []

for column in current_df.columns:

    if column == "sex":
        continue

    if current_df[column].dropna().nunique() == 2:
        binary_candidates.append(column)

if "sex" in current_df.columns and binary_candidates:

    binary_outcome = st.selectbox(
        "Select binary outcome",
        binary_candidates,
        key="binary_outcome"
    )

    if st.button(
        "Compare Binary Outcome",
        key="run_binary"
    ):

        binary_result = binary_comparison(
            current_df,
            binary_outcome,
            "sex"
        )

        st.session_state.analysis_results[
            "binary"
        ] = binary_result

    if "binary" in st.session_state.analysis_results:

        result = st.session_state.analysis_results["binary"]

        if isinstance(result, dict) and "error" in result:

            st.error(result["error"])

        else:

            st.write("Observed frequencies")

            st.dataframe(
                result["table"],
                use_container_width=True
            )

            st.write("Row percentages (%)")

            st.dataframe(
                result["percentages"],
                use_container_width=True
            )

            metric1, metric2, metric3 = st.columns(3)

            metric1.metric(
                "Chi-square statistic",
                f"{result['chi_square']:.3f}"
            )

            metric2.metric(
                "Chi-square p-value",
                f"{result['chi_square_p']:.5f}"
            )

            metric3.metric(
                "Fisher exact p-value",
                f"{result['fisher_p']:.5f}"
            )

            st.caption(
                "Fisher's exact test is useful when expected "
                "cell frequencies are small. Select the test "
                "based on the data and study design."
            )

else:

    st.info(
        "No suitable binary outcome and sex grouping variable "
        "were found in the current dataset."
    )


# ------------------------------------------------------------
# POISSON REGRESSION
# ------------------------------------------------------------

st.divider()

st.subheader("4. Poisson Regression")

count_candidates = []

for column in numeric_columns:

    values = pd.to_numeric(
        current_df[column],
        errors="coerce"
    ).dropna()

    if (
        not values.empty
        and (values >= 0).all()
        and np.allclose(values, np.round(values))
        and values.nunique() > 1
    ):

        count_candidates.append(column)

if "sex" in current_df.columns and count_candidates:

    poisson_outcome = st.selectbox(
        "Select count outcome",
        count_candidates,
        key="poisson_outcome"
    )

    exposure_options = ["None"] + numeric_columns

    poisson_exposure = st.selectbox(
        "Select exposure or follow-up time (optional)",
        exposure_options,
        key="poisson_exposure"
    )

    if poisson_exposure == "None":
        poisson_exposure = None

    if st.button(
        "Run Poisson Regression",
        key="run_poisson"
    ):

        poisson_result = poisson_regression(
            current_df,
            poisson_outcome,
            "sex",
            poisson_exposure
        )

        st.session_state.analysis_results[
            "poisson"
        ] = poisson_result

    if "poisson" in st.session_state.analysis_results:

        result = st.session_state.analysis_results["poisson"]

        if isinstance(result, dict) and "error" in result:

            st.error(result["error"])

        else:

            st.write("Poisson regression results")

            st.dataframe(
                result,
                use_container_width=True,
                hide_index=True
            )

            dispersion = result.attrs.get(
                "pearson_dispersion",
                np.nan
            )

            st.metric(
                "Pearson dispersion statistic",
                (
                    f"{dispersion:.3f}"
                    if pd.notna(dispersion)
                    else "Not available"
                )
            )

            st.caption(
                "IRR means Incidence Rate Ratio. The reference "
                "group is shown in the model results. A Pearson "
                "dispersion value substantially above 1 may "
                "indicate overdispersion and should prompt "
                "further model assessment."
            )

else:

    st.info(
        "Poisson regression requires a two-category sex "
        "variable and a suitable count outcome."
    )


# ------------------------------------------------------------
# NEGATIVE BINOMIAL REGRESSION
# ------------------------------------------------------------

st.divider()

st.subheader("5. Negative Binomial Regression")

if "sex" in current_df.columns and count_candidates:

    nb_outcome = st.selectbox(
        "Select count outcome",
        count_candidates,
        key="nb_outcome"
    )

    nb_exposure = st.selectbox(
        "Select exposure or follow-up time (optional)",
        ["None"] + numeric_columns,
        key="nb_exposure"
    )

    if nb_exposure == "None":
        nb_exposure = None

    if st.button(
        "Run Negative Binomial Regression",
        key="run_nb"
    ):

        nb_result = negative_binomial_regression(
            current_df,
            nb_outcome,
            "sex",
            nb_exposure
        )

        st.session_state.analysis_results[
            "negative_binomial"
        ] = nb_result

    if "negative_binomial" in st.session_state.analysis_results:

        result = st.session_state.analysis_results[
            "negative_binomial"
        ]

        if isinstance(result, dict) and "error" in result:

            st.error(result["error"])

        else:

            st.write("Negative Binomial regression results")

            st.dataframe(
                result,
                use_container_width=True,
                hide_index=True
            )

            st.caption(
                "Negative Binomial regression can account for "
                "overdispersion in count outcomes. Alpha is the "
                "dispersion parameter, not an incidence rate ratio."
            )

else:

    st.info(
        "Negative Binomial regression requires a suitable "
        "count outcome and a two-category sex variable."
    )


# ============================================================
# 26. AI-ASSISTED RESEARCH WORKSPACE
# ============================================================

st.divider()

st.header("🤖 AI-Assisted Research Workspace")

st.write(
    "Use the AI assistant to develop research questions, "
    "review methodology, understand statistical concepts, "
    "and organize research proposals."
)

topic = st.text_area(
    "Research topic",
    value=st.session_state.selected_topic,
    height=90,
    key="ai_research_topic"
)

objective = st.text_area(
    "Research objective",
    value=st.session_state.selected_objective,
    height=90,
    key="ai_research_objective"
)

selected_module = st.selectbox(
    "Select research support module",
    list(MODULES.keys()),
    key="ai_module"
)

st.info(
    MODULES[selected_module]
)

additional_context = st.text_area(
    "Additional instructions or research context",
    placeholder=(
        "Enter your research question, study details, "
        "variables, or methodological concerns..."
    ),
    height=120,
    key="ai_context"
)

uploaded_research_file = st.file_uploader(
    "Optional: Upload a text file containing research notes",
    type=["txt"],
    key="ai_research_file"
)

uploaded_research_text = ""

if uploaded_research_file is not None:

    try:

        uploaded_research_text = (
            uploaded_research_file
            .getvalue()
            .decode("utf-8")
        )

    except Exception:

        st.warning(
            "The uploaded text file could not be decoded."
        )

if st.button(
    "Generate AI Research Guidance",
    type="primary",
    key="generate_ai"
):

    client = get_groq_client()

    with st.spinner(
        "Generating research guidance..."
    ):

        ai_response = generate_ai_response(
            client=client,
            module=selected_module,
            topic=topic,
            objective=objective,
            context=additional_context,
            uploaded_text=uploaded_research_text
        )

    st.session_state.ai_history.append({
        "module": selected_module,
        "topic": topic,
        "response": ai_response,
        "time": datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    })


# ------------------------------------------------------------
# DISPLAY AI OUTPUT
# ------------------------------------------------------------

if st.session_state.ai_history:

    st.subheader("AI Research Output")

    for index, item in enumerate(
        reversed(st.session_state.ai_history)
    ):

        with st.expander(
            f"{item['module']} — {item['time']}",
            expanded=(index == 0)
        ):

            st.markdown(
                item["response"]
            )

            st.download_button(
                label="Download this AI response",
                data=item["response"].encode("utf-8"),
                file_name=(
                    f"research_guidance_{index + 1}.txt"
                ),
                mime="text/plain",
                key=f"download_ai_{index}"
            )


# ============================================================
# 27. RESEARCH INTERPRETATION AND LIMITATIONS
# ============================================================

st.divider()

st.header("📌 Research Interpretation Guidelines")

st.markdown(
    """
    - Statistical significance does not automatically imply
      clinical or practical significance.
    - Effect sizes and confidence intervals should be
      considered alongside p-values.
    - Regression estimates depend on model specification
      and assumptions.
    - Confounding, missing data, selection bias, and
      measurement quality must be considered.
    - Synthetic demonstration data are not real patient
      records and cannot establish clinical evidence.
    - All final statistical analyses and interpretations
      require appropriate researcher and supervisor review.
    """
)


# ============================================================
# 28. FOOTER
# ============================================================

st.divider()

st.caption(
    "AI-Assisted Quantitative Research Assistant | "
    "Educational research prototype | "
    "Not a substitute for expert statistical review."
)

# ============================================================
# END OF APP.PY
# ============================================================
