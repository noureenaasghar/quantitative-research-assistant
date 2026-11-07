
import os
import io
import json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from scipy import stats
from scipy.stats import chi2_contingency, fisher_exact
from statsmodels.stats.proportion import proportion_confint
import statsmodels.api as sm
import statsmodels.formula.api as smf

from groq import Groq


# ============================================================
# 1. APP CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Quantitative Research Assistant",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 AI-Assisted Quantitative Research Assistant")

st.markdown(
    """
    An integrated research-support tool for developing,
    reviewing, and demonstrating quantitative health research.

    **Demonstration topic:** Sex-Based Differences in Kidney-Related
    Healthcare Utilization, Monitoring, and Clinical Outcomes
    Among Patients with Tuberous Sclerosis Complex (TSC).
    """
)

st.info(
    "Important: AI-generated outputs are research-support suggestions. "
    "They require verification by the researcher. Synthetic data are "
    "for software demonstration only and are not clinical evidence."
)


# ============================================================
# 2. GROQ CONFIGURATION
# ============================================================

MODEL_NAME = "openai/gpt-oss-20b"

MAX_TOKENS = 2200

MAX_CONTEXT_CHARS = 12000


def get_client():

    api_key = st.secrets.get(
        "GROQ_API_KEY",
        os.getenv("GROQ_API_KEY")
    )

    if not api_key:
        st.error(
            "GROQ_API_KEY is missing. Add it to Streamlit Secrets "
            "or your environment variables."
        )
        return None

    return Groq(api_key=api_key)


def ask_groq(prompt, context=""):

    client = get_client()

    if client is None:
        return "AI service unavailable: API key not configured."

    combined_context = str(context)[:MAX_CONTEXT_CHARS]

    full_prompt = f"""
    You are an academic quantitative health research assistant.

    Your role is to support a researcher with a Pharm.D background.
    Explain statistical and methodological concepts clearly.

    IMPORTANT RULES:

    1. Do not invent research findings, citations, or study results.
    2. Distinguish established evidence from hypotheses.
    3. Identify assumptions and methodological limitations.
    4. Do not automatically recommend advanced methods without
       explaining why they are appropriate.
    5. Do not claim causation from observational associations.
    6. Explain technical terminology in accessible language.
    7. Clearly identify information that requires verification.
    8. Never describe synthetic data as real patient data.

    PREVIOUS RESEARCH CONTEXT:
    {combined_context}

    CURRENT REQUEST:
    {prompt}
    """

    try:

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": full_prompt
                }
            ],
            temperature=0.2,
            max_tokens=MAX_TOKENS
        )

        return response.choices[0].message.content

    except Exception as e:

        return f"AI request failed: {str(e)}"


# ============================================================
# 3. RESEARCH TOPIC AND DEFAULT CONTEXT
# ============================================================

DEFAULT_RESEARCH_TOPIC = (
    "Sex-Based Differences in Kidney-Related Healthcare "
    "Utilization, Monitoring, and Clinical Outcomes "
    "Among Patients with Tuberous Sclerosis Complex"
)

DEFAULT_RESEARCH_CONTEXT = """
Research area:
Tuberous Sclerosis Complex (TSC) and kidney-related healthcare.

Primary research interest:
Differences between male and female patients in kidney-related
healthcare utilization, monitoring, and clinical outcomes.

Potential data sources:
Administrative claims, electronic health records, or linked
clinical datasets, depending on availability.

Potential outcomes:
Kidney-related encounters, renal imaging, nephrology visits,
kidney function monitoring, and validated kidney outcomes.

Important methodological considerations:
- Define the eligible TSC population.
- Avoid excluding patients with no kidney-related encounters.
- Distinguish baseline kidney disease from incident outcomes.
- Account for differences in follow-up duration.
- Consider confounding and missing data.
- Validate diagnosis, procedure, and laboratory definitions.
- Avoid causal claims from observational comparisons.

The actual study population, data source, age range, and follow-up
period must be determined according to data availability and
the principal investigator's research guidance.
"""


# ============================================================
# 4. RESEARCH MODULE DEFINITIONS
# ============================================================

MODULES = {

    "1. Research Questions": """
    Develop focused and answerable quantitative research questions.

    Include:
    - Primary research question
    - Secondary research questions
    - PICO or PECO framework where appropriate
    - Research objectives
    - Testable hypotheses
    - Feasibility considerations

    Ensure that questions are suitable for observational
    health services research.
    """,

    "2. Study Design & Methodology": """
    Recommend an appropriate quantitative study design.

    Consider:
    - Retrospective cohort
    - Cross-sectional study
    - Case-control study, if appropriate

    Explain:
    - Target population
    - Eligibility criteria
    - Study setting and data source
    - Index date
    - Baseline period
    - Follow-up period
    - Potential selection bias
    - Information bias
    - Confounding

    Do not select a design without explaining its rationale.
    """,

    "3. Variables & Data Collection": """
    Develop a structured variable framework.

    Organize variables into:
    - Exposure
    - Primary outcomes
    - Secondary outcomes
    - Potential confounders
    - Effect modifiers

    For every variable, provide:
    - Definition
    - Measurement
    - Data type
    - Possible data source
    - Coding considerations

    Distinguish baseline conditions from incident outcomes.
    Do not invent validated clinical definitions.
    """,

    "4. Statistical Analysis Planner": """
    Develop an analysis plan that connects each research
    objective to an appropriate statistical method.

    Consider:
    - Descriptive statistics
    - Group comparisons
    - Confidence intervals
    - Effect sizes
    - Count outcomes
    - Binary outcomes
    - Time-to-event outcomes
    - Multivariable adjustment

    Explain assumptions and alternatives.

    Do not automatically recommend a t-test for skewed
    count outcomes.
    """,

    "5. Advanced Statistical Methods": """
    Evaluate whether advanced statistical methods are
    justified by the research question and available data.

    Potential methods:
    - Poisson regression
    - Negative Binomial regression
    - Logistic regression
    - Cox proportional hazards regression
    - Competing-risk analysis
    - GEE or mixed-effects models

    Explain:
    - When each method is appropriate
    - Required variables
    - Key assumptions
    - Diagnostics
    - Interpretation

    Do not recommend machine learning, propensity scores,
    mediation, or causal inference automatically.
    """,

    "6. Research Quality Reviewer": """
    Review the proposed research for methodological quality.

    Assess:
    - Research question clarity
    - Study design suitability
    - Eligibility criteria
    - Variable definitions
    - Bias and confounding
    - Missing data
    - Statistical assumptions
    - Generalizability
    - Reproducibility

    Identify major concerns and practical improvements.
    Separate critical problems from optional enhancements.
    """,

    "7. Research Gap & Literature Review": """
    Develop a systematic literature search framework.

    Include:
    - Research themes
    - Search terms
    - Boolean search strategy
    - Relevant databases
    - Inclusion and exclusion criteria
    - Screening approach
    - Data extraction fields
    - Quality appraisal approach

    IMPORTANT:
    Do not claim that a research gap is established unless
    relevant literature has actually been searched and reviewed.

    Label any proposed gap as provisional.
    Do not fabricate references or publication details.
    """,

    "8. Proposal Builder": """
    Develop a structured quantitative research proposal.

    Include:
    - Proposed title
    - Background and rationale
    - Research question
    - Aim and objectives
    - Study design
    - Population and eligibility
    - Variables and outcomes
    - Statistical analysis plan
    - Ethical considerations
    - Limitations
    - Feasibility

    Ensure consistency between objectives, variables,
    study design, and statistical methods.

    Clearly identify information that still requires
    principal investigator or data-source confirmation.
    """,

    "9. Scientific Validation": """
    Conduct a scientific and methodological review of
    the proposed research.

    Check:
    - Whether claims are supported by evidence
    - Whether outcome definitions are valid
    - Whether study design supports the conclusions
    - Whether statistical methods match variable types
    - Whether assumptions are acknowledged
    - Whether causal language is justified
    - Whether limitations are adequately described

    Do not invent external validation or citations.
    """
}


# ============================================================
# END OF PART 1
# ============================================================

# ============================================================
# 5. SESSION STATE MANAGEMENT
# ============================================================

if "module_outputs" not in st.session_state:
    st.session_state.module_outputs = {}

if "research_topic" not in st.session_state:
    st.session_state.research_topic = DEFAULT_RESEARCH_TOPIC

if "research_context" not in st.session_state:
    st.session_state.research_context = DEFAULT_RESEARCH_CONTEXT

if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = None

if "active_dataset" not in st.session_state:
    st.session_state.active_dataset = None

if "dataset_source" not in st.session_state:
    st.session_state.dataset_source = None


# ============================================================
# 6. SYNTHETIC TSC DEMONSTRATION DATA
# ============================================================

def make_demo_data(n=500, seed=42):

    """
    Generate synthetic patient-level data for software
    demonstration only.

    These records do not represent real TSC patients.
    """

    rng = np.random.default_rng(seed)

    patient_id = [
        f"TSC_{i:04d}" for i in range(1, n + 1)
    ]

    sex = rng.choice(
        ["Female", "Male"],
        size=n,
        p=[0.52, 0.48]
    )

    age_years = np.clip(
        rng.normal(35, 14, n),
        18,
        80
    ).round(1)

    baseline_comorbidity = rng.binomial(
        1,
        0.30,
        n
    )

    followup_years = rng.uniform(
        1,
        5,
        n
    ).round(2)

    # Synthetic pattern created only to demonstrate
    # how analysis functions operate.

    sex_effect = np.where(
        sex == "Male",
        0.25,
        0
    )

    expected_encounters = (
        2.0
        + 0.035 * age_years
        + 1.0 * baseline_comorbidity
        + sex_effect
    )

    kidney_encounters = rng.poisson(
        expected_encounters
    )

    kidney_encounters = np.maximum(
        kidney_encounters,
        0
    )

    any_kidney_use = (
        kidney_encounters > 0
    ).astype(int)

    kidney_imaging_count = rng.poisson(
        1.0 + 0.35 * baseline_comorbidity
    )

    nephrology_visit_count = rng.poisson(
        0.8 + 0.30 * baseline_comorbidity
    )

    kidney_monitoring = rng.binomial(
        1,
        0.65,
        n
    )

    data = pd.DataFrame({

        "patient_id": patient_id,

        "sex": sex,

        "age_years": age_years,

        "baseline_comorbidity": baseline_comorbidity,

        "followup_years": followup_years,

        "any_kidney_use": any_kidney_use,

        "kidney_encounters": kidney_encounters,

        "kidney_imaging_count": kidney_imaging_count,

        "nephrology_visit_count": nephrology_visit_count,

        "kidney_monitoring": kidney_monitoring

    })

    return data


# ============================================================
# 7. DATASET VALIDATION
# ============================================================

def validate_dataset(data):

    """
    Check basic dataset structure and identify
    potential data-quality problems.
    """

    errors = []
    warnings = []

    if not isinstance(data, pd.DataFrame):
        errors.append("Uploaded data is not a valid table.")
        return False, errors, warnings

    if data.empty:
        errors.append("The dataset contains no records.")
        return False, errors, warnings

    if len(data.columns) == 0:
        errors.append("The dataset contains no columns.")
        return False, errors, warnings

    if data.columns.duplicated().any():
        errors.append("Duplicate column names were detected.")

    required_columns = [
        "sex"
    ]

    for column in required_columns:

        if column not in data.columns:
            errors.append(
                f"Required column missing: {column}"
            )

    if "sex" in data.columns:

        valid_sex = [
            "Female",
            "Male",
            "female",
            "male",
            "F",
            "M"
        ]

        observed = data["sex"].dropna().astype(str).unique()

        unexpected = [
            value for value in observed
            if value not in valid_sex
        ]

        if unexpected:
            warnings.append(
                "Unexpected sex categories found: "
                + ", ".join(unexpected)
            )

    if data.isna().sum().sum() > 0:

        missing_count = int(
            data.isna().sum().sum()
        )

        warnings.append(
            f"The dataset contains {missing_count} missing cells."
        )

    if "patient_id" in data.columns:

        if data["patient_id"].duplicated().any():

            warnings.append(
                "Duplicate patient IDs detected. "
                "Confirm whether the dataset contains repeated "
                "observations per patient."
            )

    numeric_columns = [
        "age_years",
        "followup_years",
        "kidney_encounters",
        "kidney_imaging_count",
        "nephrology_visit_count"
    ]

    for column in numeric_columns:

        if column in data.columns:

            if not pd.api.types.is_numeric_dtype(data[column]):

                errors.append(
                    f"{column} must contain numeric values."
                )

    if "age_years" in data.columns:

        if pd.api.types.is_numeric_dtype(data["age_years"]):

            if (data["age_years"].dropna() < 0).any():

                errors.append(
                    "Negative age values detected."
                )

    if "followup_years" in data.columns:

        if pd.api.types.is_numeric_dtype(data["followup_years"]):

            if (data["followup_years"].dropna() <= 0).any():

                warnings.append(
                    "Zero or negative follow-up values detected. "
                    "These records cannot contribute valid "
                    "person-time rates."
                )

    return len(errors) == 0, errors, warnings


# ============================================================
# 8. DATASET SUMMARY
# ============================================================

def dataset_summary(data):

    """
    Generate a basic overview of the selected dataset.
    """

    summary = {

        "Number of records": len(data),

        "Number of variables": len(data.columns),

        "Missing cells": int(
            data.isna().sum().sum()
        ),

        "Duplicate rows": int(
            data.duplicated().sum()
        )

    }

    if "patient_id" in data.columns:

        summary["Unique patient IDs"] = int(
            data["patient_id"].nunique()
        )

    return summary


# ============================================================
# 9. DATA PREPARATION
# ============================================================

def prepare_analysis_data(data):

    """
    Prepare a copy of the dataset for analysis.

    Does not overwrite the original uploaded dataset.
    """

    prepared = data.copy()

    if "sex" in prepared.columns:

        prepared["sex"] = (
            prepared["sex"]
            .astype("string")
            .str.strip()
            .str.capitalize()
        )

    return prepared


# ============================================================
# END OF PART 2
# ============================================================
def _prepare_count_model_data(
    data,
    outcome_col,
    group_col,
    exposure_time_col
):
    required = [outcome_col, group_col, exposure_time_col]

    missing = [col for col in required if col not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = data[required].copy()

    df[outcome_col] = pd.to_numeric(
        df[outcome_col], errors="coerce"
    )
    df[exposure_time_col] = pd.to_numeric(
        df[exposure_time_col], errors="coerce"
    )

    df = df.dropna()

    df = df[
        (df[outcome_col] >= 0) &
        (df[exposure_time_col] > 0)
    ].copy()

    if len(df) < 10:
        raise ValueError("Not enough valid observations for regression.")

    if df[group_col].nunique() < 2:
        raise ValueError("Grouping variable must have at least two groups.")

    # Set the first alphabetically sorted group as reference
    levels = sorted(df[group_col].astype(str).unique())

    df[group_col] = pd.Categorical(
        df[group_col].astype(str),
        categories=levels
    )

    y = df[outcome_col].astype(float)
    exposure = df[exposure_time_col].astype(float)

    # Create dummy variables without relying on column names in formulas
    dummies = pd.get_dummies(
        df[group_col],
        drop_first=True,
        dtype=float
    )

    X = pd.concat(
        [
            pd.Series(1.0, index=df.index, name="Intercept"),
            dummies
        ],
        axis=1
    ).astype(float)

    parameter_labels = ["Intercept"]

    for level in levels[1:]:
        parameter_labels.append(
            f"{group_col}: {level} vs {levels[0]}"
        )

    X.columns = parameter_labels

    return df, y, X, exposure, parameter_labels, levels


def _format_count_model_results(
    params,
    standard_errors,
    p_values,
    confidence_intervals,
    parameter_labels
):
    results = []

    for i, label in enumerate(parameter_labels):

        coefficient = float(params[i])
        standard_error = float(standard_errors[i])
        p_value = float(p_values[i])

        ci_lower = float(confidence_intervals[i, 0])
        ci_upper = float(confidence_intervals[i, 1])

        results.append({
            "Parameter": label,
            "Coefficient": coefficient,
            "Standard Error": standard_error,
            "IRR": float(np.exp(coefficient)),
            "95% CI Lower": float(np.exp(ci_lower)),
            "95% CI Upper": float(np.exp(ci_upper)),
            "p-value": p_value
        })

    return pd.DataFrame(results).set_index("Parameter")

def descriptive_statistics(data):
    if data is None or data.empty:
        return pd.DataFrame()

    numeric_data = data.select_dtypes(include=[np.number])

    if numeric_data.empty:
        return pd.DataFrame()

    return numeric_data.describe().T
def poisson_regression(
    data,
    outcome_col,
    group_col="sex",
    exposure_time_col="followup_years"
):
    """
    Poisson regression with log follow-up time as an offset.
    """

    df, y, X, exposure, labels, levels = _prepare_count_model_data(
        data,
        outcome_col,
        group_col,
        exposure_time_col
    )

    model = sm.GLM(
        y,
        X,
        family=sm.families.Poisson(),
        offset=np.log(exposure)
    )

    fitted = model.fit()

    params = np.asarray(fitted.params)
    standard_errors = np.asarray(fitted.bse)
    p_values = np.asarray(fitted.pvalues)
    confidence_intervals = np.asarray(fitted.conf_int())

    results = _format_count_model_results(
        params,
        standard_errors,
        p_values,
        confidence_intervals,
        labels
    )

    # Model diagnostics
    dispersion = (
        fitted.pearson_chi2 / fitted.df_resid
        if fitted.df_resid > 0 else np.nan
    )

    results.attrs["pearson_dispersion"] = float(dispersion)
    results.attrs["deviance"] = float(fitted.deviance)
    results.attrs["degrees_of_freedom"] = float(fitted.df_resid)
    results.attrs["reference_group"] = levels[0]

    return results


def negative_binomial_regression(
    data,
    outcome_col,
    group_col="sex",
    exposure_time_col="followup_years"
):
    """
    Negative Binomial NB2 regression with follow-up exposure.
    Alpha is displayed separately as a dispersion parameter.
    """

    df, y, X, exposure, labels, levels = _prepare_count_model_data(
        data,
        outcome_col,
        group_col,
        exposure_time_col
    )

    model = sm.NegativeBinomial(
        y,
        X,
        loglike_method="nb2",
        exposure=exposure
    )

    fitted = model.fit(disp=False)

    params = np.asarray(fitted.params)
    standard_errors = np.asarray(fitted.bse)
    p_values = np.asarray(fitted.pvalues)
    confidence_intervals = np.asarray(fitted.conf_int())

    number_of_predictors = len(labels)

    # Only the intercept and group coefficients are converted to IRRs
    results = _format_count_model_results(
        params[:number_of_predictors],
        standard_errors[:number_of_predictors],
        p_values[:number_of_predictors],
        confidence_intervals[:number_of_predictors],
        labels
    )

    # Alpha is a dispersion parameter, NOT an incidence rate ratio
    alpha = float(params[number_of_predictors])
    alpha_se = float(standard_errors[number_of_predictors])
    alpha_p = float(p_values[number_of_predictors])

    results.loc["Alpha (dispersion parameter)", :] = [
        alpha,
        alpha_se,
        np.nan,
        np.nan,
        np.nan,
        alpha_p
    ]

    results.attrs["alpha"] = alpha
    results.attrs["reference_group"] = levels[0]

    return results

# ============================================================
# 16. DATA VISUALIZATION
# ============================================================

def make_plot(
    data,
    plot_type,
    group_col,
    outcome_col
):

    required = [group_col, outcome_col]

    missing = [
        col for col in required
        if col not in data.columns
    ]

    if missing:
        return None, "Required columns are missing."

    plot_data = data[required].dropna().copy()

    if plot_data.empty:
        return None, "No complete observations available."

    if not pd.api.types.is_numeric_dtype(
        plot_data[outcome_col]
    ):
        return None, "The selected outcome must be numeric."

    fig, ax = plt.subplots(figsize=(8, 5))

    try:

        if plot_type == "Box plot":

            groups = list(
                plot_data[group_col].unique()
            )

            values = [
                plot_data.loc[
                    plot_data[group_col] == group,
                    outcome_col
                ].values
                for group in groups
            ]

            ax.boxplot(
                values,
                tick_labels=[str(x) for x in groups]
            )

            ax.set_title(
                f"{outcome_col} by {group_col}"
            )

            ax.set_xlabel(group_col)
            ax.set_ylabel(outcome_col)

        elif plot_type == "Histogram":

            ax.hist(
                plot_data[outcome_col],
                bins=20,
                edgecolor="black"
            )

            ax.set_title(
                f"Distribution of {outcome_col}"
            )

            ax.set_xlabel(outcome_col)
            ax.set_ylabel("Frequency")

        elif plot_type == "Bar chart":

            grouped = (
                plot_data.groupby(group_col)[outcome_col]
                .mean()
            )

            ax.bar(
                grouped.index.astype(str),
                grouped.values
            )

            ax.set_title(
                f"Mean {outcome_col} by {group_col}"
            )

            ax.set_xlabel(group_col)
            ax.set_ylabel(f"Mean {outcome_col}")

        else:

            plt.close(fig)

            return None, "Unsupported plot type."

        ax.grid(
            axis="y",
            alpha=0.2
        )

        fig.tight_layout()

        return fig, None

    except Exception as e:

        plt.close(fig)

        return None, f"Visualization error: {str(e)}"


# ============================================================
# 17. CRUDE UTILIZATION RATES PER 100 PERSON-YEARS
# ============================================================

def calculate_crude_rates(
    data,
    group_col="sex",
    event_col="kidney_encounters",
    followup_col="followup_years"
):

    """
    Calculate crude event rates per 100 person-years.

    Assumption:
    Each row represents one patient's observation period.
    If data contain repeated patient-period records, the
    dataset must be appropriately structured before analysis.
    """

    required = [
        group_col,
        event_col,
        followup_col
    ]

    missing = [
        col for col in required
        if col not in data.columns
    ]

    if missing:
        return None, (
            "Missing required columns: "
            + ", ".join(missing)
        )

    subset = data[required].copy()

    for column in [event_col, followup_col]:

        subset[column] = pd.to_numeric(
            subset[column],
            errors="coerce"
        )

    subset = subset.dropna()

    subset = subset[
        (subset[followup_col] > 0)
        &
        (subset[event_col] >= 0)
    ].copy()

    if subset.empty:
        return None, "No valid observations for rate calculation."

    if not np.allclose(
        subset[event_col],
        np.floor(subset[event_col])
    ):
        return None, "Event counts must be whole numbers."

    subset[event_col] = subset[event_col].astype(int)

    results = []

    for group, group_data in subset.groupby(group_col):

        events = int(
            group_data[event_col].sum()
        )

        person_years = float(
            group_data[followup_col].sum()
        )

        if person_years <= 0:
            continue

        rate = (
            events / person_years
        ) * 100

        # Exact Poisson confidence interval
        # for the event rate.

        if events == 0:

            lower_count = 0

        else:

            lower_count = (
                0.5
                * stats.chi2.ppf(
                    0.025,
                    2 * events
                )
            )

        upper_count = (
            0.5
            * stats.chi2.ppf(
                0.975,
                2 * (events + 1)
            )
        )

        lower_rate = (
            lower_count / person_years
        ) * 100

        upper_rate = (
            upper_count / person_years
        ) * 100

        if "patient_id" in data.columns:

            patient_ids = data.loc[
                group_data.index,
                "patient_id"
            ].nunique()

        else:

            patient_ids = len(group_data)

        results.append({

            "Group": group,

            "Patients_or_records": patient_ids,

            "Events": events,

            "Person_years": round(person_years, 2),

            "Rate_per_100_person_years": round(rate, 2),

            "95%_CI_lower": round(lower_rate, 2),

            "95%_CI_upper": round(upper_rate, 2)

        })

    if not results:
        return None, "Unable to calculate valid group rates."

    return pd.DataFrame(results), None


# ============================================================
# 18. CSV EXPORT
# ============================================================

def export_csv(data):

    """
    Convert a DataFrame into downloadable CSV bytes.
    """

    if data is None or data.empty:
        return None

    output = io.StringIO()

    data.to_csv(
        output,
        index=False
    )

    return output.getvalue().encode("utf-8-sig")


# ============================================================
# 19. ANALYSIS RESULTS EXPORT
# ============================================================

def export_analysis_results(results):

    """
    Export statistical results in a simple CSV format.
    """

    if not results:
        return None

    export_rows = []

    for analysis_name, result in results.items():

        if isinstance(result, dict):

            if "error" in result:

                export_rows.append({

                    "Analysis": analysis_name,

                    "Result": result["error"]

                })

                continue

            for key, value in result.items():

                if isinstance(value, pd.DataFrame):

                    continue

                if isinstance(value, str):

                    export_value = value

                elif isinstance(value, (int, float, np.number)):

                    export_value = value

                else:

                    export_value = str(value)

                export_rows.append({

                    "Analysis": analysis_name,

                    "Result": key,

                    "Value": export_value

                })

    if not export_rows:
        return None

    export_data = pd.DataFrame(export_rows)

    return export_csv(export_data)


# ============================================================
# 20. END OF PART 4
# ============================================================

# ============================================================
# 21. CORRECTED BINARY OUTCOME COMPARISON
# ============================================================

def binary_comparison(data, group_col, outcome_col):

    subset = data[[group_col, outcome_col]].dropna().copy()

    groups = list(subset[group_col].unique())

    if len(groups) != 2:
        return {"error": "Exactly two groups are required."}

    if not set(subset[outcome_col].unique()).issubset({0, 1}):
        return {"error": "Binary outcome must be coded as 0 and 1."}

    table = pd.crosstab(
        subset[group_col],
        subset[outcome_col]
    ).reindex(
        index=groups,
        columns=[0, 1],
        fill_value=0
    )

    if (table.sum(axis=1) == 0).any():
        return {"error": "One group has no observations."}

    contingency = table.values

    chi2, chi_p, dof, expected = chi2_contingency(
        contingency,
        correction=False
    )

    if (expected < 5).any():

        odds_ratio, p_value = fisher_exact(contingency)

        test_name = "Fisher's exact test"
        statistic = odds_ratio

    else:

        p_value = chi_p
        test_name = "Pearson chi-square test"
        statistic = chi2

    n1 = int(table.iloc[0].sum())
    n2 = int(table.iloc[1].sum())

    events1 = int(table.iloc[0][1])
    events2 = int(table.iloc[1][1])

    risk1 = events1 / n1
    risk2 = events2 / n2

    difference = risk1 - risk2

    l1, u1 = proportion_confint(
        events1, n1, method="wilson"
    )

    l2, u2 = proportion_confint(
        events2, n2, method="wilson"
    )

    # Newcombe confidence interval for independent proportions

    lower = difference - np.sqrt(
        (risk1 - l1) ** 2 + (u2 - risk2) ** 2
    )

    upper = difference + np.sqrt(
        (u1 - risk1) ** 2 + (risk2 - l2) ** 2
    )

    return {
        "test": test_name,
        "statistic": statistic,
        "p_value": p_value,
        "group_1": str(groups[0]),
        "group_2": str(groups[1]),
        "risk_1": risk1,
        "risk_2": risk2,
        "risk_difference": difference,
        "risk_difference_ci_lower": lower,
        "risk_difference_ci_upper": upper,
        "contingency_table": table
    }


# ============================================================
# 22. GENERAL ANALYSIS FUNCTION
# ============================================================

def run_analysis(data, group_col, outcome_col):

    results = {}

    if group_col not in data.columns:
        return {"error": "Selected group column not found."}

    if outcome_col not in data.columns:
        return {"error": "Selected outcome column not found."}

    if not pd.api.types.is_numeric_dtype(data[outcome_col]):
        return {"error": "Selected outcome must be numeric."}

    subset = data[[group_col, outcome_col]].dropna()

    if subset.empty:
        return {"error": "No complete observations available."}

    if subset[group_col].nunique() != 2:
        return {"error": "Exactly two comparison groups are required."}

    if subset[outcome_col].nunique() < 2:
        return {"error": "Outcome has no variation."}

    # Binary outcome
    if set(subset[outcome_col].unique()).issubset({0, 1}):

        results["Binary comparison"] = binary_comparison(
            data,
            group_col,
            outcome_col
        )

    else:

        # Welch test for continuous outcomes.
        # For count outcomes, use count regression as the
        # primary model where appropriate.

        results["Welch t-test"] = welch_test(
            data,
            group_col,
            outcome_col
        )

    return results


# ============================================================
# 23. DISPLAY STATISTICAL RESULTS
# ============================================================

def display_analysis_results(results):

    for analysis_name, result in results.items():

        st.subheader(analysis_name)

        if not isinstance(result, dict):
            st.write(result)
            continue

        if "error" in result:
            st.warning(result["error"])
            continue

        if "contingency_table" in result:

            st.write("Contingency table")
            st.dataframe(result["contingency_table"])

            st.metric(
                "P-value",
                f"{result['p_value']:.4f}"
            )

            st.write(
                f"Test used: {result['test']}"
            )

            st.write(
                f"Risk in {result['group_1']}: "
                f"{result['risk_1']:.3f}"
            )

            st.write(
                f"Risk in {result['group_2']}: "
                f"{result['risk_2']:.3f}"
            )

            st.write(
                f"Risk difference: "
                f"{result['risk_difference']:.3f}"
            )

            st.write(
                "95% CI for risk difference: "
                f"{result['risk_difference_ci_lower']:.3f} "
                "to "
                f"{result['risk_difference_ci_upper']:.3f}"
            )

        elif "t_statistic" in result:

            metrics = st.columns(3)

            metrics[0].metric(
                "P-value",
                f"{result['p_value']:.4f}"
            )

            metrics[1].metric(
                "Mean difference",
                f"{result['mean_difference']:.3f}"
            )

            metrics[2].metric(
                "Hedges' g",
                f"{result['hedges_g']:.3f}"
                if pd.notna(result["hedges_g"])
                else "Not available"
            )

            st.write(
                f"95% CI: {result['ci_lower']:.3f} "
                f"to {result['ci_upper']:.3f}"
            )

            st.write(
                f"Welch t-statistic: {result['t_statistic']:.3f}"
            )

            st.write(
                f"Group means: {result['group_1']} = "
                f"{result['mean_1']:.3f}; "
                f"{result['group_2']} = "
                f"{result['mean_2']:.3f}"
            )


# ============================================================
# 24. SIDEBAR: RESEARCH SETTINGS
# ============================================================

st.sidebar.header("Research Settings")

st.sidebar.text_area(
    "Research topic",
    key="research_topic",
    height=120
)

st.sidebar.caption(
    "The selected topic will be used by AI research modules."
)


# ============================================================
# 25. RESEARCH MODULES 1–9
# ============================================================

st.header("Research Development Modules")

st.write(
    "Use these modules to develop and review your research "
    "question, methodology, analysis plan, and proposal."
)

selected_module = st.selectbox(
    "Select research module",
    list(MODULES.keys())
)

st.subheader(selected_module)

st.write(MODULES[selected_module])

user_request = st.text_area(
    "Additional instructions or research details",
    placeholder=(
        "Enter any specific question or instructions "
        "for this research module..."
    ),
    height=120,
    key="module_user_request"
)

if st.button(
    "Generate Research Output",
    key="generate_module"
):

    with st.spinner("Generating research output..."):

        previous_outputs = "\n\n".join(
            [
                f"{name}:\n{output}"
                for name, output
                in st.session_state.module_outputs.items()
            ]
        )

        context = f"""
        Research topic:
        {st.session_state.research_topic}

        Research background:
        {st.session_state.research_context}

        Previous module outputs:
        {previous_outputs}
        """

        prompt = f"""
        MODULE:
        {selected_module}

        MODULE INSTRUCTIONS:
        {MODULES[selected_module]}

        RESEARCH TOPIC:
        {st.session_state.research_topic}

        USER'S ADDITIONAL INSTRUCTIONS:
        {user_request}

        Provide a structured, scientifically cautious,
        clearly explained response.
        """

        output = ask_groq(
            prompt,
            context
        )

        st.session_state.module_outputs[
            selected_module
        ] = output

        st.rerun()


if selected_module in st.session_state.module_outputs:

    st.subheader("Generated Output")

    st.markdown(
        st.session_state.module_outputs[selected_module]
    )


# Display all previously generated outputs

with st.expander("View all saved research module outputs"):

    if st.session_state.module_outputs:

        for name, output in st.session_state.module_outputs.items():

            st.markdown(f"### {name}")
            st.markdown(output)
            st.divider()

    else:

        st.info("No research outputs generated yet.")


if st.button(
    "Clear All Research Module Outputs",
    key="clear_modules"
):

    st.session_state.module_outputs = {}

    st.rerun()


# ============================================================
# 26. MODULE 10: STATISTICAL ANALYSIS DASHBOARD
# ============================================================

st.divider()

st.header("Module 10: Statistical Analysis & Visualization")

st.markdown(
    """
    This module allows the researcher to explore a dataset,
    perform selected statistical analyses, visualize results,
    calculate crude utilization rates, and export outputs.
    """
)


# ============================================================
# 27. DATA INPUT
# ============================================================

st.subheader("Step 1: Select Dataset")

data_option = st.radio(
    "Choose your dataset source",
    [
        "Generate synthetic demonstration data",
        "Upload CSV dataset"
    ],
    key="data_option"
)

data = None

if data_option == "Generate synthetic demonstration data":

    sample_size = st.number_input(
        "Number of synthetic records",
        min_value=50,
        max_value=5000,
        value=500,
        step=50
    )

    if st.button(
        "Generate Synthetic Dataset",
        key="generate_dataset"
    ):

        data = make_demo_data(
            n=int(sample_size)
        )

        st.session_state.active_dataset = data
        st.session_state.dataset_source = "Synthetic demonstration data"

    if st.session_state.dataset_source == "Synthetic demonstration data":

        data = st.session_state.active_dataset

else:

    uploaded_file = st.file_uploader(
        "Upload CSV file",
        type=["csv"],
        key="csv_upload"
    )

    if uploaded_file is not None:

        try:

            data = pd.read_csv(uploaded_file)

            st.session_state.active_dataset = data
            st.session_state.dataset_source = "Uploaded CSV dataset"

        except Exception as e:

            st.error(
                f"Unable to read CSV file: {str(e)}"
            )


# ============================================================
# 28. DATA OVERVIEW AND VALIDATION
# ============================================================

if data is not None:

    st.subheader("Step 2: Dataset Overview")

    valid, errors, warnings = validate_dataset(data)

    overview = dataset_summary(data)

    columns = st.columns(len(overview))

    for index, (label, value) in enumerate(overview.items()):

        columns[index].metric(
            label,
            value
        )

    st.write("Dataset preview")

    st.dataframe(
        data.head(10),
        use_container_width=True
    )

    if errors:

        st.error("Dataset validation errors:")

        for error in errors:
            st.write(f"- {error}")

    if warnings:

        st.warning("Dataset warnings:")

        for warning in warnings:
            st.write(f"- {warning}")

    if valid:

        st.success("Basic dataset validation passed.")

    st.subheader("Descriptive Statistics")

    desc = descriptive_statistics(data)

    if not desc.empty:

        st.dataframe(
            desc,
            use_container_width=True
        )


    # ========================================================
    # 29. STATISTICAL COMPARISON
    # ========================================================

    st.divider()

    st.subheader("Step 3: Statistical Comparison")

    numeric_columns = list(
        data.select_dtypes(include=np.number).columns
    )

    if len(numeric_columns) == 0:

        st.warning(
            "No numeric variables are available."
        )

    elif "sex" not in data.columns:

        st.warning(
            "A 'sex' column is required for the default "
            "male/female comparison. You can use another "
            "categorical group variable in the plot section."
        )

    else:

        group_options = list(
            data.select_dtypes(
                include=["object", "category", "string"]
            ).columns
        )

        if group_options:

            group_col = st.selectbox(
                "Grouping variable",
                group_options,
                index=(
                    group_options.index("sex")
                    if "sex" in group_options
                    else 0
                ),
                key="analysis_group"
            )

        else:

            group_col = None

        outcome_col = st.selectbox(
            "Select outcome variable",
            numeric_columns,
            index=(
                numeric_columns.index("kidney_encounters")
                if "kidney_encounters" in numeric_columns
                else 0
            ),
            key="analysis_outcome"
        )

        if group_col is not None:

            if st.button(
                "Run Statistical Comparison",
                key="run_comparison"
            ):

                results = run_analysis(
                    data,
                    group_col,
                    outcome_col
                )

                st.session_state.analysis_results = results

            if st.session_state.analysis_results is not None:

                display_analysis_results(
                    st.session_state.analysis_results
                )


    # ========================================================
    # 30. COUNT REGRESSION MODELS
    # ========================================================

    st.divider()

    st.subheader("Step 4: Count Regression Models")

    count_outcomes = [
        column
        for column in numeric_columns
        if column in [
            "kidney_encounters",
            "kidney_imaging_count",
            "nephrology_visit_count"
        ]
    ]

    if count_outcomes and "followup_years" in data.columns:

        count_outcome = st.selectbox(
            "Select count outcome",
            count_outcomes,
            key="count_outcome"
        )

        if "sex" in data.columns:

            if st.button(
                "Run Poisson Regression",
                key="run_poisson"
            ):

                with st.spinner("Running Poisson regression..."):

                    poisson_result = poisson_regression(
                        data,
                        count_outcome
                    )

                    st.session_state.poisson_result = poisson_result

            if "poisson_result" in st.session_state:

                result = st.session_state.poisson_result

                if "error" in result:

                    st.error(result["error"])

                else:

                    st.write("Poisson regression results")

                    st.dataframe(
                        result["results"],
                        use_container_width=True
                    )

                    st.write(
                        "Pearson dispersion statistic: "
                        f"{result['pearson_dispersion']:.3f}"
                    )

                    st.caption(
                        "A dispersion value substantially above 1 "
                        "may indicate overdispersion. This is a "
                        "diagnostic signal, not an automatic rule "
                        "for selecting a model."
                    )

            if st.button(
                "Run Negative Binomial Regression",
                key="run_nb"
            ):

                with st.spinner(
                    "Running Negative Binomial regression..."
                ):

                    nb_result = negative_binomial_regression(
                        data,
                        count_outcome
                    )

                    st.session_state.nb_result = nb_result

            if "nb_result" in st.session_state:

                result = st.session_state.nb_result

                if "error" in result:

                    st.error(result["error"])

                else:

                    st.write(
                        "Negative Binomial regression results"
                    )

                    st.dataframe(
                        result["results"],
                        use_container_width=True
                    )

                    st.write(
                        f"Estimated alpha: {result['alpha']:.4f}"
                    )

        else:

            st.info(
                "Sex variable is required for this demonstration."
            )

    else:

        st.info(
            "Count regression requires a supported count outcome "
            "and a positive follow-up duration variable."
        )


    # ========================================================
    # 31. VISUALIZATION
    # ========================================================

    st.divider()

    st.subheader("Step 5: Data Visualization")

    categorical_columns = list(
        data.select_dtypes(
            include=["object", "category", "string"]
        ).columns
    )

    if categorical_columns and numeric_columns:

        plot_group = st.selectbox(
            "Select grouping variable for visualization",
            categorical_columns,
            key="plot_group"
        )

        plot_outcome = st.selectbox(
            "Select visualization outcome",
            numeric_columns,
            key="plot_outcome"
        )

        plot_type = st.selectbox(
            "Choose graph type",
            [
                "Box plot",
                "Histogram",
                "Bar chart"
            ],
            key="plot_type"
        )

        fig, error = make_plot(
            data,
            plot_type,
            plot_group,
            plot_outcome
        )

        if error:

            st.warning(error)

        elif fig is not None:

            st.pyplot(fig)

            plt.close(fig)

    else:

        st.info(
            "Categorical and numeric variables are required "
            "for the selected visualizations."
        )


    # ========================================================
    # 32. CRUDE UTILIZATION RATES
    # ========================================================

    st.divider()

    st.subheader("Step 6: Crude Utilization Rates")

    rate_event_options = [
        column
        for column in numeric_columns
        if column in [
            "kidney_encounters",
            "kidney_imaging_count",
            "nephrology_visit_count"
        ]
    ]

    if (
        rate_event_options
        and "followup_years" in data.columns
        and "sex" in data.columns
    ):

        rate_event = st.selectbox(
            "Select event count",
            rate_event_options,
            key="rate_event"
        )

        rate_table, rate_error = calculate_crude_rates(
            data,
            group_col="sex",
            event_col=rate_event,
            followup_col="followup_years"
        )

        if rate_error:

            st.warning(rate_error)

        else:

            st.dataframe(
                rate_table,
                use_container_width=True
            )

            st.caption(
                "Rates are expressed per 100 person-years. "
                "These are crude rates and are not adjusted "
                "for age, comorbidities, or other confounders."
            )

    else:

        st.info(
            "Crude rates require sex, a supported event-count "
            "variable, and follow-up duration."
        )


    # ========================================================
    # 33. EXPORT DATA AND RESULTS
    # ========================================================

    st.divider()

    st.subheader("Step 7: Export Results")

    dataset_csv = export_csv(data)

    if dataset_csv is not None:

        st.download_button(
            label="Download Dataset CSV",
            data=dataset_csv,
            file_name="TSC_research_dataset.csv",
            mime="text/csv"
        )

    if st.session_state.analysis_results is not None:

        results_csv = export_analysis_results(
            st.session_state.analysis_results
        )

        if results_csv is not None:

            st.download_button(
                label="Download Statistical Results CSV",
                data=results_csv,
                file_name="TSC_statistical_results.csv",
                mime="text/csv"
            )


# ============================================================
# 34. SCIENTIFIC DISCLAIMER
# ============================================================

st.divider()

st.subheader("Research and Interpretation Disclaimer")

st.markdown(
    """
    - This application is a research-support prototype.
    - AI-generated recommendations require expert review.
    - Synthetic data are not real patient records.
    - Unadjusted comparisons do not establish causation.
    - Regression models require assessment of assumptions,
      model fit, and data quality.
    - Clinical outcome definitions must be validated against
      appropriate clinical and coding standards.
    - Real-world research requires appropriate ethics approval,
      data governance, and privacy safeguards.
    """
)

st.caption(
    "Quantitative Research Assistant | Research planning, "
    "statistical demonstration, and methodological review"
)

# ============================================================
# END OF PART 5 — END OF APP
# ============================================================
