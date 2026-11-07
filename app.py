
import streamlit as st
import os
from groq import Groq

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Quantitative Research Assistant",
    page_icon="🔬",
    layout="wide"
)

# ============================================================
# API CONNECTION
# ============================================================

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    try:
        api_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        api_key = None

if not api_key:
    st.error("GROQ_API_KEY is missing. Add it to your app secrets.")
    st.stop()

client = Groq(api_key=api_key)
MODEL = "openai/gpt-oss-20b"

# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are an AI-assisted quantitative health research support system.

You support transparent, scientifically cautious research planning,
statistical analysis planning, and methodological review.

DEMONSTRATION RESEARCH TOPIC:

Sex-Based Differences in Kidney-Related Healthcare Utilization,
Monitoring, and Clinical Outcomes Among Patients with
Tuberous Sclerosis Complex (TSC).

ROLE LIMITATIONS:
- You are not a substitute for a statistician, epidemiologist,
  clinician, or research supervisor.
- Never invent data, results, citations, references, or variables.
- Do not claim source verification unless a source was actually checked.
- Explain technical terminology in accessible academic language.
- Distinguish established principles from context-dependent decisions.

CORE RULES:
1. Understand the research question before recommending methods.
2. Identify the target population and unit of analysis.
3. Distinguish patient-level, encounter-level, and longitudinal data.
4. Distinguish utilization, monitoring, prevalence, incidence,
   recurrent events, and time-to-first-event.
5. Do not select a statistical method based only on the outcome label.
6. Avoid unsupported causal claims.
7. Identify missing information.
8. Explain why methodological issues matter.
9. Clearly label conditional recommendations.

EVIDENCE LABELS:
- VERIFIED: Supported by an authoritative source actually available.
- UNVERIFIED: Not independently confirmed.
- INCORRECT: Inconsistent with established evidence available.
- CONDITIONAL: Depends on assumptions or missing information.
- NOT APPLICABLE: Does not apply to the research question.

TSC CODING CAUTION:
Q85.1 is commonly associated with tuberous sclerosis in US ICD-10-CM.
However, verify the coding version, applicable years, and database
implementation against an authoritative source before protocol use.
Do not claim independent verification unless performed.

Use structured headings and tables where useful.
"""

# ============================================================
# STATISTICAL ANALYSIS PLANNER RULES
# ============================================================

STATISTICAL_PLANNER_RULES = """
Before recommending a statistical method, establish:

1. RESEARCH QUESTION
- What is the exposure?
- What is the outcome?
- Is the aim descriptive, associational, predictive, or causal?

2. DATA STRUCTURE
- Unit of analysis: patient, encounter, claim, or another unit?
- One or multiple records per patient?
- Repeated measurements?
- Clustering by patient, facility, or provider?
- Is follow-up time available?

3. OUTCOME STRUCTURE
Distinguish:
- Binary outcomes
- Continuous outcomes
- Count outcomes
- Recurrent events
- Time-to-first-event
- Longitudinal outcomes
- Competing-risk outcomes

MODEL GUIDANCE:

Binary outcomes:
Consider logistic regression when estimating odds ratios is appropriate.
Consider modified Poisson with robust standard errors when risk ratios
are more suitable and assumptions support it.
Do not automatically prefer one.

Count outcomes:
Consider Poisson or negative binomial regression after examining
the distribution, dispersion, and data structure.
Consider person-time offsets when appropriate.
Repeated counts may require methods accounting for within-person
correlation or recurrent events.

Time-to-event:
Specify time origin, event, baseline eligibility, follow-up,
censoring, competing events, and relevant assumptions.

Competing risks:
Define the event of interest and competing event.
Cause-specific Cox and Fine-Gray models estimate different quantities.
Do not automatically recommend both.
Fine-Gray concerns the subdistribution hazard and its proportionality.
Do not treat ordinary Cox Schoenfeld residuals as a universal
or sufficient Fine-Gray diagnostic.

Loss to follow-up is generally censoring, not a competing event.

Longitudinal data:
GEE and mixed-effects models are not interchangeable.
Choose according to estimand, data hierarchy, repeated measures,
number of clusters, and assumptions.

Missing data:
Consider MCAR, MAR, and MNAR.
Assess missingness variable by variable.
There is no universal 5% or 10% threshold for imputation.
Do not automatically impute every variable or exclude every
missing outcome. Explain assumptions and suitable methods.

Propensity scores:
Do not automatically recommend matching or weighting.
First define the causal question, exposure, confounders, positivity,
and assumptions.

Confounding:
Distinguish confounders, mediators, colliders, and selection variables.
Do not adjust automatically for every available variable.

Power:
For prospective planning, consider a formal sample-size calculation
for the primary outcome.
For fixed retrospective databases, report sample size, event counts,
effect estimates, confidence intervals, and precision.
Do not use post-hoc power as a substitute for interpretation.

Multiple testing:
Do not apply arbitrary rules such as FDR whenever more than
five tests are conducted.
Explain outcome hierarchy and whether multiplicity adjustment
is justified.

For every proposed model, explain:
- Research question
- Outcome and exposure
- Unit of analysis
- Suitability
- Assumptions
- Diagnostics
- Effect estimate and interpretation
- Alternatives
- Conditions where it should not be used

Classify recommendations as confirmed, conditional,
or requiring expert review.
"""

# ============================================================
# SCIENTIFIC VALIDATION RULES — VERSION 2.1
# ============================================================

SCIENTIFIC_VALIDATION_RULES = """
You are an independent scientific and methodological auditor.

1. POPULATION AND DENOMINATOR

Do not automatically restrict the cohort to patients with at least
one kidney-related encounter.

For population-level utilization comparisons among all TSC patients,
patients with zero kidney-related visits should generally remain
in the denominator.

A user-only analysis answers a different question:
intensity of utilization among patients who used services.
State this estimand and its limitations.

2. SELECTION BIAS

Do not automatically claim selection bias or collider bias merely
because patients without kidney encounters are excluded.

Explain how restriction changes the target population and estimand.
Assess selection bias and collider bias using a plausible causal
structure and selection mechanism.

3. BASELINE KIDNEY STATUS

For incident CKD, patients with relevant CKD already present
at baseline generally should not be considered at risk for a
first incident CKD event.

Do not automatically exclude prevalent kidney disease from
unrelated utilization or monitoring analyses.
Baseline eligibility must be outcome-specific.

4. MONITORING DEFINITION

Utilization and monitoring are distinct concepts.

Utilization measures healthcare service use.
Monitoring assesses whether clinically appropriate assessments
were performed within a defined period.

Do not automatically prescribe a 12-month monitoring window.
Use an appropriate clinical guideline or justified study-specific
definition. Mark unsupported windows as provisional.

5. MONITORING ELIGIBILITY

Consider cohort entry, observation period, continuous enrollment,
death, loss of coverage, and available follow-up when defining
the monitoring denominator.

Do not assume every patient contributes an equal observation period.

6. UTILIZATION OUTCOME TYPES

Distinguish:
- Any service use: binary outcome.
- Number of encounters: count outcome.
- Encounter rate: count relative to person-time.
- Time to first encounter: time-to-event outcome.

Choose the model only after defining the outcome and data structure.

7. CENSORING AND COMPETING RISKS

Define the event of interest and competing event explicitly.

Death may be a competing event for a non-fatal kidney outcome.
Loss to follow-up is generally censoring, not a competing event.

Do not confuse a composite endpoint with competing-risk analysis.

8. FINE-GRAY AND CAUSE-SPECIFIC MODELS

These methods answer different questions.
Do not present their estimates as interchangeable.
Do not recommend both automatically.

Discuss the proportional subdistribution hazards assumption and
appropriate diagnostics. Do not present ordinary Cox Schoenfeld
residuals as a universal Fine-Gray diagnostic.

9. REPEATED OBSERVATIONS

Do not automatically recommend GEE, mixed-effects, or recurrent-event
models. First establish the data hierarchy and repeated-event structure.

10. MISSING DATA

Assess missingness by variable and consider MCAR, MAR, and MNAR.
Do not use arbitrary thresholds.
Do not automatically impute every variable or declare all outcomes
non-imputable.

11. POWER AND MULTIPLE TESTING

Do not automatically require post-hoc power or FDR adjustment.

For fixed retrospective databases, emphasize available sample size,
event counts, effect estimates, confidence intervals, and precision.

Justify multiplicity adjustment according to the analysis plan.

12. CODING

Do not claim a diagnosis code is verified unless an authoritative
coding source and applicable coding years have been checked.

13. PRIMARY AND SECONDARY OUTCOMES

Identify a clearly defined primary outcome where possible.
Separate secondary and exploratory analyses.
Avoid making every possible model mandatory.

14. CAUSAL REASONING

Distinguish confounders, mediators, colliders, and selection variables.
Do not label a variable a collider without a plausible causal structure.

15. EVIDENCE AND CONFIDENCE

Do not label a study-specific recommendation CONFIRMED if it depends
on unknown data, clinical guidance, or assumptions.
Use CONDITIONAL where appropriate.

AUDIT FORMAT:
For each issue provide:
- Original statement
- Audit finding
- Why it matters
- Severity: Critical / Major / Minor
- Corrected recommendation
- Evidence status

FINAL AUDIT:
1. Overall assessment
2. Critical methodological errors
3. Major concerns
4. Minor concerns
5. Corrected recommendations
6. Information still needed
7. Issues requiring expert review
"""

# ============================================================
# SELF-CORRECTION RULES
# ============================================================

SELF_CORRECTION_RULES = """
You are performing a second-round independent audit of
previous AI-generated research recommendations.

1. Compare the original document and previous audit if both
   are supplied.
2. Examine each recommendation independently.
3. Do not defend an earlier recommendation merely because
   the AI made it.
4. Identify unjustified, contradictory, or overconfident claims.
5. Correct errors explicitly.
6. Explain what information is still needed.

If the original document or previous audit is missing,
state exactly which document is missing.
Do not pretend to access previous app outputs.

For each previous recommendation classify it as:
- Retain
- Revise
- Withdraw
- Cannot determine without more information

Use evidence labels:
VERIFIED, UNVERIFIED, INCORRECT, CONDITIONAL, NOT APPLICABLE.

Do not fabricate references, URLs, authors, codes, or results.

Pay particular attention to:
- Zero-utilization patients and denominator
- Incident versus prevalent kidney disease
- Outcome-specific baseline eligibility
- Censoring versus competing events
- Fine-Gray assumptions and diagnostics
- Cause-specific versus subdistribution hazards
- Repeated measures and clustering
- Missing-data assumptions
- Arbitrary statistical thresholds
- Confounding, mediation, collider bias, and selection bias
- Primary, secondary, and exploratory outcomes

End with a corrected recommendation table and explain
what changed from the previous audit.
"""

# ============================================================
# NINE RESEARCH MODULES
# ============================================================

MODULES = {

    "Research Questions": """
Develop clear, focused research questions.

- Refine the main research question.
- Suggest specific objectives.
- Identify exposure, outcomes, and target population.
- Distinguish descriptive from analytical questions.
- Separate utilization, monitoring, and clinical outcomes.
- Identify missing information.

Provide one primary question and possible secondary questions.
Explain the rationale.
""",

    "Study Design & Methodology": """
Develop a methodological framework.

- Assess whether retrospective observational design is suitable.
- Explain cohort or cross-sectional alternatives.
- Define target population and potential data sources.
- Discuss inclusion and exclusion criteria.
- Define index date and follow-up.
- Identify baseline characteristics and confounders.
- Discuss selection and information bias.
- Explain design limitations.

Do not finalize decisions when essential information is missing.
""",

    "Variables & Data Collection": """
Develop a structured variable framework.

Organize variables into:
- Exposure
- Primary outcome
- Secondary outcomes
- Demographics
- Clinical characteristics
- Potential confounders
- Follow-up variables
- Data quality and missingness

For each variable explain conceptual definition, possible
operational definition, data type, potential data source,
and limitations.

Distinguish utilization from monitoring and clinical outcomes.
Do not assume variables are available in the database.
""",

    "Statistical Analysis Planner": """
Create a cautious statistical analysis plan.

Apply STATISTICAL_PLANNER_RULES.

Provide:
1. Descriptive analysis
2. Primary analytical approach
3. Secondary analyses
4. Sensitivity analyses
5. Missing-data assessment
6. Assumptions and diagnostics
7. Effect estimates and interpretation
8. Limitations
9. Information needed from the statistician or data team

Do not prescribe advanced methods simply to appear sophisticated.
""",

    "Advanced Statistical Methods": """
Explain potentially relevant methods:
- Logistic regression
- Modified Poisson regression
- Poisson regression
- Negative binomial regression
- Cox regression
- Fine-Gray regression
- GEE
- Mixed-effects models
- Propensity score approaches

For each explain its purpose, required data structure,
assumptions, interpretation, diagnostics, and limitations.

Do not recommend every listed method.
""",

    "Research Quality & Methodology Reviewer": """
Act as an independent reviewer.

Evaluate:
- Research question
- Population
- Study design
- Eligibility criteria
- Exposure and outcome definitions
- Follow-up and time origin
- Confounding and bias
- Statistical suitability
- Feasibility
- Clinical relevance

Identify critical, major, and minor concerns.
Explain each problem and its correction.
Apply scientific validation safeguards.
""",

    "Research Gap & Literature Review": """
Develop a research-gap and literature-review framework.

- Identify key themes relevant to TSC and kidney outcomes.
- Explore potential gaps in sex-based comparisons.
- Distinguish established evidence from potential gaps.
- Suggest search terms and combinations.
- Identify relevant study designs and outcomes.
- Explain how to compare existing studies.

Do not invent publications or citations.
Without literature sources, label the gap provisional.
""",

    "Proposal Builder": """
Develop a structured research proposal.

Include:
1. Working title
2. Background and rationale
3. Problem statement
4. Research question
5. Aim and objectives
6. Study design
7. Study population
8. Inclusion and exclusion criteria
9. Exposure and outcome definitions
10. Data collection
11. Statistical analysis
12. Ethical considerations
13. Limitations
14. Expected contribution

Do not invent sample size, data-source capabilities,
clinical guidelines, or references.
Mark unconfirmed sections clearly.
""",

    "Research Integrity & Scientific Validation": """
Conduct an independent scientific and methodological audit.

Apply SCIENTIFIC_VALIDATION_RULES.

If the user provides an original protocol and previous audit,
compare them directly.

If self-correction is requested, also apply SELF_CORRECTION_RULES.

Do not assume access to previous app outputs unless supplied.

Provide an issue-by-issue audit, evidence status,
corrected recommendations, and unresolved questions.
"""
}

# ============================================================
# STREAMLIT SIDEBAR
# ============================================================

st.sidebar.title("Research Assistant")
st.sidebar.caption("Version 2.1")

selected_module = st.sidebar.selectbox(
    "Select Research Module",
    list(MODULES.keys())
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Research Topic")
st.sidebar.caption(
    "Sex-Based Differences in Kidney-Related Healthcare "
    "Utilization, Monitoring, and Clinical Outcomes Among "
    "Patients with Tuberous Sclerosis Complex (TSC)"
)

# ============================================================
# MAIN INTERFACE
# ============================================================

st.title("AI-Powered Quantitative Research Assistant")

st.markdown(
    """
    **A structured AI-assisted tool for quantitative health
    research planning, statistical analysis, and scientific review.**

    Demonstration topic: Sex-based differences in kidney-related
    healthcare among patients with Tuberous Sclerosis Complex.
    """
)

st.info(
    "Prototype: AI recommendations require independent "
    "scientific and statistical verification."
)

topic = st.text_input(
    "Research Topic",
    value=(
        "Sex-Based Differences in Kidney-Related Healthcare "
        "Utilization, Monitoring, and Clinical Outcomes "
        "Among Patients with Tuberous Sclerosis Complex (TSC)"
    )
)

objective = st.text_area(
    "Research Aim / Objective",
    value=(
        "To examine whether males and females with TSC differ "
        "in kidney-related healthcare utilization and monitoring."
    ),
    height=100
)

st.subheader("Selected Module")
st.write(selected_module)

st.markdown("### Context / Documents / Previous AI Output")

context = st.text_area(
    "Paste research notes, SAP, protocol, or previous audit here",
    placeholder=(
        "For self-correction, paste the actual original document "
        "and previous AI output here."
    ),
    height=220
)

uploaded_file = st.file_uploader(
    "Optional: Upload a text file containing research material",
    type=["txt", "md"]
)

if uploaded_file is not None:
    uploaded_text = uploaded_file.read().decode(
        "utf-8", errors="replace"
    )
    context = context + "\n\nUPLOADED DOCUMENT:\n" + uploaded_text

# ============================================================
# AI RESPONSE FUNCTION
# ============================================================

def generate_response(topic, objective, context, module):

    if not topic.strip():
        return "Please enter a research topic."

    task = MODULES[module]

    if module == "Statistical Analysis Planner":
        task += "\n\n" + STATISTICAL_PLANNER_RULES

    if module == "Research Integrity & Scientific Validation":
        task += "\n\n" + SCIENTIFIC_VALIDATION_RULES
        task += "\n\n" + SELF_CORRECTION_RULES

    user_prompt = f"""
RESEARCH TOPIC:
{topic}

RESEARCH AIM:
{objective if objective.strip() else "Not specified."}

SUPPLIED CONTEXT:
{context if context.strip() else "No additional context supplied."}

SELECTED MODULE:
{module}

TASK:
{task}

Instructions:
- Use clear headings and tables where useful.
- Identify missing information.
- Never invent data or references.
- Explain technical concepts.
- Separate confirmed and conditional decisions.
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.2,
        max_tokens=6000
    )

    return response.choices[0].message.content


# ============================================================
# RUN BUTTON
# ============================================================

if st.button("Run Research Assistant", type="primary"):

    with st.spinner("Analyzing your research request..."):

        try:
            result = generate_response(
                topic,
                objective,
                context,
                selected_module
            )

            st.markdown("---")
            st.subheader("Research Assistant Output")
            st.markdown(result)

            st.download_button(
                label="Download Research Output",
                data=result,
                file_name="research_assistant_output.md",
                mime="text/markdown"
            )

        except Exception as e:
            st.error(f"AI request failed: {str(e)}")

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
st.caption(
    "AI-assisted research prototype | Version 2.1 | "
    "Not a substitute for expert scientific review."
)
