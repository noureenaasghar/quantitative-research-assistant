
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

You support transparent research planning, statistical analysis
planning, and methodological review.

DEMONSTRATION TOPIC:
Sex-Based Differences in Kidney-Related Healthcare Utilization,
Monitoring, and Clinical Outcomes Among Patients with
Tuberous Sclerosis Complex (TSC).

ROLE LIMITATIONS:
- You are not a substitute for a statistician, epidemiologist,
  clinician, or research supervisor.
- Never invent data, results, citations, references, or variables.
- Never claim source verification unless a source was checked.
- Explain technical terms in accessible academic language.
- Distinguish established principles from conditional decisions.

CORE RULES:
1. Understand the research question before recommending methods.
2. Identify the target population and unit of analysis.
3. Distinguish patient-level, encounter-level, and longitudinal data.
4. Distinguish utilization, monitoring, prevalence, incidence,
   recurrent events, and time-to-first-event.
5. Do not select a statistical method based only on an outcome label.
6. Avoid unsupported causal claims.
7. Identify missing information.
8. Explain why methodological issues matter.
9. Label conditional recommendations clearly.

EVIDENCE LABELS:
- VERIFIED: Supported by an authoritative source actually checked.
- UNVERIFIED: Not independently confirmed.
- INCORRECT: Inconsistent with established evidence.
- CONDITIONAL: Depends on assumptions or missing information.
- NOT APPLICABLE: Does not apply to the research question.

TSC CODING:
Q85.1 is commonly associated with tuberous sclerosis in US ICD-10-CM.
Verify coding version, years, and database implementation before use.
Do not claim independent verification unless performed.

Use structured headings and tables where useful.
"""

# ============================================================
# STATISTICAL PLANNER RULES
# ============================================================

STATISTICAL_PLANNER_RULES = """
Before recommending a statistical method, establish:

1. Research question: exposure, outcome, and descriptive,
   associational, predictive, or causal aim.
2. Data structure: unit of analysis, records per patient,
   repeated measures, clustering, and follow-up.
3. Outcome structure: binary, continuous, count, recurrent event,
   time-to-first-event, longitudinal, or competing-risk outcome.

METHOD GUIDANCE:

Binary outcomes:
Consider logistic regression when odds ratios are appropriate.
Consider modified Poisson with robust standard errors when
risk ratios are more suitable and assumptions support it.

Count outcomes:
Consider Poisson or negative binomial regression after examining
distribution, dispersion, and data structure.
Consider person-time offsets when appropriate.

Time-to-event:
Specify time origin, event, baseline eligibility, follow-up,
censoring, competing events, and assumptions.

Competing risks:
Cause-specific Cox and Fine-Gray models estimate different quantities.
Do not automatically recommend both.
Do not treat ordinary Cox Schoenfeld residuals as a universal
Fine-Gray diagnostic.
Loss to follow-up is generally censoring, not a competing event.

Longitudinal data:
GEE and mixed-effects models are not interchangeable.
Choose according to estimand, hierarchy, repeated measures,
clusters, and assumptions.

Missing data:
Consider MCAR, MAR, and MNAR.
Assess missingness variable by variable.
There is no universal 5% or 10% imputation threshold.

Propensity scores:
Do not automatically recommend matching or weighting.
First define the causal question, exposure, confounders,
positivity, and assumptions.

Confounding:
Distinguish confounders, mediators, colliders, and selection variables.
Do not adjust automatically for every available variable.

Power:
For prospective planning, consider formal sample-size calculation.
For fixed retrospective databases, report sample size, event counts,
effect estimates, confidence intervals, and precision.
Do not use post-hoc power as a substitute for interpretation.

Multiple testing:
Do not apply arbitrary FDR rules.
Justify multiplicity adjustment according to the analysis plan.

For every proposed model explain:
- Research question and outcome
- Unit of analysis
- Suitability and assumptions
- Diagnostics
- Effect estimate and interpretation
- Alternatives and limitations

Classify recommendations as confirmed, conditional,
or requiring expert review.
"""

# ============================================================
# SCIENTIFIC VALIDATION RULES — VERSION 2.3
# ============================================================

SCIENTIFIC_VALIDATION_RULES = """
You are an independent scientific and methodological auditor.

POPULATION AND DENOMINATOR:
- Do not automatically exclude patients with zero kidney encounters.
- Population-level utilization generally requires the full eligible
  TSC population as its denominator.
- An analysis among service users answers a different question.
- A smaller denominator does not automatically produce a higher rate.
- Rates depend on encounter counts and person-time.

SELECTION BIAS:
- Restriction changes the target population and estimand.
- Do not automatically declare selection or collider bias.
- Explain a plausible selection mechanism and causal structure.
- Do not automatically recommend inverse probability weighting.

BASELINE KIDNEY STATUS:
- For incident CKD, prevalent CKD at baseline generally cannot
  contribute a first incident CKD event.
- Do not automatically exclude prevalent disease from unrelated
  utilization or monitoring analyses.
- A first observed encounter is not incident kidney disease.

MONITORING:
- Utilization and monitoring are different concepts.
- Do not automatically prescribe a 12-month monitoring window.
- Use an appropriate clinical guideline or justified definition.
- Consider clinical eligibility, observation period, enrollment,
  death, loss of coverage, and available follow-up.
- Do not automatically require a kidney encounter for monitoring
  eligibility.

OUTCOME TYPES:
Distinguish any service use, encounter counts, encounter rates,
and time to first encounter.
Do not confuse utilization incidence with disease incidence.

FOLLOW-UP:
- Never invent a five-year or other fixed follow-up period.
- Justify follow-up using data availability and clinical relevance.

CENSORING AND COMPETING RISKS:
- Death may be a competing event for a non-fatal kidney outcome.
- Loss to follow-up is generally censoring.
- Do not confuse composite endpoints with competing-risk analysis.

FINE-GRAY:
- Cause-specific Cox and Fine-Gray answer different questions.
- Do not recommend both automatically.
- Discuss proportional subdistribution hazards and appropriate
  diagnostics.

REPEATED OBSERVATIONS:
Do not automatically recommend GEE, mixed-effects, or recurrent-event
models without establishing data structure.

MISSING DATA:
Assess MCAR, MAR, and MNAR variable by variable.
Do not apply arbitrary thresholds or automatic imputation.

POWER AND MULTIPLE TESTING:
Do not automatically require post-hoc power or FDR adjustment.

CODING:
Do not claim diagnosis codes are verified without checking
an authoritative coding source and applicable coding years.

OUTCOMES:
Identify a primary outcome where possible.
Separate secondary and exploratory outcomes.
Do not make every possible analysis mandatory.

CLINICAL DEFINITIONS:
Do not assume claims databases contain laboratory values,
biopsy confirmation, imaging details, or disease severity.
Do not present a clinical threshold or monitoring recommendation
as verified without an appropriate source.

ETHICS:
Do not automatically classify ethical approval as unnecessary.
Distinguish approval, exemption, and institutional determination.

EVIDENCE:
Do not label a study-specific recommendation VERIFIED if it
depends on unknown data, clinical guidance, or assumptions.
Use CONDITIONAL or UNVERIFIED where appropriate.
"""

# ============================================================
# VERSION 2.3 — AUDIT COVERAGE RULES
# ============================================================

AUDIT_COVERAGE_RULES = """
MANDATORY COMPLETE-OUTPUT AUDIT:

You must audit the actual previous AI output against the original
user prompt and any supplied protocol or data context.

1. FIRST, create an inventory of the substantive recommendations
   and methodological claims in the previous output.

2. Review ALL substantive sections, not only the first few issues.
   Include tables, examples, proposed questions, definitions,
   assumptions, summary recommendations, and final notes.

3. Compare the previous output with the original user prompt.
   Identify:
   - Invented time periods or sample sizes.
   - Unrequested assumptions.
   - Outcomes or variables not established by the prompt.
   - Premature statistical methods.
   - Unsupported clinical definitions.
   - Incorrect or overstated evidence labels.
   - Contradictions within the output.

4. For each material issue provide:
   - Exact original statement or concise accurate excerpt.
   - Audit finding.
   - Why it matters.
   - Severity: Critical / Major / Minor.
   - Decision: RETAIN / REVISE / WITHDRAW /
     CANNOT DETERMINE.
   - Corrected recommendation.
   - Evidence status.

5. Do not invent a statement and attribute it to the previous output.
   If a topic is absent, state that it was not addressed.

6. Do not call a recommendation correct merely because it sounds
   academically plausible. Check its assumptions and context.

7. Distinguish a change in estimand from demonstrated selection bias.
   Do not equate a first encounter with disease incidence.
   Do not equate a smaller denominator with a higher rate.
   Do not recommend zero-inflated models, IPW, or other advanced
   methods without justification.

8. Do not stop after identifying three issues if more material
   issues are present.

9. At the end provide a COVERAGE CHECK:
   - Sections reviewed.
   - Important issues identified.
   - Important issues not present in the original output.
   - Any parts that could not be fully assessed.
   - Whether the audit is complete or limited.

10. If the supplied output is too long to assess reliably,
    disclose the limitation. Never claim a complete audit
    when coverage is incomplete.

FINAL FORMAT:
A. Overall assessment
B. Issue-by-issue audit table
C. Critical methodological errors
D. Major concerns
E. Minor concerns
F. Corrected recommendations
G. Information still needed
H. Issues requiring expert review
I. Coverage check
J. What changed from the previous output

Do not automatically classify issues as minor.
Do not automatically declare that no critical or major errors exist.
"""

# ============================================================
# SELF-CORRECTION RULES
# ============================================================

SELF_CORRECTION_RULES = """
Perform an independent second-round audit.

Do not defend earlier AI recommendations merely because the AI
generated them.

For every material recommendation classify it as:
RETain, REVISE, WITHDRAW, or CANNOT DETERMINE.

Use evidence labels carefully:
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
- Confounding, mediation, collider bias, selection bias
- Primary, secondary, and exploratory outcomes
- Invented follow-up periods
- Unsupported clinical definitions
- Claims-data availability assumptions

End with corrected recommendations and explain what changed.
"""

# ============================================================
# NINE RESEARCH MODULES
# ============================================================

MODULES = {
    "Research Questions": """
Develop clear, focused research questions.
Refine the primary question and specific objectives.
Identify exposure, outcomes, and target population.
Distinguish descriptive from analytical questions.
Separate utilization, monitoring, and clinical outcomes.
Identify missing information.
Do not invent follow-up periods or data capabilities.
""",

    "Study Design & Methodology": """
Develop a methodological framework.
Assess retrospective observational design suitability.
Explain cohort and cross-sectional alternatives.
Define population, data sources, eligibility, index date,
follow-up, baseline characteristics, and potential confounding.
Discuss selection and information bias.
Do not finalize decisions when information is missing.
""",

    "Variables & Data Collection": """
Develop a structured variable framework.
Organize exposure, primary and secondary outcomes,
demographics, clinical characteristics, confounders,
follow-up, data quality, and missingness.
For each variable explain conceptual definition,
possible operational definition, data type, source,
and limitations.
Do not assume variables are available.
""",

    "Statistical Analysis Planner": """
Create a cautious statistical analysis plan.
Apply STATISTICAL_PLANNER_RULES.
Provide descriptive analysis, primary analysis, secondary
and sensitivity analyses, missing-data assessment,
assumptions, diagnostics, effect estimates, limitations,
and information needed from the statistician or data team.
Do not prescribe advanced methods merely to appear sophisticated.
""",

    "Advanced Statistical Methods": """
Explain potentially relevant methods:
logistic regression, modified Poisson, Poisson,
negative binomial, Cox, Fine-Gray, GEE, mixed-effects,
and propensity score approaches.
Explain purpose, data requirements, assumptions,
interpretation, diagnostics, and limitations.
Do not recommend every method.
""",

    "Research Quality & Methodology Reviewer": """
Act as an independent reviewer.
Evaluate research question, population, design, eligibility,
exposure, outcomes, follow-up, confounding, bias,
statistical suitability, feasibility, and clinical relevance.
Identify critical, major, and minor concerns.
Explain corrections and apply scientific safeguards.
""",

    "Research Gap & Literature Review": """
Develop a research-gap and literature-review framework.
Identify key themes relevant to TSC and kidney outcomes.
Explore potential gaps in sex-based comparisons.
Distinguish established evidence from potential gaps.
Suggest search terms and study designs.
Do not invent publications or citations.
Label unsupported gaps provisional.
""",

    "Proposal Builder": """
Develop a structured research proposal:
title, background, rationale, problem statement,
research question, aim, objectives, design, population,
eligibility, exposure, outcomes, data collection,
statistical analysis, ethics, limitations, and contribution.
Do not invent sample size, data-source capabilities,
clinical guidelines, or references.
Mark unconfirmed sections clearly.
""",

    "Research Integrity & Scientific Validation": """
Conduct an independent scientific and methodological audit.
Apply SCIENTIFIC_VALIDATION_RULES,
AUDIT_COVERAGE_RULES, and SELF_CORRECTION_RULES.
Audit the actual previous output against the original prompt.
Do not substitute a generic audit framework.
"""
}

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("Research Assistant")
st.sidebar.caption("Version 2.3 — Final Demonstration Prototype")

selected_module = st.sidebar.selectbox(
    "Select Research Module",
    list(MODULES.keys())
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Demonstration Research Topic")
st.sidebar.caption(
    "Sex-Based Differences in Kidney-Related Healthcare "
    "Utilization, Monitoring, and Clinical Outcomes Among "
    "Patients with Tuberous Sclerosis Complex (TSC)"
)

# ============================================================
# MAIN INTERFACE
# ============================================================

st.title("AI-Powered Quantitative Research Assistant")

st.markdown("""
**A structured AI-assisted tool for quantitative health
research planning, statistical analysis, and scientific review.**

Demonstration topic: Sex-based differences in kidney-related
healthcare among patients with Tuberous Sclerosis Complex.
""")

st.info(
    "Research prototype. AI-generated recommendations require "
    "independent scientific, clinical, and statistical review."
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

# ============================================================
# MODULE INPUTS
# ============================================================

context = ""
audit_instructions = ""
previous_output = ""
original_prompt = ""

if selected_module == "Research Integrity & Scientific Validation":

    st.markdown("---")
    st.subheader("Scientific Audit Workspace")

    st.caption(
        "For a meaningful audit, provide the original prompt "
        "and the complete AI-generated response separately."
    )

    audit_instructions = st.text_area(
        "1. Audit Instructions",
        value=(
            "Conduct a complete independent scientific audit. "
            "Compare the previous AI response with the original "
            "prompt. Identify all material errors, unsupported "
            "assumptions, omissions, and overstatements. Quote "
            "actual statements, classify each issue, and provide "
            "corrected recommendations. Do not invent evidence."
        ),
        height=140
    )

    original_prompt = st.text_area(
        "2. Original User Prompt — Recommended",
        placeholder=(
            "Paste the exact prompt that generated the previous "
            "AI response. This helps identify invented assumptions."
        ),
        height=180
    )

    previous_output = st.text_area(
        "3. Previous AI Output — REQUIRED",
        placeholder=(
            "Paste the complete previous AI-generated response here..."
        ),
        height=350
    )

    context = st.text_area(
        "4. Optional: Original Protocol / SAP / Additional Context",
        placeholder=(
            "Paste protocol, statistical analysis plan, data dictionary, "
            "or relevant study information."
        ),
        height=160
    )

else:

    st.markdown("### Context / Documents / Previous AI Output")

    context = st.text_area(
        "Paste research notes, SAP, protocol, or previous AI output here",
        placeholder="Provide relevant research information.",
        height=220
    )

# ============================================================
# OPTIONAL FILE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Optional: Upload research material",
    type=["txt", "md"]
)

if uploaded_file is not None:
    uploaded_text = uploaded_file.read().decode(
        "utf-8", errors="replace"
    )

    if selected_module == "Research Integrity & Scientific Validation":
        context += "\n\nUPLOADED RESEARCH DOCUMENT:\n" + uploaded_text
    else:
        context += "\n\nUPLOADED DOCUMENT:\n" + uploaded_text

# ============================================================
# RESPONSE FUNCTION
# ============================================================

def generate_response(
    topic,
    objective,
    context,
    module,
    audit_instructions="",
    previous_output="",
    original_prompt=""
):

    if not topic.strip():
        return "Please enter a research topic."

    task = MODULES[module]

    if module == "Statistical Analysis Planner":
        task += "\n\n" + STATISTICAL_PLANNER_RULES

    if module == "Research Integrity & Scientific Validation":

        task += "\n\n" + SCIENTIFIC_VALIDATION_RULES
        task += "\n\n" + AUDIT_COVERAGE_RULES
        task += "\n\n" + SELF_CORRECTION_RULES

        user_prompt = f"""
RESEARCH TOPIC:
{topic}

RESEARCH AIM:
{objective if objective.strip() else "Not specified."}

ORIGINAL USER PROMPT:
<<< BEGIN ORIGINAL PROMPT >>>
{original_prompt if original_prompt.strip() else "Not supplied."}
<<< END ORIGINAL PROMPT >>>

AUDIT INSTRUCTIONS:
{audit_instructions}

ACTUAL PREVIOUS AI OUTPUT:
<<< BEGIN PREVIOUS AI OUTPUT >>>
{previous_output}
<<< END PREVIOUS AI OUTPUT >>>

ORIGINAL PROTOCOL / SAP / ADDITIONAL CONTEXT:
{context if context.strip() else "Not supplied."}

SELECTED MODULE:
{module}

TASK:
{task}

MANDATORY:
- Audit the actual supplied output, not an imagined response.
- Compare against the original prompt when supplied.
- Review all substantive sections and material recommendations.
- Do not stop after three issues if more are present.
- Do not invent statements absent from the output.
- Identify invented assumptions and unsupported specificity.
- Distinguish a changed estimand from demonstrated selection bias.
- Do not equate encounter incidence with disease incidence.
- Do not recommend models without adequate justification.
- State explicitly if any part could not be assessed.
- Never claim complete coverage unless the output was reviewed.
- Do not claim source verification without an actual source.
"""

    else:

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
        max_tokens=3000
    )

    return response.choices[0].message.content

# ============================================================
# RUN BUTTON
# ============================================================

if st.button("Run Research Assistant", type="primary"):

    if not topic.strip():

        st.error("Please enter a research topic.")

    elif (
        selected_module == "Research Integrity & Scientific Validation"
        and not previous_output.strip()
    ):

        st.error(
            "Please paste the actual previous AI output "
            "in the required field."
        )

    elif (
        selected_module == "Research Integrity & Scientific Validation"
        and "[PASTE" in previous_output.upper()
    ):

        st.error(
            "A placeholder was detected. Replace it with "
            "the complete actual AI-generated output."
        )

    else:

        with st.spinner("Analyzing your research request..."):

            try:

                result = generate_response(
                    topic=topic,
                    objective=objective,
                    context=context,
                    module=selected_module,
                    audit_instructions=audit_instructions,
                    previous_output=previous_output,
                    original_prompt=original_prompt
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
    "AI-assisted research prototype | Version 2.3 | "
    "Not a substitute for expert scientific review."
)
