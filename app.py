import os
import streamlit as st
from groq import Groq

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
    A step-by-step research development tool for quantitative
    health and pharmacy research.

    **Research planning → Methodology → Variables → Analysis plan
    → Interpretation → Research proposal**
    """
)

st.caption(
    "Educational research-support prototype. "
    "All AI-generated methodological suggestions require researcher review."
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

if "research_outputs" not in st.session_state:
    st.session_state.research_outputs = {}

if "research_topic" not in st.session_state:
    st.session_state.research_topic = DEFAULT_TOPIC

if "research_objective" not in st.session_state:
    st.session_state.research_objective = DEFAULT_OBJECTIVE

# ============================================================
# 4. GROQ CONNECTION
# ============================================================

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


MODEL_NAME = "openai/gpt-oss-20b"

# ============================================================
# 5. AI INSTRUCTIONS
# ============================================================

SYSTEM_PROMPT = """
You are an AI-assisted quantitative health research assistant.

Your role is to support researchers in developing quantitative
health and pharmacy research, with particular attention to
pharmacy practice, healthcare utilization, health services,
clinical outcomes, and population health.

IMPORTANT INSTRUCTIONS:

1. Explain research and statistical concepts in clear,
   accessible academic language.

2. Help the researcher understand the reasoning behind
   methodological decisions instead of merely listing methods.

3. Do not invent study findings, patient data, references,
   journal articles, DOIs, or citations.

4. Clearly distinguish established information, assumptions,
   methodological suggestions, and points requiring verification.

5. Consider study objectives, research questions, population,
   study design, outcome types, and available data before
   recommending statistical methods.

6. Consider confounding, selection bias, information bias,
   missing data, sample size, and study limitations.

7. Do not automatically recommend advanced methods.
   Explain when a method is necessary and when it may not be suitable.

8. For count outcomes, explain Poisson and negative binomial
   regression where appropriate.

9. For binary outcomes, explain suitable approaches such as
   logistic regression where appropriate.

10. For time-to-event outcomes, explain survival analysis
    where appropriate.

11. Never interpret synthetic or hypothetical examples as
    real patient evidence.

12. Do not present a possible literature gap as an established
    fact without supporting literature.

13. Keep the researcher's original topic and objectives in focus.

14. Explain technical terminology briefly when introducing it.

15. Use structured headings and practical recommendations.
"""

# ============================================================
# 6. AI RESPONSE FUNCTION
# ============================================================

def generate_ai_response(stage, topic, objective, instructions,
                         previous_work=""):

    client = get_groq_client()

    if client is None:
        return (
            "Groq API connection is unavailable.\n\n"
            "Please check that GROQ_API_KEY is correctly "
            "configured in Streamlit Secrets."
        )

    prompt = f"""
RESEARCH TOPIC:
{topic}

OVERALL RESEARCH OBJECTIVE:
{objective}

CURRENT RESEARCH DEVELOPMENT STAGE:
{stage}

TASK:
{instructions}

WORK COMPLETED IN PREVIOUS STAGES:
{previous_work[:5000]}

Please complete the current stage.

Explain the reasoning behind your recommendations.
Define technical terms briefly.
Identify assumptions and limitations.
Do not invent evidence, citations, or study results.
Keep recommendations relevant to the stated research topic.
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
                    "content": prompt
                }
            ],
            temperature=0.2,
            max_completion_tokens=1800
        )

        return response.choices[0].message.content

    except Exception as error:

        return (
            "AI request could not be completed.\n\n"
            f"Error: {str(error)}"
        )

# ============================================================
# 7. RESEARCH CONTEXT
# ============================================================

st.header("Research Foundation")

st.write(
    "Start by defining the research topic and its main objective. "
    "These will guide all subsequent research-development stages."
)

st.session_state.research_topic = st.text_area(
    "Research topic",
    value=st.session_state.research_topic,
    height=90,
    key="research_topic_input"
)

st.session_state.research_objective = st.text_area(
    "Overall research objective",
    value=st.session_state.research_objective,
    height=90,
    key="research_objective_input"
)

topic = st.session_state.research_topic
objective = st.session_state.research_objective

st.divider()

# ============================================================
# 8. PREVIOUS WORK CONTEXT
# ============================================================

def get_previous_work():

    collected = []

    for stage_name, output in st.session_state.research_outputs.items():

        collected.append(
            f"\n{stage_name}\n{output}"
        )

    return "\n".join(collected)

# ============================================================
# 9. RESEARCH WORKFLOW FUNCTION
# ============================================================

def research_stage(stage_number, title, description, instructions):

    stage_key = f"stage_{stage_number}"

    st.subheader(f"Step {stage_number}. {title}")

    st.write(description)

    user_instructions = st.text_area(
        "Additional information or instructions (optional)",
        key=f"{stage_key}_input",
        height=90,
        placeholder=(
            "Add any specific research details, "
            "questions, or concerns for this stage..."
        )
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

            complete_instructions = (
                instructions
                + "\n\nAdditional researcher input:\n"
                + user_instructions
            )

            previous_work = get_previous_work()

            with st.spinner(
                f"Developing Step {stage_number}..."
            ):

                result = generate_ai_response(
                    stage=title,
                    topic=topic,
                    objective=objective,
                    instructions=complete_instructions,
                    previous_work=previous_work
                )

            st.session_state.research_outputs[title] = result

    if title in st.session_state.research_outputs:

        st.markdown("#### AI Research Output")

        st.markdown(
            st.session_state.research_outputs[title]
        )

        st.download_button(
            label=f"Download {title}",
            data=st.session_state.research_outputs[title],
            file_name=f"research_step_{stage_number}.txt",
            mime="text/plain",
            key=f"{stage_key}_download"
        )

# ============================================================
# 10. STEP 1 — RESEARCH QUESTIONS
# ============================================================

st.header("Research Development Workflow")

st.markdown(
    "Complete the following stages in sequence. "
    "Each stage builds on the research work developed earlier."
)

research_stage(
    stage_number=1,
    title="Research Questions and Objectives",
    description=(
        "Develop focused, clear, and answerable research "
        "questions from the overall research objective."
    ),
    instructions="""
Develop this research stage.

1. Identify the central research problem.
2. Develop one primary research question.
3. Suggest relevant secondary research questions.
4. Refine the primary and secondary objectives.
5. Explain how each question can be answered quantitatively.
6. Check whether the questions are specific and measurable.

Explain the difference between a research question,
a research aim, and a research objective.
"""
)

st.divider()

# ============================================================
# 11. STEP 2 — STUDY DESIGN AND METHODOLOGY
# ============================================================

research_stage(
    stage_number=2,
    title="Study Design and Methodology",
    description=(
        "Determine how the research questions can be investigated, "
        "including the population, setting, and study design."
    ),
    instructions="""
Develop the study methodology.

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
# 12. STEP 3 — VARIABLES AND DATA COLLECTION
# ============================================================

research_stage(
    stage_number=3,
    title="Variables and Data Collection",
    description=(
        "Translate the research questions into measurable variables "
        "and define how information would be collected."
    ),
    instructions="""
Develop a variable framework for this study.

1. Identify the primary outcome variable.
2. Identify relevant secondary outcome variables.
3. Identify the main exposure or grouping variable.
4. Identify potential confounding variables.
5. Suggest operational definitions for each variable.
6. Identify suitable measurement scales and data types.
7. Explain how variables relate to the research questions.
8. Present a structured variable table.

For each suggested variable, explain what it measures
and why it is relevant.

Do not assume that all suggested variables will be available
in the final dataset.
"""
)

st.divider()

# ============================================================
# 13. STEP 4 — STATISTICAL ANALYSIS PLAN
# ============================================================

research_stage(
    stage_number=4,
    title="Statistical Analysis Plan",
    description=(
        "Select statistical methods according to the research "
        "questions, outcome variables, study design, and data."
    ),
    instructions="""
Prepare a structured statistical analysis plan.

1. Explain the descriptive analysis required.
2. Match each research question with an appropriate
   statistical comparison or model.
3. Explain methods for continuous outcomes.
4. Explain methods for binary outcomes.
5. Explain methods for count outcomes.
6. Discuss regression approaches and adjustment for confounders.
7. Explain assumptions and diagnostic checks.
8. Discuss missing data.
9. Explain how results could be presented using tables
   and figures.

Where relevant, discuss:
- Welch's t-test
- Chi-square test
- Fisher's exact test
- Logistic regression
- Poisson regression
- Negative binomial regression

Do not recommend every method automatically.
Explain which methods are relevant to this specific study
and what information is needed before choosing them.
"""
)

st.divider()

# ============================================================
# 14. STEP 5 — INTERPRETATION AND RESEARCH QUALITY
# ============================================================

research_stage(
    stage_number=5,
    title="Interpretation, Bias and Research Quality",
    description=(
        "Consider how results should be interpreted and what "
        "methodological limitations may affect the findings."
    ),
    instructions="""
Review the methodological quality of the proposed research.

1. Explain how the findings should be interpreted.
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

Do not invent findings or claim that hypothetical results
have already been observed.
"""
)

st.divider()

# ============================================================
# 15. STEP 6 — RESEARCH PROPOSAL
# ============================================================

research_stage(
    stage_number=6,
    title="Research Proposal Development",
    description=(
        "Bring the previous research-development stages together "
        "into a coherent quantitative research proposal."
    ),
    instructions="""
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
# 16. FINAL CONSOLIDATED PROPOSAL
# ============================================================

st.header("Final Research Proposal")

st.write(
    "After developing the research stages, generate one "
    "consolidated proposal using the work completed above."
)

if st.button(
    "Generate Consolidated Research Proposal",
    type="primary",
    key="final_proposal_button"
):

    previous_work = get_previous_work()

    if not previous_work.strip():

        st.warning(
            "Please complete at least the first research stage "
            "before generating a proposal."
        )

    else:

        final_instructions = """
Prepare a coherent consolidated quantitative research proposal
using the research work supplied from the completed stages.

Organize the proposal into:

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

Identify inconsistencies between the earlier stages.
Do not silently invent missing information.
Clearly mark sections that need further development,
literature verification, or supervisor input.
"""

        with st.spinner(
            "Preparing the consolidated research proposal..."
        ):

            final_result = generate_ai_response(
                stage="Consolidated Research Proposal",
                topic=topic,
                objective=objective,
                instructions=final_instructions,
                previous_work=previous_work
            )

        st.session_state.research_outputs[
            "Final Research Proposal"
        ] = final_result

if "Final Research Proposal" in st.session_state.research_outputs:

    st.markdown("### Consolidated Proposal")

    st.markdown(
        st.session_state.research_outputs["Final Research Proposal"]
    )

    st.download_button(
        label="Download Complete Research Proposal",
        data=st.session_state.research_outputs[
            "Final Research Proposal"
        ],
        file_name="quantitative_research_proposal.txt",
        mime="text/plain",
        key="final_proposal_download"
    )

st.divider()

# ============================================================
# 17. RESEARCH WORK SUMMARY
# ============================================================

st.header("Research Progress")

completed_stages = len(st.session_state.research_outputs)

total_stages = 6

st.progress(
    min(completed_stages, total_stages) / total_stages
)

st.write(
    f"Research stages completed: "
    f"{min(completed_stages, total_stages)} / {total_stages}"
)

for stage_name in [
    "Research Questions and Objectives",
    "Study Design and Methodology",
    "Variables and Data Collection",
    "Statistical Analysis Plan",
    "Interpretation, Bias and Research Quality",
    "Research Proposal Development"
]:

    if stage_name in st.session_state.research_outputs:
        st.success(f"Completed: {stage_name}")
    else:
        st.info(f"Pending: {stage_name}")

st.divider()

# ============================================================
# 18. RESEARCH GUIDELINES
# ============================================================

st.subheader("Important Research Guidelines")

st.markdown(
    """
    - AI suggestions are a starting point, not final research decisions.
    - Study design must match the research question.
    - Variables must be clearly defined before analysis.
    - Statistical methods must match the outcome type and study design.
    - Confounding, missing data, and bias must be considered.
    - Literature claims and references must be independently verified.
    - Final methods and interpretations require supervisor review.
    """
)

# ============================================================
# 19. FOOTER
# ============================================================

st.divider()

st.caption(
    "AI-Assisted Quantitative Research Assistant | "
    "Research planning and proposal development prototype"
)
