"""
StudyMate AI
Your AI-powered study companion.

Streamlit application entry point.
"""

import streamlit as st

import llm_service
import prompts
import validators


# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="StudyMate AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================================
# PREMIUM DARK THEME
# ============================================================================

st.markdown(
    """
<style>

/* =========================
   GLOBAL
   ========================= */

.stApp {
    background:
        radial-gradient(
            circle at 75% 5%,
            rgba(124, 92, 255, 0.10),
            transparent 28%
        ),
        #090B10;
    color: #ECECF4;
}

.main .block-container {
    max-width: 1250px;
    padding: 2.5rem 3.5rem 4rem 3.5rem;
}


/* =========================
   SIDEBAR
   ========================= */

section[data-testid="stSidebar"] {
    background: #0D1017;
    border-right: 1px solid #1D2230;
}

section[data-testid="stSidebar"] > div {
    padding: 2rem 1.15rem;
}

.sidebar-brand {
    padding: 4px 6px 22px 6px;
}

.sidebar-logo {
    font-size: 1.45rem;
    font-weight: 800;
    color: #F2F1F8;
    letter-spacing: -0.4px;
}

.sidebar-tagline {
    color: #85899A;
    font-size: 0.84rem;
    margin-top: 5px;
}

.sidebar-label {
    color: #686D80;
    font-size: 0.68rem;
    font-weight: 750;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    margin: 12px 6px 10px 6px;
}


/* Radio navigation */

section[data-testid="stSidebar"] div[role="radiogroup"] {
    gap: 7px;
}

section[data-testid="stSidebar"] div[role="radiogroup"] label {
    background: transparent;
    border: 1px solid transparent;
    border-radius: 11px;
    padding: 10px 12px;
    color: #A4A7B5;
    transition: all 0.2s ease;
}

section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
    background: #151925;
    color: #E9E8F2;
    border-color: #202536;
}

section[data-testid="stSidebar"] div[role="radiogroup"]
label[data-checked="true"] {
    background: linear-gradient(
        90deg,
        rgba(124, 92, 255, 0.18),
        rgba(124, 92, 255, 0.07)
    );
    border-color: #302A58;
    color: #D9D4FF;
}


/* =========================
   HERO
   ========================= */

.hero {
    position: relative;
    overflow: hidden;

    background:
        radial-gradient(
            circle at 90% 0%,
            rgba(139, 124, 255, 0.18),
            transparent 35%
        ),
        linear-gradient(
            145deg,
            #11141D,
            #0E1118
        );

    border: 1px solid #222737;
    border-radius: 22px;

    padding: 32px 34px;
    margin-bottom: 26px;

    box-shadow:
        0 20px 60px rgba(0, 0, 0, 0.22);
}

.hero-icon {
    width: 48px;
    height: 48px;

    display: flex;
    align-items: center;
    justify-content: center;

    background: rgba(139, 124, 255, 0.13);
    border: 1px solid rgba(139, 124, 255, 0.22);

    border-radius: 14px;

    font-size: 24px;
    margin-bottom: 18px;
}

.hero-title {
    color: #F2F1F8;
    font-size: 2.15rem;
    line-height: 1.15;
    font-weight: 800;
    letter-spacing: -0.8px;
}

.hero-subtitle {
    color: #858A9D;
    font-size: 0.98rem;
    margin-top: 9px;
    line-height: 1.6;
}


/* =========================
   LABELS
   ========================= */

label,
.stMarkdown p {
    color: #B7BAC8;
}


/* =========================
   INPUTS
   ========================= */

textarea,
input {
    background: #11141C !important;
    color: #ECECF4 !important;

    border: 1px solid #292E3E !important;
    border-radius: 13px !important;

    box-shadow: none !important;
}

textarea::placeholder,
input::placeholder {
    color: #565B6D !important;
}

textarea:focus,
input:focus {
    border-color: #7568E8 !important;
    box-shadow:
        0 0 0 1px rgba(117, 104, 232, 0.35) !important;
}


/* =========================
   PRIMARY BUTTON
   ========================= */

.stButton > button {
    background: linear-gradient(
        135deg,
        #7568E8,
        #6255D2
    );

    color: #FFFFFF;

    border: 1px solid #8175EE;
    border-radius: 12px;

    font-weight: 700;
    letter-spacing: 0.1px;

    min-height: 45px;

    box-shadow:
        0 8px 25px rgba(98, 85, 210, 0.20);

    transition: all 0.2s ease;
}

.stButton > button:hover {
    background: linear-gradient(
        135deg,
        #8175F0,
        #6B5DE0
    );

    border-color: #948AFF;

    transform: translateY(-1px);

    box-shadow:
        0 12px 30px rgba(98, 85, 210, 0.30);
}

.stButton > button:active {
    transform: translateY(0);
}


/* =========================
   SELECTBOX
   ========================= */

div[data-baseweb="select"] > div {
    background: #11141C !important;
    color: #ECECF4 !important;

    border: 1px solid #292E3E !important;
    border-radius: 12px !important;
}


/* =========================
   RESULT HEADER
   ========================= */

.result-header {
    display: flex;
    align-items: center;
    gap: 11px;

    background: linear-gradient(
        90deg,
        rgba(124, 92, 255, 0.12),
        rgba(124, 92, 255, 0.04)
    );

    border: 1px solid #2B2850;
    border-radius: 13px;

    padding: 13px 16px;
    margin-top: 28px;
    margin-bottom: 14px;
}

.result-icon {
    font-size: 19px;
}

.result-title {
    color: #CFC9FF;
    font-weight: 700;
    font-size: 0.92rem;
}


/* =========================
   FEATURE INFO
   ========================= */

.info-card {
    background: #0F121A;
    border: 1px solid #202534;
    border-radius: 15px;

    padding: 18px 20px;
    margin-bottom: 18px;
}

.info-title {
    color: #E9E8F2;
    font-weight: 700;
    font-size: 0.94rem;
}

.info-text {
    color: #73788B;
    font-size: 0.83rem;
    line-height: 1.55;
    margin-top: 5px;
}


/* =========================
   STATUS
   ========================= */

div[data-testid="stAlert"] {
    border-radius: 12px;
}


/* =========================
   DIVIDERS
   ========================= */

hr {
    border-color: #202432 !important;
}


/* =========================
   CAPTIONS
   ========================= */

.stCaption,
[data-testid="stCaptionContainer"] {
    color: #686D80 !important;
}


/* =========================
   FOOTER
   ========================= */

.footer {
    text-align: center;
    color: #4F5465;
    font-size: 0.76rem;
    padding-top: 30px;
}


/* =========================
   MOBILE
   ========================= */

@media (max-width: 768px) {

    .main .block-container {
        padding: 1.3rem 1rem 3rem 1rem;
    }

    .hero {
        padding: 24px;
        border-radius: 18px;
    }

    .hero-title {
        font-size: 1.75rem;
    }

}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================================
# FEATURES
# ============================================================================

FEATURES = {
    "Explain Concept": {
        "icon": "🧠",
        "description": "Understand difficult topics in simple language.",
    },
    "Summarize Notes": {
        "icon": "📝",
        "description": "Turn long notes into concise study material.",
    },
    "Generate Quiz": {
        "icon": "❓",
        "description": "Create practice questions from your material.",
    },
    "Improve Answer": {
        "icon": "✨",
        "description": "Make your answers clearer and exam-ready.",
    },
}


# ============================================================================
# HELPERS
# ============================================================================

def show_hero(icon: str, title: str, description: str) -> None:
    """Display premium feature header."""

    st.markdown(
        f'<div class="hero">'
        f'<div class="hero-icon">{icon}</div>'
        f'<div class="hero-title">{title}</div>'
        f'<div class="hero-subtitle">{description}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def show_result(result) -> None:
    """Display AI output using native Streamlit Markdown."""

    st.markdown(
        '<div class="result-header">'
        '<div class="result-icon">✨</div>'
        '<div class="result-title">StudyMate AI Response</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # Native Markdown is intentionally used here so headings,
    # bullets, numbering and bold text render correctly.
    st.markdown(result.text)


def run_llm_and_display(prompt_text: str, spinner_message: str) -> None:
    """Call the LLM service and display the response."""

    with st.spinner(spinner_message):
        result = llm_service.generate_response(prompt_text)

    if result.success:
        show_result(result)
    else:
        st.error(result.error)


def api_key_warning_if_needed() -> None:
    """Show API configuration status."""

    if not llm_service.is_configured():
        st.warning(
            "Your AI API key is not configured. Add it to your `.env` file "
            "to enable StudyMate AI.",
            icon="🔑",
        )


# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:

    st.markdown(
        '<div class="sidebar-brand">'
        '<div class="sidebar-logo">🧠 StudyMate AI</div>'
        '<div class="sidebar-tagline">Your intelligent study companion</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="sidebar-label">Study Tools</div>',
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Study Tools",
        options=list(FEATURES.keys()),
        format_func=lambda name: (
            f"{FEATURES[name]['icon']}  {name}"
        ),
        label_visibility="collapsed",
    )

    st.divider()

    if llm_service.is_configured():
        st.success("AI connected", icon="✅")
    else:
        st.warning("API key required", icon="🔑")

    st.markdown(
        '<div class="info-card">'
        '<div class="info-title">Built for students</div>'
        '<div class="info-text">'
        'Understand concepts, summarize notes, practice with quizzes, '
        'and improve your written answers.'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )


# ============================================================================
# EXPLAIN CONCEPT
# ============================================================================

if page == "Explain Concept":

    show_hero(
        "🧠",
        "Explain Concept",
        "Understand any topic with a clear, student-friendly explanation.",
    )

    api_key_warning_if_needed()

    concept_input = st.text_area(
        "What do you want to understand?",
        placeholder=(
            "Try: Explain gradient descent with a simple example..."
        ),
        height=145,
    )

    st.caption(
        "Ask about a concept, definition, process, algorithm, or example."
    )

    if st.button(
        "✨  Explain this concept",
        type="primary",
        use_container_width=True,
    ):

        is_valid, error_message = (
            validators.validate_concept_input(concept_input)
        )

        if not is_valid:
            st.error(error_message)
        else:
            prompt_text = prompts.build_explain_concept_prompt(
                concept_input.strip()
            )

            run_llm_and_display(
                prompt_text,
                "Creating your explanation...",
            )


# ============================================================================
# SUMMARIZE NOTES
# ============================================================================

elif page == "Summarize Notes":

    show_hero(
        "📝",
        "Summarize Notes",
        "Turn long study material into concise, exam-ready notes.",
    )

    api_key_warning_if_needed()

    notes_input = st.text_area(
        "Paste your notes",
        placeholder=(
            "Paste lecture notes, textbook material, revision notes, "
            "or study content here..."
        ),
        height=310,
    )

    char_count = len(notes_input.strip()) if notes_input else 0

    st.caption(
        f"{char_count:,} / {validators.MAX_NOTES_LENGTH:,} characters"
    )

    if st.button(
        "✨  Summarize my notes",
        type="primary",
        use_container_width=True,
    ):

        is_valid, error_message = (
            validators.validate_notes_input(notes_input)
        )

        if not is_valid:
            st.error(error_message)
        else:
            prompt_text = prompts.build_summarize_notes_prompt(
                notes_input.strip()
            )

            run_llm_and_display(
                prompt_text,
                "Summarizing your notes...",
            )


# ============================================================================
# GENERATE QUIZ
# ============================================================================

elif page == "Generate Quiz":

    show_hero(
        "❓",
        "Generate Quiz",
        "Test your knowledge with AI-generated multiple-choice questions.",
    )

    api_key_warning_if_needed()

    quiz_input = st.text_area(
        "Topic or study material",
        placeholder=(
            "Try: DBMS normalization, machine learning, operating systems..."
        ),
        height=230,
    )

    num_questions = st.selectbox(
        "Number of questions",
        options=validators.ALLOWED_QUIZ_COUNTS,
        index=0,
    )

    if st.button(
        "✨  Generate quiz",
        type="primary",
        use_container_width=True,
    ):

        is_valid_text, text_error = (
            validators.validate_quiz_topic_input(quiz_input)
        )

        is_valid_count, count_error = (
            validators.validate_quiz_count(num_questions)
        )

        if not is_valid_text:
            st.error(text_error)
        elif not is_valid_count:
            st.error(count_error)
        else:
            prompt_text = prompts.build_generate_quiz_prompt(
                quiz_input.strip(),
                int(num_questions),
            )

            run_llm_and_display(
                prompt_text,
                f"Creating your {num_questions}-question quiz...",
            )


# ============================================================================
# IMPROVE ANSWER
# ============================================================================

elif page == "Improve Answer":

    show_hero(
        "✨",
        "Improve Answer",
        "Turn your draft into a clearer, stronger, exam-ready answer.",
    )

    api_key_warning_if_needed()

    answer_input = st.text_area(
        "Paste your answer",
        placeholder=(
            "Paste an answer you wrote for an assignment, CT, or exam..."
        ),
        height=270,
    )

    st.caption(
        "StudyMate improves clarity, structure, terminology, and "
        "exam-readiness while preserving your original meaning."
    )

    if st.button(
        "✨  Improve my answer",
        type="primary",
        use_container_width=True,
    ):

        is_valid, error_message = (
            validators.validate_answer_input(answer_input)
        )

        if not is_valid:
            st.error(error_message)
        else:
            prompt_text = prompts.build_improve_answer_prompt(
                answer_input.strip()
            )

            run_llm_and_display(
                prompt_text,
                "Polishing your answer...",
            )


# ============================================================================
# FOOTER
# ============================================================================

st.markdown(
    '<div class="footer">'
    '🧠 StudyMate AI &nbsp;·&nbsp; '
    'Built with Streamlit &nbsp;·&nbsp; '
    'ShadowFox AI Engineer Internship'
    '</div>',
    unsafe_allow_html=True,
)