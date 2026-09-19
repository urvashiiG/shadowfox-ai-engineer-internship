# StudyMate AI

**Your AI-powered study companion**

A focused, beginner-level AI application built for the **ShadowFox AI Engineer Internship**.
StudyMate AI gives students four practical, LLM-powered study utilities in a single, clean
Streamlit interface — not a generic chatbot, but a purpose-built study tool.

---

## 1. Project Overview

StudyMate AI is a Python + Streamlit web application that integrates with an LLM (Anthropic's
Claude API) to help students study more effectively. Instead of a free-form chat window, it
offers four dedicated, structured workflows, each with its own carefully engineered prompt,
input validation, and formatted output.

## 2. Problem Statement

Students constantly need to:
- Understand new concepts quickly and in plain language.
- Turn long, messy lecture notes into something revisable before an exam.
- Practice with quiz questions instead of only re-reading material.
- Improve the answers they've written before submitting an assignment or exam.

Doing all of this manually is slow, and generic chatbots require students to know how to
prompt well. StudyMate AI removes that friction by providing purpose-built, structured
workflows for each of these needs.

## 3. Why This Is Useful for Students

- **Saves time**: turns a page of notes into a ready-to-review summary in seconds.
- **Improves understanding**: explanations are broken into definition, intuition, steps,
  example, and common mistakes — not just a wall of text.
- **Supports active recall**: auto-generated quizzes let students test themselves instead of
  passively re-reading.
- **Builds better writing habits**: the "Improve Answer" tool shows *what* was improved and
  *why*, helping students learn to write clearer answers themselves over time.

## 4. Features

| Feature | Description |
|---|---|
| 🧠 **Explain Concept** | Enter any topic and receive a structured, student-friendly explanation (definition, intuition, steps, example, key points, common mistake). |
| 📝 **Summarize Notes** | Paste raw notes and get an organized summary: overview, key points, definitions, formulas, exam-relevant points, and a final revision recap. |
| ❓ **Generate Quiz** | Provide a topic or material and choose 5, 10, or 15 questions to get a formatted multiple-choice quiz with answers and explanations. |
| ✍️ **Improve Answer** | Paste an answer you wrote and get an improved version, a list of what was changed, clearly-labeled suggested additions, and tips for a stronger answer. |

## 5. Technology Stack

- **Python 3.11+**
- **Streamlit** — UI framework
- **Anthropic Python SDK** (`anthropic`) — official SDK for the Claude API
- **python-dotenv** — loads the API key from a local `.env` file
- **pytest** — unit testing for validation logic

This project intentionally avoids LangChain, vector databases, RAG, FastAPI, Docker, and
agentic frameworks — those belong to later (Intermediate/Advanced) stages of the internship
track. This is a beginner-level, single-service LLM integration project.

## 6. Project Structure

```
studymate-ai/
│
├── app.py                    # Streamlit UI, navigation, and page logic
├── llm_service.py             # LLM client init, API calls, error handling
├── prompts.py                 # All structured prompt builders (one per feature)
├── validators.py               # Reusable input validation functions
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
└── tests/
    └── test_validators.py     # Unit tests for validators.py
```

**Separation of concerns:**
- `app.py` — presentation only (Streamlit widgets, layout, calling the other modules).
- `llm_service.py` — API integration only (no UI code).
- `prompts.py` — prompt text only (no UI or API code).
- `validators.py` — pure validation functions (no UI or API code), independently testable.

## 7. Application Workflow

1. The user selects a feature from the sidebar (Explain Concept, Summarize Notes,
   Generate Quiz, or Improve Answer).
2. The user enters text (and, for quizzes, a question count) into the page.
3. On clicking the action button, `app.py` calls the relevant function in `validators.py`.
4. If validation fails, a friendly error message is shown and **no API call is made**.
5. If validation passes, `app.py` calls the matching prompt builder in `prompts.py` to build
   a structured prompt from the user's input.
6. `app.py` passes that prompt to `llm_service.generate_response()`, which calls the Claude
   API and returns a clean `LLMResult` (success + text, or failure + friendly error).
7. `app.py` renders either the formatted AI output or the friendly error message.

## 8. LLM Integration Explanation

`llm_service.py` is the single integration point with the LLM provider:

- It loads `LLM_API_KEY` from the environment (via `.env`, using `python-dotenv`).
- It lazily builds an `anthropic.Anthropic` client only when a request is made.
- It calls `client.messages.create(...)` using the Claude Sonnet model, a fixed
  `max_tokens` budget, and a single user message containing the fully-built prompt.
- It extracts plain text from the response's content blocks and returns it via a small
  `LLMResult` dataclass so `app.py` never has to deal with SDK-specific response objects.
- No API key is ever hardcoded; it must be supplied via the environment.

This is a direct, single-call LLM integration (not merely a hardcoded prompt demo) — every
feature in the UI dynamically builds a prompt from live user input and sends it to the API
on demand.

## 9. Prompt Engineering Approach

Every prompt in `prompts.py` follows the same deliberate structure:

- **ROLE** — the persona the model should adopt (e.g. "expert academic tutor").
- **TASK** — what it must accomplish.
- **INPUT** — the user's actual content, inserted directly into the prompt.
- **OUTPUT FORMAT** — an exact section-by-section structure the model must follow
  (e.g. Markdown headings for "Simple Definition", "Intuition", "Example", etc.).
- **CONSTRAINTS** — explicit rules, such as "do not invent information not present in the
  notes" or "do not change the intended meaning of the answer."

This structure keeps outputs consistent, readable, and on-topic, and is a core demonstration
of basic prompt engineering for this internship level.

## 10. Validation Strategy

`validators.py` provides small, composable, pure functions:

- `is_empty_or_whitespace()` — detects missing/blank input.
- `validate_text_input()` — generic length + emptiness validation, reused by feature-specific
  wrappers (`validate_concept_input`, `validate_notes_input`, `validate_quiz_topic_input`,
  `validate_answer_input`) each with sensible per-feature character limits.
- `validate_quiz_count()` — restricts the quiz size to 5, 10, or 15 questions.

All validation happens **before** any API call is made, so invalid input never reaches (or
wastes) the LLM API.

## 11. Error Handling Strategy

The app is designed to never crash and never leak internal details:

- **Missing API key** → `llm_service.is_configured()` lets the UI show a persistent warning
  banner, and any generation attempt returns a clear configuration error instead of crashing.
- **Authentication errors** → friendly message about checking the API key.
- **Rate-limit errors** → friendly message asking the user to wait and retry.
- **Network / connection errors** → friendly message about checking the internet connection.
- **Timeouts** → friendly message asking the user to retry.
- **General API errors** → friendly generic "service unavailable" message.
- **Empty/malformed responses** → detected and reported as a friendly error instead of
  silently showing a blank result.
- In every case, raw exceptions, stack traces, and credentials are **never** shown to the
  user — only a clean, human-readable message.

## 12. Setup Instructions

### 12.1 Create and activate a virtual environment

```bash
python -m venv .venv
```

**macOS / Linux:**
```bash
source .venv/bin/activate
```

**Windows (PowerShell):**
```powershell
.venv\Scripts\Activate.ps1
```

**Windows (cmd.exe):**
```cmd
.venv\Scripts\activate.bat
```

### 12.2 Install dependencies

```bash
pip install -r requirements.txt
```

## 13. Environment Variable Setup

1. Copy `.env.example` to a new file named `.env`:
   ```bash
   cp .env.example .env
   ```
   (On Windows: `copy .env.example .env`)
2. Open `.env` and replace `your_api_key_here` with your real Anthropic API key:
   ```
   LLM_API_KEY=sk-ant-your-real-key-here
   ```
3. **Never commit `.env`** — it is already excluded via `.gitignore`.

## 14. How to Run the Application

```bash
streamlit run app.py
```

Streamlit will start a local server and open the app in your browser
(typically at `http://localhost:8501`).

If `LLM_API_KEY` is not set, the app will still load and clearly indicate in the sidebar and
on every page that the API key is missing — it will not pretend to work or fabricate output.

## 15. How to Run Tests

```bash
pytest
```

This runs the unit tests in `tests/test_validators.py`, covering empty input, whitespace-only
input, valid input, excessively long input, valid/invalid quiz counts, and boundary values.

## 16. Example Use Cases

- A student pastes their biology lecture notes into **Summarize Notes** the night before an
  exam to get a quick revision sheet.
- A student struggling with a CS concept types "Explain recursion" into **Explain Concept**
  for a beginner-friendly walkthrough with an example.
- A student preparing for a test uses **Generate Quiz** on their History chapter to create
  10 practice multiple-choice questions.
- A student pastes a short-answer response they wrote into **Improve Answer** to see how to
  make it clearer and more complete before submitting it.

## 17. Limitations

- Requires an active internet connection and a valid API key; there is no offline mode.
- Does not accept file uploads (PDF/DOCX/TXT) — input is plain pasted text only, per the
  beginner-level scope of this project.
- Summaries and quizzes are only as good as the material the user provides; the app does not
  fetch or verify external information (no retrieval/RAG in this version).
- No user accounts, history, or persistence — each session is stateless.
- Quiz size is limited to 5, 10, or 15 questions to keep output focused and readable.

## 18. Future Improvements

- Support file uploads (PDF, DOCX, TXT) for notes and study material.
- Add downloadable/exportable summaries and quizzes (e.g. as PDF or Markdown files).
- Add a difficulty selector for generated quizzes.
- Add session history so students can revisit previous explanations/summaries.
- Introduce retrieval-augmented generation (RAG) for larger documents — planned for the
  Intermediate/Advanced tracks, intentionally out of scope here.

## 19. Internship Relevance

This project satisfies the ShadowFox **AI Engineer Internship – Beginner Level** requirements
by demonstrating:

- Real LLM API integration (not a static/hardcoded demo).
- Deliberate, structured prompt engineering across four distinct use cases.
- Clean separation between UI, prompt logic, API logic, and validation logic.
- Defensive input validation applied before every API call.
- Comprehensive, non-crashing error handling for API and network failures.
- A polished, usable interface solving a real, relatable student problem.
- Unit-tested core logic (`tests/test_validators.py`) and a professional project structure
  suitable for a GitHub submission.
