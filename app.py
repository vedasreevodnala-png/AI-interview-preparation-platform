import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
import ollama
import re
import uuid


# ============================================================
# AI INTERVIEW PREPARATION PLATFORM
# Single File Application
# ============================================================


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Interview Preparation Platform",
    page_icon="🎯",
    layout="wide"
)


# ============================================================
# LOAD SENTENCE TRANSFORMER
# ============================================================

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )


# ============================================================
# LOAD CHROMADB
# ============================================================

@st.cache_resource
def load_chroma():

    client = chromadb.Client()

    collection = client.get_or_create_collection(
        name="interview_questions"
    )

    return collection


embedding_model = load_embedding_model()

collection = load_chroma()


# ============================================================
# SESSION STATE
# ============================================================

if "resume_text" not in st.session_state:
    st.session_state.resume_text = ""

if "questions" not in st.session_state:
    st.session_state.questions = []

if "answers" not in st.session_state:
    st.session_state.answers = []

if "scores" not in st.session_state:
    st.session_state.scores = []

if "interview_started" not in st.session_state:
    st.session_state.interview_started = False

if "current_question" not in st.session_state:
    st.session_state.current_question = 0

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ============================================================
# OLLAMA FUNCTION
# ============================================================

def ask_ollama(prompt):

    try:

        response = ollama.chat(
            model="llama3.2",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return response["message"]["content"]

    except Exception as e:

        return ""


# ============================================================
# PDF TEXT EXTRACTION
# ============================================================

def extract_pdf_text(uploaded_file):

    try:

        reader = PdfReader(
            uploaded_file
        )

        text = ""

        for page in reader.pages:

            page_text = page.extract_text()

            if page_text:

                text += page_text + "\n"

        return text.strip()

    except Exception as e:

        return ""


# ============================================================
# QUESTION BANK
# ============================================================

question_bank = {

    "Python": [

        "What are the differences between a list, tuple, set and dictionary in Python?",

        "What is a Python decorator?",

        "What is the difference between deep copy and shallow copy?",

        "Explain exception handling in Python.",

        "What are Python generators?",

        "What is object-oriented programming in Python?",

        "What is the difference between == and is in Python?",

        "Explain inheritance in Python.",

        "What is a lambda function in Python?",

        "What is the difference between append() and extend()?"
    ],


    "C Programming": [

        "What is a pointer in C?",

        "What is the difference between malloc() and calloc()?",

        "Explain structures and unions in C.",

        "What is recursion?",

        "What is a segmentation fault?",

        "Explain call by value and call by reference.",

        "What are storage classes in C?",

        "What is a dangling pointer?",

        "What is the difference between array and pointer?",

        "What is a function pointer?"
    ],


    "Java": [

        "What are the main features of Java?",

        "Explain JVM, JRE and JDK.",

        "What is inheritance in Java?",

        "What is polymorphism?",

        "What is the difference between method overloading and overriding?",

        "Explain exception handling in Java.",

        "What is an interface?",

        "What is encapsulation?",

        "What is abstraction?",

        "What is the difference between == and equals()?"
    ],


    "DBMS": [

        "What is DBMS?",

        "What is normalization?",

        "Explain primary key and foreign key.",

        "What are ACID properties?",

        "What is a transaction?",

        "What is the difference between DELETE, DROP and TRUNCATE?",

        "What are SQL joins?",

        "What is a candidate key?",

        "What is indexing?",

        "What is a database schema?"
    ],


    "Data Structures": [

        "What is a stack?",

        "What is a queue?",

        "Explain linked lists.",

        "What is a binary tree?",

        "What is binary search?",

        "What is the difference between BFS and DFS?",

        "What is hashing?",

        "What is a binary search tree?",

        "What is a heap?",

        "What is the difference between array and linked list?"
    ],


    "Operating Systems": [

        "What is an operating system?",

        "What is a process?",

        "What is a thread?",

        "What is deadlock?",

        "Explain virtual memory.",

        "What is process scheduling?",

        "What is the difference between process and thread?",

        "What is paging?",

        "What is a context switch?",

        "What are the states of a process?"
    ],


    "Computer Networks": [

        "What is the OSI model?",

        "Explain TCP and UDP.",

        "What is an IP address?",

        "What is DNS?",

        "What is HTTP and HTTPS?",

        "Explain the three-way handshake.",

        "What is a MAC address?",

        "What is a router?",

        "What is the difference between IPv4 and IPv6?",

        "What is a firewall?"
    ],


    "Electronics": [

        "What is a transistor?",

        "Explain the working of a BJT.",

        "What is an amplifier?",

        "What is a diode?",

        "Explain MOSFET.",

        "What is an operational amplifier?",

        "What is feedback in an amplifier?",

        "What is the difference between BJT and FET?",

        "What is a rectifier?",

        "What is transistor biasing?"
    ],


    "HR": [

        "Tell me about yourself.",

        "Why should we hire you?",

        "What are your strengths?",

        "What are your weaknesses?",

        "Where do you see yourself in five years?",

        "Why do you want to join our company?",

        "Tell me about a challenge you faced and how you solved it.",

        "Why should we select you?",

        "How do you handle pressure?",

        "Tell me about your biggest achievement."
    ]
}


# ============================================================
# ADD QUESTIONS TO CHROMADB
# ============================================================

def add_questions_to_database():

    documents = []

    ids = []

    metadatas = []

    for category, questions in question_bank.items():

        for question in questions:

            documents.append(
                question
            )

            ids.append(
                str(uuid.uuid4())
            )

            metadatas.append(
                {
                    "category": category
                }
            )

    if documents:

        try:

            embeddings = embedding_model.encode(
                documents
            ).tolist()

            collection.add(
                documents=documents,
                embeddings=embeddings,
                ids=ids,
                metadatas=metadatas
            )

        except Exception:
            pass


# ============================================================
# INITIALIZE DATABASE
# ============================================================

if collection.count() == 0:

    add_questions_to_database()


# ============================================================
# GENERATE AI QUESTIONS
# ============================================================

def generate_questions(
    category,
    level,
    number
):

    prompt = f"""
You are an expert job interviewer.

Generate exactly {number} interview questions.

Category: {category}

Difficulty: {level}

Return ONLY the questions.

Do not provide answers.
Do not provide explanations.
Do not provide headings.

Number every question.

Example:

1. What is Python?
2. What is a list in Python?
3. What is inheritance?

Generate exactly {number} questions.
"""

    result = ask_ollama(
        prompt
    )

    if not result:

        return []

    questions = []

    lines = result.split("\n")

    for line in lines:

        line = line.strip()

        if not line:
            continue

        line = re.sub(
            r"^\s*(?:Question\s*)?\d+\s*[\.\)\-:]\s*",
            "",
            line,
            flags=re.IGNORECASE
        )

        line = re.sub(
            r"^[-•*]\s*",
            "",
            line
        )

        line = line.strip()

        if line.lower() in [
            "questions",
            "interview questions",
            "technical questions"
        ]:
            continue

        if len(line) > 10:

            questions.append(
                line
            )

    return questions[:number]


# ============================================================
# RESUME QUESTIONS
# ============================================================

def generate_resume_questions(
    resume_text
):

    if not resume_text:

        return []

    prompt = f"""
You are an expert technical interviewer.

Analyze the following resume.

Generate exactly 10 interview questions
based on the resume.

Focus on:

- Skills
- Projects
- Internships
- Education
- Technologies
- Practical experience

Return ONLY questions.

Resume:

{resume_text[:12000]}
"""

    result = ask_ollama(
        prompt
    )

    if not result:

        return []

    questions = []

    for line in result.split("\n"):

        line = line.strip()

        if not line:
            continue

        line = re.sub(
            r"^\s*(?:Question\s*)?\d+\s*[\.\)\-:]\s*",
            "",
            line,
            flags=re.IGNORECASE
        )

        line = re.sub(
            r"^[-•*]\s*",
            "",
            line
        )

        if len(line) > 10:

            questions.append(
                line
            )

    return questions[:10]


# ============================================================
# EVALUATE ANSWER
# ============================================================

def evaluate_answer(
    question,
    answer
):

    prompt = f"""
You are an expert interview evaluator.

Evaluate the candidate's answer.

Question:

{question}

Candidate Answer:

{answer}

Give a score from 0 to 10.

Evaluate:

1. Technical correctness
2. Understanding
3. Relevance
4. Clarity
5. Completeness

Use exactly this format:

Score: X/10

Strengths:
- point
- point

Weaknesses:
- point
- point

Improved Answer:
Give a better interview-quality answer.
"""

    result = ask_ollama(
        prompt
    )

    if result:

        return result

    # --------------------------------------------------------
    # FALLBACK EVALUATION
    # --------------------------------------------------------

    words = len(
        answer.split()
    )

    if words >= 80:

        score = 8

    elif words >= 50:

        score = 7

    elif words >= 30:

        score = 6

    elif words >= 15:

        score = 5

    else:

        score = 4

    return f"""
Score: {score}/10

Strengths:
- You attempted the question.
- Your answer contains relevant information.

Weaknesses:
- The answer can be explained in more detail.
- Add examples and important technical points.

Improved Answer:
Try to explain the concept clearly, include the main
technical points, and provide a practical example where possible.
"""


# ============================================================
# SEMANTIC SEARCH
# ============================================================

def search_questions(
    query,
    category
):

    try:

        query_embedding = embedding_model.encode(
            [query]
        ).tolist()

        result = collection.query(
            query_embeddings=query_embedding,
            n_results=5,
            where={
                "category": category
            }
        )

        if result["documents"]:

            return result["documents"][0]

        return []

    except Exception:

        return []


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title(
        "🎯 Interview AI"
    )

    st.write(
        "AI-powered Interview Preparation Platform"
    )

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "🏠 Dashboard",
            "📄 Resume Analyzer",
            "🤖 AI Interview",
            "📚 Question Bank",
            "🔎 Similar Questions",
            "📊 Performance"
        ]
    )

    st.divider()

    st.caption(
        "Streamlit + Ollama + ChromaDB + "
        "Sentence Transformers + PyPDF"
    )


# ============================================================
# DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    st.title(
        "🎯 AI Interview Preparation Platform"
    )

    st.subheader(
        "Prepare smarter. Practice better. Get interview-ready."
    )

    st.write(
        """
This platform helps students and job seekers prepare
for technical and HR interviews using Artificial Intelligence.
"""
    )

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Question Categories",
            len(question_bank)
        )

    with col2:

        total_questions = sum(
            len(q)
            for q in question_bank.values()
        )

        st.metric(
            "Question Bank",
            total_questions
        )

    with col3:

        st.metric(
            "Questions Attempted",
            len(
                st.session_state.answers
            )
        )

    st.divider()

    st.subheader(
        "✨ Features"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            """
### 📄 Resume Analyzer

Upload your resume and generate
personalized interview questions.

### 🤖 AI Mock Interview

Practice an interview and receive
AI-based feedback.

### 📚 Question Bank

Practice technical and HR questions.
"""
        )

    with col2:

        st.markdown(
            """
### 🔎 Semantic Search

Find similar interview questions using
AI embeddings.

### 📊 Performance

Track your interview performance.

### 🧠 AI Feedback

Get strengths, weaknesses and
improved answers.
"""
        )


# ============================================================
# RESUME ANALYZER
# ============================================================

elif page == "📄 Resume Analyzer":

    st.title(
        "📄 Resume Analyzer"
    )

    st.write(
        "Upload your PDF resume to generate personalized interview questions."
    )

    uploaded_file = st.file_uploader(
        "Upload Resume",
        type=["pdf"]
    )

    if uploaded_file:

        resume_text = extract_pdf_text(
            uploaded_file
        )

        if resume_text:

            st.session_state.resume_text = resume_text

            st.success(
                "✅ Resume uploaded successfully!"
            )

            with st.expander(
                "View Extracted Resume"
            ):

                st.text_area(
                    "Resume Text",
                    resume_text,
                    height=300
                )

            if st.button(
                "🤖 Generate Resume Questions",
                use_container_width=True
            ):

                with st.spinner(
                    "Analyzing resume..."
                ):

                    questions = generate_resume_questions(
                        resume_text
                    )

                if questions:

                    st.session_state.questions = questions

                    st.success(
                        f"{len(questions)} questions generated."
                    )

                else:

                    st.warning(
                        "AI resume questions could not be generated. "
                        "Please make sure Ollama is running."
                    )

        else:

            st.error(
                "Could not extract text from this PDF."
            )

    if st.session_state.questions:

        st.subheader(
            "🎯 Personalized Interview Questions"
        )

        for i, question in enumerate(
            st.session_state.questions,
            start=1
        ):

            st.write(
                f"**{i}. {question}**"
            )


# ============================================================
# AI INTERVIEW
# ============================================================

elif page == "🤖 AI Interview":

    st.title(
        "🤖 AI Mock Interview"
    )

    # ========================================================
    # INTERVIEW NOT STARTED
    # ========================================================

    if not st.session_state.interview_started:

        st.subheader(
            "Configure Your Interview"
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            category = st.selectbox(
                "Interview Category",
                list(
                    question_bank.keys()
                )
            )

        with col2:

            level = st.selectbox(
                "Difficulty",
                [
                    "Beginner",
                    "Intermediate",
                    "Advanced"
                ]
            )

        with col3:

            number = st.slider(
                "Number of Questions",
                3,
                10,
                5
            )

        st.write("")

        if st.button(
            "🚀 Start Interview",
            use_container_width=True
        ):

            # ------------------------------------------------
            # GUARANTEED FALLBACK QUESTIONS
            # ------------------------------------------------

            available_questions = question_bank.get(
                category,
                []
            )

            questions = available_questions[
                :number
            ]

            # ------------------------------------------------
            # TRY OLLAMA
            # ------------------------------------------------

            try:

                with st.spinner(
                    "🤖 AI is preparing your questions..."
                ):

                    ai_questions = generate_questions(
                        category,
                        level,
                        number
                    )

                if ai_questions:

                    questions = ai_questions[
                        :number
                    ]

            except Exception:

                pass

            # ------------------------------------------------
            # STORE QUESTIONS
            # ------------------------------------------------

            if questions:

                st.session_state.questions = questions

                st.session_state.answers = []

                st.session_state.scores = []

                st.session_state.chat_history = []

                st.session_state.current_question = 0

                st.session_state.interview_started = True

                st.rerun()

            else:

                st.error(
                    "No questions available."
                )

    # ========================================================
    # INTERVIEW IN PROGRESS
    # ========================================================

    else:

        questions = st.session_state.questions

        current = st.session_state.current_question

        # ----------------------------------------------------
        # SAFETY CHECK
        # ----------------------------------------------------

        if not questions:

            st.error(
                "No interview questions found."
            )

            if st.button(
                "⬅ Start Again"
            ):

                st.session_state.interview_started = False

                st.rerun()

        # ----------------------------------------------------
        # DISPLAY CURRENT QUESTION
        # ----------------------------------------------------

        elif current < len(questions):

            total_questions = len(
                questions
            )

            progress_value = (
                current / total_questions
            )

            st.progress(
                progress_value
            )

            st.write(
                f"### Question {current + 1} of {total_questions}"
            )

            st.divider()

            # ------------------------------------------------
            # BIG QUESTION DISPLAY
            # ------------------------------------------------

            st.markdown(
                f"""
## ❓ {questions[current]}
"""
            )

            st.divider()

            # ------------------------------------------------
            # ANSWER
            # ------------------------------------------------

            answer = st.text_area(
                "✍️ Type your answer here:",
                height=200,
                key=f"answer_box_{current}",
                placeholder="Write your interview answer here..."
            )

            st.write("")

            # ------------------------------------------------
            # SUBMIT
            # ------------------------------------------------

            if st.button(
                "✅ Submit Answer",
                use_container_width=True
            ):

                if not answer.strip():

                    st.warning(
                        "⚠️ Please enter your answer before submitting."
                    )

                else:

                    with st.spinner(
                        "🤖 AI is evaluating your answer..."
                    ):

                        evaluation = evaluate_answer(
                            questions[current],
                            answer
                        )

                    # Save answer
                    st.session_state.answers.append(
                        answer
                    )

                    # Save evaluation
                    st.session_state.scores.append(
                        evaluation
                    )

                    # Save history
                    st.session_state.chat_history.append(
                        {
                            "question": questions[current],
                            "answer": answer,
                            "evaluation": evaluation
                        }
                    )

                    # Next question
                    st.session_state.current_question += 1

                    st.rerun()

        # ====================================================
        # INTERVIEW COMPLETED
        # ====================================================

        else:

            st.success(
                "🎉 Interview Completed Successfully!"
            )

            st.subheader(
                "📊 Interview Results"
            )

            st.write(
                f"Total Questions: **{len(questions)}**"
            )

            st.write(
                f"Questions Attempted: "
                f"**{len(st.session_state.answers)}**"
            )

            st.divider()

            # ------------------------------------------------
            # SHOW ALL ANSWERS AND FEEDBACK
            # ------------------------------------------------

            for i, item in enumerate(
                st.session_state.chat_history,
                start=1
            ):

                with st.expander(
                    f"Question {i}: {item['question']}"
                ):

                    st.markdown(
                        "**Your Answer:**"
                    )

                    st.write(
                        item["answer"]
                    )

                    st.markdown(
                        "**🤖 AI Evaluation:**"
                    )

                    st.write(
                        item["evaluation"]
                    )

            st.divider()

            # ------------------------------------------------
            # NEW INTERVIEW
            # ------------------------------------------------

            if st.button(
                "🔄 Start New Interview",
                use_container_width=True
            ):

                st.session_state.interview_started = False

                st.session_state.questions = []

                st.session_state.answers = []

                st.session_state.scores = []

                st.session_state.chat_history = []

                st.session_state.current_question = 0

                st.rerun()


# ============================================================
# QUESTION BANK
# ============================================================

elif page == "📚 Question Bank":

    st.title(
        "📚 Interview Question Bank"
    )

    category = st.selectbox(
        "Select Category",
        list(
            question_bank.keys()
        )
    )

    questions = question_bank[
        category
    ]

    st.subheader(
        f"{category} Questions"
    )

    for i, question in enumerate(
        questions,
        start=1
    ):

        st.markdown(
            f"""
**{i}. {question}**
"""
        )


# ============================================================
# SEMANTIC SEARCH
# ============================================================

elif page == "🔎 Similar Questions":

    st.title(
        "🔎 AI Semantic Question Search"
    )

    st.write(
        "Search for interview questions based on meaning."
    )

    category = st.selectbox(
        "Category",
        list(
            question_bank.keys()
        )
    )

    query = st.text_input(
        "Enter your topic or question"
    )

    if st.button(
        "🔍 Search",
        use_container_width=True
    ):

        if not query.strip():

            st.warning(
                "Please enter a search query."
            )

        else:

            with st.spinner(
                "Searching..."
            ):

                results = search_questions(
                    query,
                    category
                )

            st.subheader(
                "Similar Questions"
            )

            if results:

                for i, result in enumerate(
                    results,
                    start=1
                ):

                    st.write(
                        f"**{i}. {result}**"
                    )

            else:

                st.info(
                    "No similar questions found."
                )


# ============================================================
# PERFORMANCE
# ============================================================

elif page == "📊 Performance":

    st.title(
        "📊 Interview Performance"
    )

    total = len(
        st.session_state.answers
    )

    evaluations = (
        st.session_state.scores
    )

    st.metric(
        "Questions Attempted",
        total
    )

    if evaluations:

        scores = []

        for evaluation in evaluations:

            match = re.search(
                r"Score:\s*(\d+)\s*/\s*10",
                evaluation,
                re.IGNORECASE
            )

            if match:

                scores.append(
                    int(
                        match.group(1)
                    )
                )

        if scores:

            average = (
                sum(scores) /
                len(scores)
            )

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Average Score",
                    f"{average:.1f}/10"
                )

            with col2:

                st.metric(
                    "Highest Score",
                    f"{max(scores)}/10"
                )

            with col3:

                st.metric(
                    "Lowest Score",
                    f"{min(scores)}/10"
                )

            st.divider()

            st.subheader(
                "📈 Question-wise Performance"
            )

            for i, score in enumerate(
                scores,
                start=1
            ):

                st.write(
                    f"Question {i}: **{score}/10**"
                )

                st.progress(
                    score / 10
                )

            st.divider()

            st.subheader(
                "🧠 Detailed AI Feedback"
            )

            for i, evaluation in enumerate(
                evaluations,
                start=1
            ):

                with st.expander(
                    f"Question {i} Feedback"
                ):

                    st.write(
                        evaluation
                    )

        else:

            st.info(
                "No numerical scores available."
            )

    else:

        st.info(
            "Complete an AI interview to see your performance."
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "AI Interview Preparation Platform | "
    "Streamlit + Ollama + ChromaDB + "
    "Sentence Transformers + PyPDF"
)