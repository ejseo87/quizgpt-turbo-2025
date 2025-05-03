import streamlit as st
import os
import json
from langchain.retrievers import WikipediaRetriever
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import UnstructuredFileLoader
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.schema.runnable import RunnablePassthrough

st.set_page_config(
    page_title="QuizGPT Turbo",
    page_icon="❓",
)

# Create necessary directories in /tmp
CACHE_DIR = "./.cache"
os.makedirs("./.cache", exist_ok=True)
os.makedirs(os.path.join(CACHE_DIR, "files"), exist_ok=True)

function = {
    "name": "create_quiz",
    "description": "function that takes a list of questions and answers "
    "and returns a quiz",
    "parameters": {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                        },
                        "answers": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "answer": {
                                        "type": "string",
                                    },
                                    "correct": {
                                        "type": "boolean",
                                    },
                                },
                                "required": ["answer", "correct"],
                            },
                        },
                    },
                    "required": ["question", "answers"],
                },
            }
        },
        "required": ["questions"],
    },
}


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


questions_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
    You are a helpful assistant that is role playing as a teacher.

    Based ONLY on the following context make 10 questions
    to test the user's knowledge about the text.

    Each question should have 4 answers,
    three of them must be incorrect answers
    and one should be a correct answer.
    The correct answer must be chosen randomly from the four answers.

    Context: {context}

    The quizzes have three difficulty levels: easy, medium, and hard.
    You should create 10 questions with 4 answers for the following level.

    Level: {level}
""",
        )
    ]
)


@st.cache_data(show_spinner="Splitting file...")
def split_file(file):
    file_content = file.read()
    file_path = f"{CACHE_DIR}/files/{file.name}"
    # Save the file
    with open(file_path, "wb") as f:
        f.write(file_content)
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    return docs


@st.cache_data(show_spinner="Making quiz...")
def run_quiz_chain(_docs, level, topic):
    chain = {
        "context": format_docs,
        "level": RunnablePassthrough(),
    } | questions_prompt | llm
    return chain.invoke(_docs)


@st.cache_data(show_spinner="Searching Wikipedia...")
def wiki_search(topic):
    retriever = WikipediaRetriever()
    docs = retriever.get_relevant_documents(topic)
    return docs


with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Key", type="password")

# Initialize ChatOpenAI with API key from user input
if openai_api_key:
    llm = ChatOpenAI(
        temperature=0.1,
        streaming=True,
        callbacks=[
            StreamingStdOutCallbackHandler(),
        ],
        model="gpt-3.5-turbo",
        openai_api_key=openai_api_key
    ).bind(
        function_call={
            "name": "create_quiz",
        },
        functions=[
            function,
        ]
    )
else:
    st.warning("Please enter your OpenAI API key to continue.")
    st.stop()

with st.sidebar:
    docs = None
    topic = None  # Initialize topic variable
    level = st.selectbox(
        "Select the level of the quiz",
        ["Easy", "Medium", "Hard"]
    )
    st.write("You selected: ", level, " level")
    choice = st.selectbox(
        "Choose what you want to use.",
        (
            "File",
            "Wikepedia Article",
        )
    )
    if choice == "File":
        file = st.file_uploader(
            "Upload a .docx, .txt, or .pdf file",
            type=["txt", "pdf", "docx"]
        )
        if file:
            docs = split_file(file)
    else:
        topic = st.text_input("Search Wikipedia for...")
        if topic:
            docs = wiki_search(topic)
if not docs:
    st.markdown(
        """
      # Welcome to QuizGPT!

      I will make a quiz from Wikipedia articles
      or files you upload to test your knowledge and help you study.

      Get started by uploading a file or searching Wikipedia in the sidebar.
      """
    )
else:
    st.title("Take the quiz")
    response = run_quiz_chain(docs, level, topic if topic else file.name)
    response = response.additional_kwargs["function_call"]["arguments"]
    response = json.loads(response)
    st.write(response)
    correct_answers = 0

    with st.form("qustions_form"):
        number = 0
        for quiz in response["questions"]:
            number += 1
            st.write(f"{number}. {quiz['question']}")
            value = st.radio(
                "Select a option.",
                [answer["answer"]for answer in quiz["answers"]],
                index=None,
                key=f"question_{number}"
            )
            if {"answer": value, "correct": True} in quiz["answers"]:
                st.success("Correct!")
                correct_answers += 1
            elif value is not None:
                st.error("Wrong answer!")

        button = st.form_submit_button("Submit")
        if button:
            if correct_answers == len(response["questions"]):
                st.balloons()
            with st.sidebar:
                st.write(
                    f"You got {correct_answers} out of "
                    f"{len(response['questions'])} questions correct."
                )
