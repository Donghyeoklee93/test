# RAG with LangGraph

This project implements and compares three RAG architectures using LangGraph:

- **001. Naive RAG with LangGraph**
- **002. Query Rewrite RAG with LangGraph**
- **003. Web Search RAG with LangGraph (Final Selected Architecture)**

All architectures use a single PDF document as the initial knowledge source:
- Input document:** `data/Deepseek-r1.pdf`


## 0. Project Configuration

```bash
pip install -r requirements.txt
```

## 1. RAG Architectures (003. Web Search RAG with LangGraph (Final))

The Web Search RAG architecture adds:
- **Query Rewrite**
- **Relevance Check (Groundedness)**
- **Web Search Fallback (Tavily)**

**State Definition**
```bash
class GraphState(TypedDict):
    question: Annotated[List[str], add_messages]
    context: Annotated[str, "Context"]
    answer: Annotated[str, "Answer"]
    messages: Annotated[list, add_messages]
    relevance: Annotated[str, "Relevance"]
```

**PDF Retrieval Node**
```bash
def retrieve_document(state: GraphState) -> GraphState:
    latest_question = state["question"][-1].content
    retrieved_docs = pdf_retriever.invoke(latest_question)
    retrieved_docs = format_docs(retrieved_docs)
    return {"context": retrieved_docs}
```

**Relevance Check Node**
```bash
from tools.evaluator import GroundednessChecker

def relevance_check(state: GraphState) -> GraphState:
    question_answer_relevant = GroundednessChecker(
        llm=ChatOpenAI(model="gpt-4o-mini", temperature=0),
        target="question-retrieval"
    ).create()

    response = question_answer_relevant.invoke(
        {"question": state["question"][-1].content, "context": state["context"]}
    )

    return {"relevance": response.score}
```

**Web Search Node**
```bash
def web_search(state: GraphState) -> GraphState:
    tavily_tool = TavilySearch()
    latest_question = state["question"][-1].content

    search_result = tavily_tool.search(
        query=latest_question,
        topic="general",
        max_results=6,
        format_output=True,
    )

    return {"context": search_result}
```



