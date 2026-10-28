import os
from crewai import Agent, Crew, Process, Task
from crewai.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from duckduckgo-search import DDGS

os.environ["GEMINI_API_KEY"] = "YOUR_GEMINI_API_KEY"

@tool("Internet Search Tool")
def internet_search(query: str) -> str:
    """Search the web for a given query and return top results."""
    try:
        with DDGS() as ddgs:
            results = [r for r in ddgs.text(query, max_results=3)]
            return str(results)
    except Exception as e:
        return f"An error occurred during search: {e}"

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.5,
    google_api_key=os.environ["GEMINI_API_KEY"]
)

researcher = Agent(
    role="Market Research Lead",
    goal="Identify high-potential business prospects in a specific niche using live web data and analyze their real operational pain points.",
    backstory="You are an expert market analyst with a knack for unearthing obscure operational bottlenecks in target industries via web research.",
    tools=[internet_search],
    llm=llm,
    verbose=True
)

writer = Agent(
    role="B2B Copywriting Specialist",
    goal="Write hyper-personalized, non-spammy cold outreach emails that focus purely on delivering value.",
    backstory="You are a world-class sales copywriter who crafts concise, high-converting messages.",
    llm=llm,
    verbose=True
)

target_niche = "Mid-sized digital marketing agencies in Europe looking to offer AI automation services to their clients."

task_research = Task(
    description=f"Search the web for current trends regarding: '{target_niche}'. Identify real market conditions, compliance hurdles (like EU AI regulations), and list 3 critical operational inefficiencies they face today.",
    expected_output="A detailed bulleted report outlining current market realities and 3 specific, verified pain points.",
    agent=researcher
)

task_write = Task(
    description="Take the research report provided by the researcher. Draft a cold outreach email pitching custom AI Agent development services under 150 words.",
    expected_output="A polished, copy-paste ready cold email script.",
    agent=writer
)

crew = Crew(
    agents=[researcher, writer],
    tasks=[task_research, task_write],
    process=Process.sequential,
    verbose=True
)

if __name__ == "__main__":
    print("## Starting B2B Lead Generation Crew Execution ##")
    result = crew.kickoff()
    print("\n\n########################")
    print("## RESULT OUTPUT ##")
    print("########################\n")
    print(result)
