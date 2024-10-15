from crewai import Crew, Process, Agent, Task
from langchain_community.llms import Ollama
from crewai_tools import PDFSearchTool, FileReadTool
import os
os.environ["OPENAI_API_KEY"] = "NA"

llm = Ollama(
    model="llama3",
    base_url="http://localhost:11434"
)

tool=PDFSearchTool(
    file_path= "Final_Type_Qual_Example.pdf",
    config=dict(
        llm=dict(
            provider="ollama",
            config=dict(
                model="llama3",
                temperature=0.5,
                top_p=1.0,
                stream=False
            ),
        ),
        embedder=dict(
            provider="ollama",
            config=dict(
                model="nomic-embed-text"
            ),
        ),
    )
)

Officer = Agent(
    role="Document Analyzer",
    goal="Read the file and point out important, pertinent, or recurring information within the file to suggest tags for the file.",
    backstory= "you are a civilian engineer who has been working with the navy for years",
    llm=llm,
    tools=[tool]
)

summarize = Task(description='Read final type qualification example PDF',
                        agent=Officer, expected_output='The expected output should be 5-8 one-word tags.')


crew = Crew(
    agents=[Officer],
    tasks=[summarize],
    verbose=True,
)


crew.kickoff()