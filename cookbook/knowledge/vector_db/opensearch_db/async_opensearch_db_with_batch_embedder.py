import asyncio

from agno.agent import Agent
from agno.knowledge.embedder.openai import OpenAIEmbedder
from agno.knowledge.knowledge import Knowledge
from agno.models.openai import OpenAIChat
from agno.vectordb.opensearch import OpensearchDb

vector_db = OpensearchDb(
    index_name="recipes_batch",
    dimension=1536,
    hosts=[
        {
            "host": "localhost",
            "port": 9200,
        }
    ],
    # Uncomment the following line to use basic authentication
    # http_auth=("username", "password"),
    embedder=OpenAIEmbedder(enable_batch=True)
)

knowledge_base = Knowledge(
    vector_db=vector_db,
)

agent = Agent(model=OpenAIChat(id="gpt-4o-mini"), knowledge=knowledge_base)

if __name__ == "__main__":
    # Comment out after first run
    asyncio.run(
        knowledge_base.add_content_async(
            url="https://docs.agno.com/concepts/agents/introduction.md"
        )
    )

    # Create and use the agent
    asyncio.run(
        agent.aprint_response("What is the purpose of an Agno Agent?", markdown=True)
    )
