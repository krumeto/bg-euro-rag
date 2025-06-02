import asyncio
import argparse
from openai import OpenAI
from dotenv import load_dotenv
import os

from src.settings import AGENT_MODEL, EURO_RAG_SYSTEM_PROMPT
from src.search_qa import search_qa, search_bnb_law

# Initialize OpenAI client
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

async def process_query(query: str) -> str:
    """Process a user query using RAG flow with parallel searches"""
    # 1. Perform searches in parallel
    q_and_a_task = asyncio.create_task(asyncio.to_thread(search_qa, query))
    bnb_law_task = asyncio.create_task(asyncio.to_thread(search_bnb_law, query))
    
    # Wait for both searches to complete
    q_and_a_results, bnb_law_results = await asyncio.gather(q_and_a_task, bnb_law_task)
    
    # 2. Format the system prompt with retrieved information
    formatted_prompt = f"""## User question: {query}

### Closest Q and A documents from the Bulgarian National Bank:
{q_and_a_results}

### Closest articles from the Bulgarian National Bank law:
{bnb_law_results}

### Your answer (according to your instructions):
"""
    
    # 3. Call the OpenAI API
    response = client.chat.completions.create(
        model=AGENT_MODEL,
        messages=[
            {"role": "system", "content": EURO_RAG_SYSTEM_PROMPT},
            {"role": "user", "content": formatted_prompt}
        ],
        temperature=0.3,
        max_tokens=1024
    )
    
    # 4. Return the response
    return response.choices[0].message.content

async def main():
    parser = argparse.ArgumentParser(description="RAG-based Euro information assistant")
    parser.add_argument('--question', type=str, help='Ask a single question (non-interactive mode)')
    args = parser.parse_args()
    
    if args.question:
        # Single question mode
        answer = await process_query(args.question)
        print(answer)
        return
        
    # Interactive conversation mode
    print("Welcome to the Euro adoption information assistant.")
    print("Ask a question about Bulgaria's adoption of the Euro or type 'exit' to quit.\n")
    
    while True:
        user_input = input("> ")
        
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("Thank you for using the Euro information assistant. Goodbye!")
            break
            
        answer = await process_query(user_input)
        print("\nAssistant:")
        print(answer)
        print("\nAsk another question or type 'exit' to quit:")

if __name__ == "__main__":
    asyncio.run(main())
