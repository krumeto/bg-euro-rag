from agents import Agent, Runner, function_tool
import asyncio
import argparse
from dotenv import load_dotenv
from src.settings import AGENT_MODEL, DEFAULT_QUESTION, EURO_AGENT_INSTRUCTIONS

load_dotenv()

import logging
logging.getLogger("openai.agents").setLevel(logging.DEBUG)
logging.getLogger("openai.agents").addHandler(logging.StreamHandler())
from src.search_qa import search_qa, search_bnb_law

search_euro_qa = function_tool(strict_mode=False)(search_qa)
search_bnb_law = function_tool(strict_mode=False)(search_bnb_law)

# #TODO a tool to read the src/resources/q_and_a.pdf file and return the content as a string
# euro_q_and_a = function_tool(strict_mode=False)(lambda: "This is a placeholder for the Euro Q&A content.")
# #TODO a tool to read the src/resources/newbnblaw_bg.pdf file and return the content as a string
# newbnblaw_q_and_a = function_tool(strict_mode=False)(lambda: "This is a placeholder for the New BNB Law Q&A content.")


# extract_agent = Agent(
#     name="Text Extractor",
#     instructions="""You are an expert at reading and extracting information about Bulgaria's process of adopting the Euro. """,
#     tools=[search_with_sources],
# )

# geopolitical_agent = Agent(
#     name="Geo, the geopolitical agent",
#     instructions="""You are an expert on geopolitics. You will start your response with 'Geo:'. 
#     You must run the content_hybrid_search tool to get the latest news on geopolitics and answer only based on the results.""",
#     tools=[search_with_sources],
# )

# euro_specialist_agent = Agent(
#     name="Specialist on Euro adoption"
#     instructions="You are an expert at reading and extracting information about Bulgaria's process of adopting the Euro. ",
#     handoffs=[financial_agent, geopolitical_agent],
# )

extract_agent = Agent(
    name="Text Extractor",
    model=AGENT_MODEL,
    instructions=EURO_AGENT_INSTRUCTIONS,
    tools=[search_euro_qa, search_bnb_law],
)

async def conversational_mode():
    print("Starting conversational mode. Type 'exit' or 'quit' to end the conversation.\n")
    
    # Initialize the runner with the agent
    runner = Runner(extract_agent)
    
    # Welcome message
    print("Ask a question about Bulgaria's adoption of the Euro:")
    
    while True:
        # Get user input
        user_input = input("> ")
        
        # Check if user wants to exit
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("Thank you for using the Euro information assistant. Goodbye!")
            break
            
        # Process the user's question
        result = await runner.run(input=user_input)
        
        # Display the response
        print("\nAssistant:")
        print(result.final_output)
        print("\nAsk another question or type 'exit' to quit:")

async def main():
    parser = argparse.ArgumentParser(description="Run Euro information assistant")
    parser.add_argument('--question', type=str, help='Single question mode: Ask one question and exit')
    parser.add_argument('--interactive', action='store_true', help='Start in interactive conversational mode')
    args = parser.parse_args()
    
    if args.question:
        # Single question mode (original behavior)
        result = await Runner.run(extract_agent, input=args.question)
        print(result.final_output)
    else:
        # Default to conversational mode
        await conversational_mode()

if __name__ == "__main__":
    asyncio.run(main())