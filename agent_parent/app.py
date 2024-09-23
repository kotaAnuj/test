import asyncio
# Usage example with LLMConnector
from llmconnector import LLMConnector
from main import AIAgentCreator, LLMAgentDesigner

api_key = "2d2d8893395640a89cf366271e1a84fe"  # Your API key
creator = AIAgentCreator(api_key)



async def main():
    if __name__ == "__main__":
        llm_connector = LLMConnector(api_key="2d2d8893395640a89cf366271e1a84fe", model_name="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo", base_url="https://api.aimlapi.com")

        designer = LLMAgentDesigner(llm_connector)
        creator = AIAgentCreator()
        task = "Create a customer service chatbot that can handle basic inquiries and route complex issues to human agents."
        try:
            # Use LLMAgentDesigner to design and develop the agent
            requirements = {"task": task}
            design = await designer.design_agent(requirements)
            agent = await designer.develop_agent(design)

            # Use AIAgentCreator to further refine and test the agent
            agent_code, explanation = creator.create_agent(task)
            print("Agent Code:")
            print(agent_code)
            print("\nExplanation:")
            print(explanation)
        except Exception as e:
            print(f"Error: {str(e)}")


asyncio.run(main())