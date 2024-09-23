from llm_connector import LLMConnector
import json
import asyncio

class SearchAIAgent:
    def __init__(self, name, api_key, memory_db, model_name="mistralai/Mistral-7B-Instruct-v0.2", base_url="https://api.aimlapi.com"):
        self.name = name
        self.llm_connector = LLMConnector(api_key, model_name, base_url)
        self.character = ""
        self.back_story = ""
        self.task = ""
        self.tools = {}
        self.reasoning = None
        self.memory_db = memory_db
        self.context = ""  # Initialize the context attribute
        self.conversation_log = []
        self.sub_agents = {}

    def set_attributes(self, character, back_story, task, tools, sub_agents=None):
        self.character = character
        self.back_story = back_story
        self.task = task
        self.tools = tools
        if sub_agents:
            self.sub_agents = sub_agents

    def update_context(self, context):
        self.context = context

    def get_context(self):
        return self.context

    async def reason_and_act(self, query):
        # Step 1: Log Reasoning
        reasoning_log = f"Agent {self.name} reasoning: {query}"
        self.conversation_log.append(reasoning_log)

        if self.reasoning:
            self.reasoning(query)

        # If the agent has sub-agents, distribute the query
        if self.sub_agents:
            responses = {}
            for sub_agent_name, sub_agent in self.sub_agents.items():
                self.conversation_log.append(f"Distributing query to {sub_agent_name}...")
                responses[sub_agent_name] = await sub_agent.reason_and_act(query)

            # Aggregate the responses
            final_response = self.aggregate_responses(responses)
            self.conversation_log.append(f"Final response from sub-agents: {final_response}")
            return final_response

        # Step 2: Observation (use the tool)
        if not self.tools:
            error_message = f"Error: No tools found for {self.name}."
            self.conversation_log.append(error_message)
            return error_message

        tool_name, tool = list(self.tools.items())[0]
        tool_response = tool.execute(query)
        tool_log = f"Agent {self.name} used {tool_name} tool: {json.dumps(tool_response)}"
        self.conversation_log.append(tool_log)

        # Step 3: Take Action
        if tool_response["status"] == "success":
            llm_response = await self.llm_connector.connect(self.task, tool_response["data"])
            action_log = f"Agent {self.name} LLM response: {llm_response}"
            self.conversation_log.append(llm_response)
            return llm_response
        else:
            error_message = tool_response["message"]
            self.conversation_log.append(error_message)
            return error_message

    def aggregate_responses(self, responses):
        # This method aggregates responses from sub-agents
        aggregated_response = "\n".join([f"{agent}: {resp}" for agent, resp in responses.items()])
        return aggregated_response

    def set_reasoning(self, reasoning_function):
        self.reasoning = reasoning_function

    def get_status(self):
        return "Active"

    def get_conversation_log(self):
        return "\n".join(self.conversation_log)
