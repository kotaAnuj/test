import tensorflow as tf
import json
import subprocess
import logging
from ai_agent_memory import AIAgentMemory
from llm_connector import LLMConnector

class AIAgentFramework:
    def __init__(self, name, role, task_type, tools, llm_connector: LLMConnector, memory: AIAgentMemory):
        self.name = name
        self.role = role
        self.task_type = task_type  # e.g., "neural_network", "chatbot", "crew_agent"
        self.tools = tools
        self.llm_connector = llm_connector
        self.memory = memory
        self.conversation_history = self.memory.load_conversation_history()
        
        # Initialize model if task requires neural network processing
        if task_type == "neural_network":
            self.model = self._build_model()

    def _build_model(self):
        input_spec = (10,)  # Example input specification
        output_spec = 1  # Example output specification
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=input_spec),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(output_spec, activation='linear')
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    async def process_input(self, user_input: str) -> str:
        system_prompt = self._construct_system_prompt()
        self.conversation_history.append({"role": "user", "content": user_input})
        
        full_prompt = [{"role": "system", "content": system_prompt}]
        
        # Append formatted conversation history to prompt
        for entry in self.conversation_history:
            if isinstance(entry, dict) and "role" in entry and "content" in entry:
                full_prompt.append(entry)

        # Get LLM response
        response = await self.llm_connector.connect(full_prompt)
        logging.info(f"LLM response: {response}")

        # Process response (either run a tool or return the final output)
        final_response, reasoning = await self._process_response(response)
        
        self.conversation_history.append({"role": "assistant", "content": final_response})
        self.memory.save_message("user", user_input)
        self.memory.save_message("assistant", final_response, reasoning)

        return final_response

    def _construct_system_prompt(self) -> str:
        tool_descriptions = "\n".join([f"- {tool['name']}: {tool['description']}" for tool in self.tools])
        return f"""You are {self.name}, an AI assistant with the following role and task type:

Role: {self.role}
Task Type: {self.task_type}

You have access to the following tools:
{tool_descriptions}

To use a tool, respond with a JSON object in the following format:
{{
    "thought": "your reasoning for using the tool",
    "action": "tool_name",
    "action_input": "input for the tool"
}}

If you don't need to use a tool, simply respond normally.
"""

    async def _process_response(self, response: str) -> tuple:
        try:
            # Parse the response as JSON if a tool needs to be used
            parsed_response = json.loads(response)
            if "action" in parsed_response:
                tool_name = parsed_response["action"]
                tool_input = parsed_response["action_input"]
                reasoning = parsed_response.get("thought", "No reasoning provided.")

                # Find and execute the tool
                for tool in self.tools:
                    if tool["name"] == tool_name:
                        tool_result = await tool["function"](tool_input)
                        return f"Tool '{tool_name}' executed. Result: {tool_result}", reasoning
                return f"No tool named '{tool_name}' is available.", None
            else:
                # If not a tool request, treat the response as a terminal command or final output
                tool_result = await self._execute_terminal_command(response)
                return tool_result, None

        except json.JSONDecodeError:
            # Return the response as is if it's not in JSON format
            return response, None

    async def _execute_terminal_command(self, command):
        try:
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            return f"Result: {result.stdout}"
        except subprocess.CalledProcessError as e:
            return f"Error: {e}"
