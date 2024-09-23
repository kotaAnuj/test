# agent.py
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
from tools import Tool
from task import Task
from prompts import AGENT_PROMPT_TEMPLATE, TASK_EXECUTION_TEMPLATE, TOOL_USE_TEMPLATE
from llm_connector import LLMConnector

class Agent(BaseModel):
    name: str
    role: str
    goal: str
    backstory: str
    tools: Dict[str, Tool] = Field(default_factory=dict)
    llm_connector: LLMConnector

    class Config:
        arbitrary_types_allowed = True

    async def execute_task(self, task: Task) -> str:
        agent_prompt = AGENT_PROMPT_TEMPLATE.substitute(
            name=self.name,
            role=self.role,
            goal=self.goal,
            backstory=self.backstory,
            tools="\n".join([f"- {name}: {tool.description}" for name, tool in self.tools.items()]),
            task=task.description
        )
        
        llm_response = await self.llm_connector.connect(agent_prompt, task.description)
        return await self._process_llm_response(llm_response, task)

    async def _process_llm_response(self, llm_response: str, task: Task) -> str:
        if "USE_TOOL:" in llm_response:
            tool_instruction = llm_response.split("USE_TOOL:")[1].strip()
            tool_name, *args = tool_instruction.split(maxsplit=1)
            if tool_name in self.tools:
                tool_result = await self.tools[tool_name].execute(*args)
                tool_prompt = TOOL_USE_TEMPLATE.substitute(
                    tool_name=tool_name,
                    tool_result=tool_result['data'] if tool_result['status'] == 'success' else tool_result['message']
                )
                return await self.execute_task(Task(description=task.description + "\n" + tool_prompt))
            else:
                return f"Error: Tool {tool_name} not found."
        else:
            return llm_response
    def add_tool(self, tool: Tool):
        self.tools[tool.name] = tool