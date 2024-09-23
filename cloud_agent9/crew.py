# crew.py
from pydantic import BaseModel
from typing import List
from agent import Agent
from task import Task

class Crew(BaseModel):
    agents: List[Agent]
    tasks: List[Task] = []

    async def execute_tasks(self) -> List[str]:
        results = []
        for task in self.tasks:
            for agent in self.agents:
                result = await agent.execute_task(task)
                results.append(f"Agent {agent.name}: {result}")
        return results

    def assign_task(self, task: Task):
        self.tasks.append(task)