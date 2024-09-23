# agent_crew.py
import asyncio
from memory import Memory

class AgentCrew:
    def __init__(self):
        self.agents = []
        self.shared_memory = Memory()

    def add_agent(self, agent):
        self.agents.append(agent)

    def get_agents(self):
        return self.agents

    def get_agent(self, agent_name):
        for agent in self.agents:
            if agent.name == agent_name:
                return agent
        return None  # Return None if no agent with the given name is found

    async def execute_plan(self, assignments):
        results = []
        for agent_name, step in assignments:
            agent = self.get_agent(agent_name)
            if agent:
                result = await agent.perform_task(step)
                results.append(result)
                self.shared_memory.add_to_short_term(step, result)
                await asyncio.sleep(1)  # Add a 1-second delay between tasks
            else:
                results.append(f"Error: Agent '{agent_name}' not found")
        return results

    def select_agent_for_task(self, task):
        # Simple selection based on skills
        for agent in self.agents:
            if any(skill.lower() in task.lower() for skill in agent.skills):
                return agent
        return self.agents[0] if self.agents else None  # Default to first agent if no match, or None if no agents