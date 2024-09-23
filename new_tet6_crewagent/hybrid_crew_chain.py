# hybrid_crew_chain.py
import asyncio
from agent_crew import AgentCrew
from toolkit import ToolKit
from planner import Planner
from hybrid_agent import HybridAgent
from rate_limiter import RateLimiter

class HybridCrewChain:
    def __init__(self, llm_config, calls_per_minute=10, burst_limit=15):
        self.llm_config = llm_config
        self.rate_limiter = RateLimiter(calls_per_minute, burst_limit)
        self.crew = AgentCrew()
        self.toolkit = ToolKit()
        self.planner = Planner(self.crew, self.rate_limiter)

    def add_agent(self, name, role, skills, goals):
        agent = HybridAgent(name, role, skills, goals, self.llm_config, self.rate_limiter)
        agent.toolkit = self.toolkit  # Share the same toolkit across all agents
        self.crew.add_agent(agent)

    async def run(self, task):
        try:
            print("Creating plan...")
            plan = await self.rate_limiter.run(self.planner.create_plan(task))
            print(f"Plan created: {plan}")

            print("Assigning tasks...")
            assignments = await self.rate_limiter.run(self.planner.assign_tasks(plan))
            print(f"Tasks assigned: {assignments}")

            print("Executing plan...")
            results = await self.execute_plan(assignments)
            print(f"Execution results: {results}")

            print("Synthesizing results...")
            return await self.rate_limiter.run(self.synthesize_results(results))
        except Exception as e:
            print(f"Error in crew execution: {str(e)}")
            raise

    async def execute_plan(self, assignments):
        tasks = []
        for agent_name, step in assignments:
            agent = self.crew.get_agent(agent_name)
            if agent:  # Ensure agent exists
                tasks.append(agent.perform_task(step))
        try:
            # Await all tasks and handle cancellations
            return await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            print("A task was cancelled.")
            raise
        except Exception as e:
            return [f"Error in executing plan: {str(e)}"]

    async def synthesize_results(self, results):
        system_content = "You are an AI tasked with synthesizing multiple agents' outputs into a coherent response."
        user_content = f"Synthesize these agents' results into a concise, informative response: {results}"
        
        synthesizer = self.crew.get_agents()[0]  # Use the first agent's LLM for synthesis
        try:
            return await self.rate_limiter.run(synthesizer.llm.generate(system_content, user_content))
        except Exception as e:
            return f"Error in synthesizing results: {str(e)}"
        
    def format_output(self, results):
        return "\n".join(str(result) for result in results)