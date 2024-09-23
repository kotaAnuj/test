import json
import asyncio

class CrewAgentFramework:
    def __init__(self, architecture, input_spec, output_spec):
        self.architecture = architecture
        self.input_spec = input_spec
        self.output_spec = output_spec
        self.team = self._build_team()

    def _build_team(self):
        # Customize team size and agent roles based on architecture spec
        team_size = self.architecture.get("team_size", 5)
        agent_roles = self.architecture.get("roles", ["generalist"] * team_size)
        return [SubAgent(f"SubAgent_{i}", role, self.input_spec, self.output_spec) 
                for i, role in enumerate(agent_roles)]

    async def coordinate(self, task):
        # Coordinate the task across sub-agents asynchronously
        results = await asyncio.gather(*[agent.perform_task(task) for agent in self.team])
        final_output = self._aggregate_results(results)
        return final_output

    def _aggregate_results(self, results):
        # Aggregating results from all agents
        return "\n".join(results)

class SubAgent:
    def __init__(self, name, role, input_spec, output_spec):
        self.name = name
        self.role = role
        self.input_spec = input_spec
        self.output_spec = output_spec

    async def perform_task(self, task):
        # Create system prompt based on input/output spec and role
        system_prompt = self._construct_system_prompt(task)
        full_prompt = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task}
        ]
        
        # Simulate LLM interaction (replace with actual LLM call in real scenario)
        response = await self._simulate_llm_response(full_prompt)
        return await self._handle_response(response)

    def _construct_system_prompt(self, task):
        return f"""
You are {self.name}, an agent with the following role: {self.role}.
Your task involves: {task}.
Your input spec: {self.input_spec}
Your output spec: {self.output_spec}
"""

    async def _simulate_llm_response(self, prompt):
        # Simulated LLM response (replace this part with real LLM API integration)
        return json.dumps({"response": f"Task completed by {self.name}", "thought": "All good"})

    async def _handle_response(self, response):
        try:
            # Handling the response and returning the parsed result
            parsed_response = json.loads(response)
            if "response" in parsed_response:
                return f"{self.name}: {parsed_response['response']}"
            return f"{self.name}: Task completed with no specific response."
        except json.JSONDecodeError:
            return f"{self.name}: Task failed due to response parsing error."
