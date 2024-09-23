import asyncio

class AgentCommunication:
    def __init__(self, agents, memory_db):
        self.agents = agents
        self.memory_db = memory_db

    async def distribute_query(self, query):
        responses = {}
        for agent_name, agent in self.agents.items():
            context = self.memory_db.get_context(agent_name)
            if context:
                agent.update_context(context)
            
            responses[agent_name] = await agent.reason_and_act(query)
            
            new_context = agent.get_context()
            self.memory_db.update_context(agent_name, new_context)
        return responses

    async def collaborate(self, query):
        distributed_responses = await self.distribute_query(query)
        combined_response = "\n\n".join([f"{name}: {resp}" for name, resp in distributed_responses.items()])

        # Log the conversations and planning
        conversation_logs = "\n\n".join([agent.get_conversation_log() for agent in self.agents.values()])
        combined_response_with_logs = f"Combined Response:\n{combined_response}\n\nConversation Logs:\n{conversation_logs}"

        return combined_response_with_logs

    def monitor_agents(self):
        for agent_name, agent in self.agents.items():
            print(f"Agent {agent_name} status: {agent.get_status()}")

    def feedback_loop(self):
        pass
