import asyncio
from nlp_processor import NLPProcessor
from memory import Memory
from toolkit import ToolKit
from llm_connector import LLMConnector, create_llm_provider
from rate_limiter import RateLimiter

class HybridAgent:
    def __init__(self, name, role, skills, goals, llm_config, rate_limiter):
        self.name = name
        self.role = role
        self.skills = skills
        self.goals = goals
        self.memory = Memory()
        self.toolkit = ToolKit()
        self.nlp = NLPProcessor()
        self.assigned_tasks = []  # Initialize the assigned_tasks attribute
        
        llm_provider = create_llm_provider(
            llm_config['provider'],
            llm_config['api_key'],
            base_url=llm_config.get('base_url'),
            model=llm_config.get('model')
        )
        self.llm = LLMConnector(llm_provider)
        self.rate_limiter = rate_limiter

    # Inside perform_task method in HybridAgent
    async def perform_task(self, task):
        try:
            analysis = self.nlp.analyze_text(task)
            context = self.memory.get_relevant_info(analysis['keywords'])
            
            plan = await self.generate_plan(task, analysis, context)
            result = await self.execute_plan(plan)
            
            self.memory.add_to_short_term(task, result)
            return result
        except Exception as e:
            return f"Error performing task for {self.name}: {str(e)}"

   
    #added by 40 mini 
    def should_use_tools(self, task):
        # Define logic to determine whether to use tools or generate an AI response
        # For example, use tools for factual or data-driven tasks and AI for general queries
        tool_keywords = {'search', 'calculate', 'locate'}
        return any(keyword in task.lower() for keyword in tool_keywords)
    #added by 40 mini
    async def generate_ai_response(self, task, analysis, context):
        try:
            system_content = (f"You are {self.name}, a {self.role} with skills in {', '.join(self.skills)} "
                              f"and goals to {', '.join(self.goals)}. Provide a detailed response for the task.")
            user_content = (f"Task: {task}\nSentiment: {analysis['sentiment']}\n"
                            f"Keywords: {analysis['keywords']}\nNamed entities: {analysis['named_entities']}\n"
                            f"Context: {context}")
            
            response = await self.rate_limiter.run(self.llm.generate(system_content, user_content))
            return response.strip()
        except Exception as e:
            return f"Error generating AI response for {self.name}: {str(e)}"






    async def generate_plan(self, task, analysis, context):
        try:
            system_content = (f"You are {self.name}, a {self.role} with skills in {', '.join(self.skills)} "
                              f"and goals to {', '.join(self.goals)}. Create a concise 3-step plan for the task.")
            user_content = (f"Task: {task}\nSentiment: {analysis['sentiment']}\n"
                            f"Keywords: {analysis['keywords']}\nNamed entities: {analysis['named_entities']}\n"
                            f"Context: {context}")
            
            plan = await self.llm.generate(system_content, user_content)
            return plan.strip().split('\n')
        except Exception as e:
            print(f"Error generating plan for {self.name}: {str(e)}")
            return ["Analyze task", "Execute task", "Summarize results"]  # Fallback plan

    async def execute_plan(self, plan):
        try:
            results = await asyncio.gather(*[self.execute_step(step) for step in plan])
            final_result = await self.synthesize_results(results)
            return final_result
        except Exception as e:
            return f"Error executing plan for {self.name}: {str(e)}"

    async def execute_step(self, step):
        try:
            tool = self.select_tool(step)
            return await self.toolkit.use_tool(tool, task=step)
        except Exception as e:
            return f"Error executing step '{step}' for {self.name}: {str(e)}"
  
    def select_tool(self, step):
        tool_mapping = {
            "search": "search_tool",
            "calculate": "calculator_tool",
            "locate": "geography_tool",
            "analyze": "analysis_tool"
        }
        for keyword, tool in tool_mapping.items():
            if keyword in step.lower():
                return tool
        return "general_tool"

    async def synthesize_results(self, results):
        try:
            system_content = f"You are {self.name}, tasked with synthesizing information concisely."
            user_content = f"Synthesize these results into a brief, coherent response: {results}"
            return await self.rate_limiter.run(self.llm.generate(system_content, user_content))
        except Exception as e:
            return f"Error synthesizing results for {self.name}: {str(e)}"
