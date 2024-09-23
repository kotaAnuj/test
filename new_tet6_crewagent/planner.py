from nlp_processor import NLPProcessor
from rate_limiter import RateLimiter

class Planner:
    def __init__(self, crew, rate_limiter: RateLimiter):
        self.crew = crew
        self.nlp = NLPProcessor()
        self.rate_limiter = rate_limiter
        
    async def select_best_agent_for_task(self, task):
        agents = self.crew.get_agents()
        if not agents:
            raise ValueError("No agents available in the crew")
        
        best_agent = None
        best_score = -1
        for agent in agents:
            score = await self.score_agent_for_task(agent, task)
            if score > best_score:
                best_score = score
                best_agent = agent
        return best_agent
    
    async def create_plan(self, task):
        try:
            # Analyzing the task to extract keywords and named entities
            analysis = self.nlp.analyze_text(task)
            
            # Creating a dynamic plan based on the analysis
            plan = [
                f"Analyze task: {task}",
                f"Research key elements: {', '.join(analysis['keywords'])}",
            ]
            
            for entity in analysis['named_entities']:
                plan.append(f"Gather information on {entity[1]}")
            
            plan.extend([
                "Break down task into subtasks",
                "Execute subtasks",
                "Synthesize results"
            ])
            
            # Refining the plan using an LLM
            refined_plan = await self.refine_plan(task, plan)
            
            return refined_plan
        except Exception as e:
            print(f"Error in create_plan: {str(e)}")
            return []

    async def refine_plan(self, task, initial_plan):
        # This method would use the LLM to refine the plan
        # For now, we'll just return the initial plan
        return initial_plan

    async def assign_tasks(self, plan):
        try:
            assignments = []
            for step in plan:
                agent = await self.select_best_agent_for_task(step)
                assignments.append((agent.name, step))
                
                # Update agent's task load
                agent.assigned_tasks.append(step)
                
            return assignments
        except Exception as e:
            print(f"Error in assign_tasks: {str(e)}")
            return []
    
    async def score_agent_for_task(self, agent, task):
        # This method calculates a score based on multiple factors
        task_keywords = set(task.lower().split())
        agent_keywords = set(agent.skills + [agent.role.lower()])
        skill_match_score = len(task_keywords.intersection(agent_keywords))
        
        # New: Consider agent availability (fewer tasks, higher score)
        availability_score = 1 / (len(agent.assigned_tasks) + 1)
        
        # New: Weight the skill match score more heavily
        weighted_score = skill_match_score * 0.7 + availability_score * 0.3
        
        return weighted_score
