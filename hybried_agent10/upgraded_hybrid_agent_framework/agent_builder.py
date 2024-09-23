# upgraded_hybrid_agent_framework/agent_builder.py

from .tools import ToolRegistry
from .reasoning import ReasoningModule
from .memory import MemoryManager
from .chain import ChainDesigner, ChainStep

class AgentBuilder:
    def __init__(self, name):
        self.name = name
        self.tools = ToolRegistry()
        self.reasoning_engine = ReasoningModule(strategy='hybrid')
        self.memory_manager = MemoryManager()
        self.chain_designer = ChainDesigner()

    def add_tool(self, tool_name, tool_class, config=None):
        tool_instance = tool_class(config=config) if config else tool_class()
        self.tools.register_tool(tool_name, tool_instance)
    
    def set_reasoning_strategy(self, reasoning_class, **kwargs):
        self.reasoning_engine = reasoning_class(**kwargs)
    
    def add_chain_step(self, step_name, task, condition=None):
        self.chain_designer.add_step(ChainStep(step_name, task, condition))

    def build(self):
        return ModularAgentEngine( # type: ignore
            name=self.name,
            tools=self.tools,
            reasoning_engine=self.reasoning_engine,
            memory_manager=self.memory_manager,
            task_manager=None  # Placeholder for future extension
        )

    def run(self, agent):
        self.chain_designer.execute(agent)
