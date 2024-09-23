# prompts.py
from string import Template

AGENT_PROMPT_TEMPLATE = Template("""
You are $name, an AI agent with the role of $role.
Your goal is: $goal
Backstory: $backstory

You have access to the following tools:
$tools

Your current task is: $task

Given this information, please process the task and provide your response.
If you need to use a tool, format your response as follows:
USE_TOOL: <tool_name> <tool_arguments>

Begin your work now.
""")

TASK_EXECUTION_TEMPLATE = Template("""
Task: $task_description
Expected Output: $expected_output
Context: $context

Based on your role and capabilities, please execute this task.
If you need to use any tools, specify them clearly in your response.
""")

TOOL_USE_TEMPLATE = Template("""
You used the tool: $tool_name
The result was: $tool_result

Based on this result, please continue your task or provide your next steps.
""")