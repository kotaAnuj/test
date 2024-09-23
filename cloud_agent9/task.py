# task.py
from pydantic import BaseModel
from typing import Optional

class Task(BaseModel):
    description: str
    expected_output: Optional[str] = None
    context: Optional[str] = None