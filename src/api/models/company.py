from pydantic import BaseModel
from datetime import datetime
from typing import Optional

class Company(BaseModel):
    id: str
    name: str
    code: str
    created_at: datetime
    is_active: bool = True 