from pydantic import BaseModel, EmailStr
from datetime import datetime
from typing import Optional

class UserCreate(BaseModel):
    username: str
    email: EmailStr
    password: str
    company_code: Optional[str] = None

class UserResponse(BaseModel):
    id: str
    username: str
    email: str
    company_id: str
    created_at: datetime
    is_active: bool = True 