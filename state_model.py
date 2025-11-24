# state_model.py

from typing import List, Dict, Optional
from pydantic import BaseModel, Field


class Contact(BaseModel):
    name: str
    title: Optional[str] = None
    email: Optional[str] = None
    confidence: Optional[float] = None
    source: Optional[str] = None


class Company(BaseModel):
    name: str
    domain: Optional[str] = None
    score: Optional[int] = None
    metadata: Dict = Field(default_factory=dict)
    contacts: List[Contact] = Field(default_factory=list)


class OutreachState(BaseModel):

    # User inputs
    query: str = ""
    email_style: str = "Professional"
    offering: str = ""
    max_companies: int = 5 

    # Pipelines outputs
    companies: List[Company] = Field(default_factory=list)
    research: Dict[str, dict] = Field(default_factory=dict)
    emails: Dict[str, str] = Field(default_factory=dict)

    # Errors
    errors: List[str] = Field(default_factory=list)
