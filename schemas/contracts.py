"""
Shared data contracts between pipeline stages.
"""
from __future__ import annotations
from typing import Literal, Optional
from pydantic import BaseModel, Field


class Entities(BaseModel):
    order_id: Optional[str] = None
    product_name: Optional[str] = None


class NLPResult(BaseModel):
    """Output of Person 1's extractor"""
    pass


class RouteDecision(BaseModel):
    """Output of Person 3's router"""
    pass


class ResolutionResult(BaseModel):
    """Output of Person 3's resolution check"""
    pass


class SummaryResult(BaseModel):
    """Output of Person 4's summarizer"""
    summary: str


class Ticket(BaseModel):
    """Output of Person 3's ticket logic. Consumed by Person 4's dashboard."""
    pass
