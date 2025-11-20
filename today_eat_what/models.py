from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class RecipeStep(BaseModel):
    order: int
    instruction: str


class Recipe(BaseModel):
    name: str
    description: str
    ingredients: List[str] = Field(default_factory=list)
    steps: List[RecipeStep] = Field(default_factory=list)
    meal_type: str


class AuditResult(BaseModel):
    ok: bool
    reasons: Optional[List[str]] = None


class PublishResult(BaseModel):
    success: bool
    post_id: Optional[str] = None
    detail: Optional[Dict[str, Any]] = None
