"""Request/response contracts for the public API."""
from __future__ import annotations

import datetime as dt
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

Education = Literal["Basic", "Graduation", "2n Cycle", "Master", "PhD"]
MaritalStatus = Literal["Married", "Together", "Single", "Divorced", "Widow", "Unknown"]


class CustomerInput(BaseModel):
    """One customer in the raw training schema. Bounds reject impossible values at the edge."""

    model_config = ConfigDict(extra="forbid")

    Year_Birth: int = Field(ge=1920, le=2010)
    Income: float | None = Field(default=None, ge=0, le=1_000_000)
    Kidhome: int = Field(ge=0, le=10)
    Teenhome: int = Field(ge=0, le=10)
    Recency: int = Field(ge=0, le=3650, description="Days since last purchase")
    MntWines: float = Field(ge=0, le=100_000)
    MntFruits: float = Field(ge=0, le=100_000)
    MntMeatProducts: float = Field(ge=0, le=100_000)
    MntFishProducts: float = Field(ge=0, le=100_000)
    MntSweetProducts: float = Field(ge=0, le=100_000)
    MntGoldProds: float = Field(ge=0, le=100_000)
    NumDealsPurchases: int = Field(ge=0, le=1_000)
    NumWebPurchases: int = Field(ge=0, le=1_000)
    NumCatalogPurchases: int = Field(ge=0, le=1_000)
    NumStorePurchases: int = Field(ge=0, le=1_000)
    NumWebVisitsMonth: int = Field(ge=0, le=500)
    AcceptedCmp1: bool = False
    AcceptedCmp2: bool = False
    AcceptedCmp3: bool = False
    AcceptedCmp4: bool = False
    AcceptedCmp5: bool = False
    Response: bool = False
    Complain: bool = False
    Education: Education = "Graduation"
    Marital_Status: MaritalStatus = "Married"
    Dt_Customer: dt.date


class Recommendation(BaseModel):
    segment: int
    name: str
    summary: str
    goal: str
    defining_high_traits: list[str]
    defining_low_traits: list[str]
    preferred_channel: str
    over_indexed_categories: list[str]
    actions: list[str]
    kpis: list[str]
    responsible_use: str


class AssignmentResult(BaseModel):
    segment: int
    segment_name: str
    assignment_margin: float
    fit: Literal["clear", "moderate", "borderline"]
    distances: dict[str, float]
    customer_features: dict[str, float]
    segment_average: dict[str, float]
    recommendation: Recommendation
    warnings: list[str]


class HealthResponse(BaseModel):
    status: Literal["ok"]
    version: str
    model_loaded: bool
    n_segments: int
    trained_at_utc: str
    model_source: Literal["original", "published"] = "original"
    mode: Literal["customer", "generic"] = "customer"
    labels: dict[str, str | None] = Field(default_factory=dict)


CellValue = float | Annotated[str, Field(max_length=100)] | None


class GenericRecord(BaseModel):
    """One row for a workspace built from the user's own dataset (columns vary by dataset)."""

    model_config = ConfigDict(extra="forbid")

    values: dict[Annotated[str, Field(max_length=120)], CellValue] = Field(max_length=300)


DraftId = Annotated[str, Field(pattern=r"^[a-f0-9]{32}$")]


class ChatTurn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "assistant"]
    content: str = Field(max_length=4000)


class AgentMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str = Field(max_length=2000)
    draft_id: DraftId | None = None
    history: list[ChatTurn] = Field(default_factory=list, max_length=12)


class DraftRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    draft_id: DraftId


class Condition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    feature: str = Field(max_length=40, pattern=r"^[A-Za-z_]+$")
    op: Literal[">", ">=", "<", "<=", "==", "!="]
    value: float | Annotated[str, Field(max_length=40)]


class DatasetRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["filtered", "synthetic", "summary", "template", "result", "report"]
    draft_id: DraftId | None = None
    token: DraftId | None = None
    segments: list[Annotated[int, Field(ge=0, le=20)]] = Field(default_factory=list, max_length=10)
    conditions: list[Condition] = Field(default_factory=list, max_length=10)
    n: int = Field(default=500, ge=10, le=20_000)
    seed: int = Field(default=42, ge=0, le=2**31 - 1)
    table: Literal["profiles", "index", "zscores", "channel_mix", "category_mix", "k_selection", "cleaning", "recommendations"] = "profiles"
