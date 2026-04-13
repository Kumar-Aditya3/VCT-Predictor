from datetime import date
from typing import Optional

from pydantic import BaseModel, Field


class VLRMatchRecord(BaseModel):
    match_id: str
    event_name: str
    region: str
    match_date: date
    team_a: str
    team_b: str
    event_stage: Optional[str] = None
    team_a_maps_won: int = Field(ge=0)
    team_b_maps_won: int = Field(ge=0)
    best_of: int = Field(default=3, ge=1, le=5)
    status: str = "completed"
    source_url: Optional[str] = None


class VLRMapRecord(BaseModel):
    map_id: str
    match_id: str
    map_name: str
    team_a: str
    team_b: str
    team_a_rounds: int = Field(ge=0)
    team_b_rounds: int = Field(ge=0)
    picked_by: Optional[str] = None
    duration_text: Optional[str] = None
    winner_team: Optional[str] = None
    order_index: int = Field(ge=1)


class VLRPlayerStatLine(BaseModel):
    match_id: str
    map_id: str
    map_name: str
    team_name: str
    opponent_team: str
    player_name: str
    agent_name: Optional[str] = None
    rating: Optional[float] = None
    acs: Optional[float] = None
    kills: int = Field(ge=0)
    deaths: int = Field(ge=0)
    assists: int = Field(ge=0)


class VLRMatchDetails(BaseModel):
    match: VLRMatchRecord
    maps: list[VLRMapRecord]
    player_stats: list[VLRPlayerStatLine]
