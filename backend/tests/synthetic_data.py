from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
import random

from app.models.vlr import VLRMapRecord, VLRMatchDetails, VLRMatchRecord, VLRPlayerStatLine


TIER1_REGIONS: tuple[str, ...] = ("Americas", "EMEA", "Pacific", "China")
GLOBAL_EVENTS: tuple[str, ...] = ("Masters", "Champions")
REGIONAL_EVENTS: tuple[str, ...] = ("Kickoff", "Split 1", "Split 2")
MAP_POOL: tuple[str, ...] = ("Ascent", "Bind", "Haven", "Icebox", "Lotus", "Pearl", "Split", "Sunset")
STAGES: tuple[str, ...] = ("Group Stage", "Playoffs", "Upper Bracket", "Lower Bracket", "Grand Final")
AGENTS: tuple[str, ...] = ("Jett", "Raze", "Sova", "Omen", "Killjoy", "Viper", "Fade", "Skye")

TEAM_POOLS: dict[str, tuple[str, ...]] = {
    "Americas": (
        "Sentinels",
        "Cloud9",
        "Evil Geniuses",
        "Leviatan",
        "KRU Esports",
        "NRG",
    ),
    "EMEA": (
        "FNATIC",
        "Team Liquid",
        "Team Heretics",
        "Karmine Corp",
        "BBL Esports",
        "NAVI",
    ),
    "Pacific": (
        "Paper Rex",
        "Gen.G",
        "DRX",
        "T1",
        "RRQ",
        "ZETA DIVISION",
    ),
    "China": (
        "EDward Gaming",
        "Bilibili Gaming",
        "Trace Esports",
        "Tyloo",
        "FPX",
        "Nova Esports",
    ),
}


@dataclass(slots=True)
class SyntheticVCTDataset:
    records: list[VLRMatchRecord]
    details: list[VLRMatchDetails]
    maps: list[VLRMapRecord]
    player_stats: list[VLRPlayerStatLine]


def generate_synthetic_vct_dataset(
    match_count: int = 30,
    *,
    seed: int = 42,
    start_date: date | None = None,
) -> SyntheticVCTDataset:
    if match_count < 1:
        return SyntheticVCTDataset(records=[], details=[], maps=[], player_stats=[])

    rng = random.Random(seed)
    base_date = start_date or date(2026, 1, 1)

    records: list[VLRMatchRecord] = []
    details: list[VLRMatchDetails] = []
    all_maps: list[VLRMapRecord] = []
    all_player_stats: list[VLRPlayerStatLine] = []

    for match_index in range(match_count):
        event_name = _pick_event_name(match_index, rng)
        region = _pick_region(event_name, match_index, rng)
        team_a, team_b = _pick_teams(region, match_index)
        match_date = base_date + timedelta(days=match_index * 2)
        best_of = 3 if event_name in REGIONAL_EVENTS else (5 if match_index % 4 == 0 else 3)
        maps_to_play = best_of // 2 + 1

        team_a_strength = _team_strength(team_a)
        team_b_strength = _team_strength(team_b)
        team_a_expected = team_a_strength / (team_a_strength + team_b_strength)
        team_a_wins_match = rng.random() < team_a_expected
        winning_maps = maps_to_play
        losing_maps = rng.randint(0, maps_to_play - 1)
        team_a_maps_won = winning_maps if team_a_wins_match else losing_maps
        team_b_maps_won = losing_maps if team_a_wins_match else winning_maps

        record = VLRMatchRecord(
            match_id=f"synthetic-{match_index + 1:03d}",
            event_name=event_name,
            region=region,
            match_date=match_date,
            team_a=team_a,
            team_b=team_b,
            event_stage=_pick_stage(match_index, rng),
            team_a_maps_won=team_a_maps_won,
            team_b_maps_won=team_b_maps_won,
            best_of=best_of,
            source_url=f"https://synthetic.local/match/{match_index + 1:03d}",
        )
        records.append(record)

        match_maps, match_player_stats = _build_match_details(record, rng, team_a_wins_match, winning_maps, losing_maps)
        all_maps.extend(match_maps)
        all_player_stats.extend(match_player_stats)
        details.append(VLRMatchDetails(match=record, maps=match_maps, player_stats=match_player_stats))

    return SyntheticVCTDataset(records=records, details=details, maps=all_maps, player_stats=all_player_stats)


def _pick_event_name(match_index: int, rng: random.Random) -> str:
    if match_index % 5 == 0:
        return rng.choice(GLOBAL_EVENTS)
    return REGIONAL_EVENTS[match_index % len(REGIONAL_EVENTS)]


def _pick_region(event_name: str, match_index: int, rng: random.Random) -> str:
    if event_name in GLOBAL_EVENTS:
        return "International" if match_index % 2 == 0 else rng.choice(TIER1_REGIONS)
    return TIER1_REGIONS[match_index % len(TIER1_REGIONS)]


def _pick_stage(match_index: int, rng: random.Random) -> str:
    return STAGES[(match_index + rng.randint(0, len(STAGES) - 1)) % len(STAGES)]


def _pick_teams(region: str, match_index: int) -> tuple[str, str]:
    if region == "International":
        pools = list(TEAM_POOLS.values())
        first_pool = pools[match_index % len(pools)]
        second_pool = pools[(match_index + 1) % len(pools)]
        return first_pool[match_index % len(first_pool)], second_pool[(match_index + 2) % len(second_pool)]

    pool = TEAM_POOLS[region]
    first = pool[match_index % len(pool)]
    second = pool[(match_index + 2) % len(pool)]
    if first == second:
        second = pool[(match_index + 3) % len(pool)]
    return first, second


def _team_strength(team_name: str) -> float:
    baseline = sum(ord(char) for char in team_name if char.isalnum())
    return 0.8 + (baseline % 35) / 100.0


def _build_match_details(
    record: VLRMatchRecord,
    rng: random.Random,
    team_a_wins_match: bool,
    winning_maps: int,
    losing_maps: int,
) -> tuple[list[VLRMapRecord], list[VLRPlayerStatLine]]:
    maps_to_play = winning_maps + losing_maps
    map_names = list(MAP_POOL)
    rng.shuffle(map_names)
    map_names = map_names[:maps_to_play]

    winning_team = record.team_a if team_a_wins_match else record.team_b
    losing_team = record.team_b if team_a_wins_match else record.team_a
    map_winners = [winning_team] * winning_maps + [losing_team] * losing_maps
    rng.shuffle(map_winners)

    maps: list[VLRMapRecord] = []
    player_stats: list[VLRPlayerStatLine] = []

    for map_index, (map_name, winner_team) in enumerate(zip(map_names, map_winners, strict=True), start=1):
        picked_by = record.team_a if map_index % 2 == 1 else record.team_b
        if winner_team == record.team_a:
            team_a_rounds = 13
            team_b_rounds = rng.randint(6, 11)
        else:
            team_b_rounds = 13
            team_a_rounds = rng.randint(6, 11)

        maps.append(
            VLRMapRecord(
                map_id=f"{record.match_id}:map-{map_index}",
                match_id=record.match_id,
                map_name=map_name,
                team_a=record.team_a,
                team_b=record.team_b,
                team_a_rounds=team_a_rounds,
                team_b_rounds=team_b_rounds,
                picked_by=picked_by,
                duration_text=f"{35 + map_index:02d}:00",
                winner_team=winner_team,
                order_index=map_index,
            )
        )

        player_stats.extend(_build_player_stats(record, map_name, winner_team, rng, map_index))

    return maps, player_stats


def _build_player_stats(
    record: VLRMatchRecord,
    map_name: str,
    winner_team: str,
    rng: random.Random,
    map_index: int,
) -> list[VLRPlayerStatLine]:
    stats: list[VLRPlayerStatLine] = []
    for team_name in (record.team_a, record.team_b):
        team_won_map = team_name == winner_team
        team_strength = _team_strength(team_name)
        opponent_strength = _team_strength(record.team_b if team_name == record.team_a else record.team_a)
        roster = [f"{team_name} P{player_index}" for player_index in range(1, 6)]
        for player_index, player_name in enumerate(roster, start=1):
            agent_name = AGENTS[(player_index + map_index + len(map_name)) % len(AGENTS)]
            kills_base = 15.0 + team_strength * 5.0 + (3.0 if team_won_map else -1.0) + player_index * 0.8
            deaths_base = 14.0 + opponent_strength * 2.5 + (2.0 if not team_won_map else -0.5) - player_index * 0.4
            kills = max(0, int(round(kills_base + rng.uniform(-2.0, 2.5))))
            deaths = max(0, int(round(deaths_base + rng.uniform(-2.0, 2.0))))
            acs = round(180.0 + kills * 4.5 - deaths * 2.2 + (10.0 if team_won_map else -8.0) + rng.uniform(-12.0, 12.0), 1)
            rating = round(0.85 + (kills - deaths) / 40.0 + (0.08 if team_won_map else -0.03), 2)
            stats.append(
                VLRPlayerStatLine(
                    match_id=record.match_id,
                    map_id=f"{record.match_id}:map-{map_index}",
                    map_name=map_name,
                    team_name=team_name,
                    opponent_team=record.team_b if team_name == record.team_a else record.team_a,
                    player_name=player_name,
                    agent_name=agent_name,
                    rating=rating,
                    acs=acs,
                    kills=kills,
                    deaths=deaths,
                    assists=max(0, int(round(3 + rng.uniform(0, 5)))),
                )
            )
    return stats