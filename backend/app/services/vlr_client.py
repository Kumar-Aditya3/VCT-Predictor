from __future__ import annotations

from datetime import UTC, date, datetime
import json
import re
from typing import Iterable

import requests
from bs4 import BeautifulSoup, Tag

from app.core.config import get_settings
from app.models.schemas import MatchFixture
from app.models.vlr import VLRMapRecord, VLRMatchDetails, VLRMatchRecord, VLRPlayerStatLine
from app.services.tier1_scope import is_tier1_event, is_tier1_region


BASE_URL = "https://www.vlr.gg"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
TIMEOUT_SECONDS = 20
DATE_LABEL_FORMAT = "%a, %B %d, %Y"


class VLRClient:
    def __init__(self) -> None:
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})

    def fetch_matches(self, from_date: date, to_date: date) -> list[VLRMatchRecord]:
        settings = get_settings()
        records = self._parse_paginated_results(
            from_date=from_date,
            to_date=to_date,
            max_pages=settings.training_results_pages,
        )
        self._write_raw_snapshot("results", records)
        return records

    def fetch_match_details(self, records: list[VLRMatchRecord]) -> list[VLRMatchDetails]:
        settings = get_settings()
        details: list[VLRMatchDetails] = []
        for record in records[: settings.detail_scrape_limit]:
            if not record.source_url:
                continue
            html = self.session.get(record.source_url, timeout=TIMEOUT_SECONDS)
            html.raise_for_status()
            parsed = _parse_match_detail_page(record, html.text)
            if parsed is not None:
                details.append(parsed)
        self._write_raw_snapshot("details", details)
        return details

    def fetch_upcoming_fixtures(self, from_date: date, to_date: date) -> list[MatchFixture]:
        settings = get_settings()
        fixtures = self._parse_paginated_schedule(
            from_date=from_date,
            to_date=to_date,
            max_pages=settings.upcoming_pages,
        )
        self._write_raw_snapshot("upcoming", fixtures)
        return fixtures

    def _parse_paginated_results(self, from_date: date, to_date: date, max_pages: int) -> list[VLRMatchRecord]:
        records: list[VLRMatchRecord] = []
        for page in range(1, max_pages + 1):
            html = self._get_page("/matches/results", page)
            soup = BeautifulSoup(html, "lxml")
            page_records, oldest_date = _parse_results_page(soup, from_date=from_date, to_date=to_date)
            records.extend(page_records)
            if oldest_date is not None and oldest_date < from_date:
                break
            if not page_records and oldest_date is None:
                break
        return records

    def _parse_paginated_schedule(self, from_date: date, to_date: date, max_pages: int) -> list[MatchFixture]:
        fixtures: list[MatchFixture] = []
        for page in range(1, max_pages + 1):
            html = self._get_page("/matches", page)
            soup = BeautifulSoup(html, "lxml")
            page_fixtures, newest_date = _parse_schedule_page(soup, from_date=from_date, to_date=to_date)
            fixtures.extend(page_fixtures)
            if newest_date is not None and newest_date > to_date:
                break
            if not page_fixtures and newest_date is None:
                break
        return fixtures

    def _get_page(self, path: str, page: int) -> str:
        url = f"{BASE_URL}{path}" if page == 1 else f"{BASE_URL}{path}/?page={page}"
        response = self.session.get(url, timeout=TIMEOUT_SECONDS)
        response.raise_for_status()
        return response.text

    def _write_raw_snapshot(self, kind: str, payload: Iterable[object]) -> None:
        settings = get_settings()
        settings.raw_data_dir.mkdir(parents=True, exist_ok=True)
        snapshot_path = settings.raw_data_dir / f"{kind}_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.json"
        serialized = [item.model_dump(mode="json") if hasattr(item, "model_dump") else item for item in payload]
        snapshot_path.write_text(json.dumps(serialized, indent=2), encoding="utf-8")


def _parse_results_page(soup: BeautifulSoup, from_date: date, to_date: date) -> tuple[list[VLRMatchRecord], date | None]:
    records: list[VLRMatchRecord] = []
    current_date: date | None = None
    oldest_date: date | None = None

    for node in soup.select("div.wf-label.mod-large, a.match-item"):
        classes = node.get("class", [])
        if "wf-label" in classes:
            current_date = _parse_date_label(node.get_text(" ", strip=True))
            if current_date is not None and (oldest_date is None or current_date < oldest_date):
                oldest_date = current_date
            continue
        if current_date is None or current_date > to_date:
            continue
        if current_date < from_date:
            break
        parsed = _parse_match_item(node, current_date, status="completed")
        if parsed is not None:
            records.append(parsed)
    return records, oldest_date


def _parse_schedule_page(soup: BeautifulSoup, from_date: date, to_date: date) -> tuple[list[MatchFixture], date | None]:
    fixtures: list[MatchFixture] = []
    current_date: date | None = None
    newest_date: date | None = None

    for node in soup.select("div.wf-label.mod-large, a.match-item"):
        classes = node.get("class", [])
        if "wf-label" in classes:
            current_date = _parse_date_label(node.get_text(" ", strip=True))
            if current_date is not None and (newest_date is None or current_date > newest_date):
                newest_date = current_date
            continue
        if current_date is None or current_date < from_date or current_date > to_date:
            continue
        parsed = _parse_match_item(node, current_date, status="upcoming")
        if parsed is not None:
            fixtures.append(_record_to_fixture(parsed))
    return fixtures, newest_date


def _parse_date_label(label_text: str) -> date | None:
    cleaned = re.sub(r"\s+(Today|Yesterday)$", "", label_text.strip())
    try:
        return datetime.strptime(cleaned, DATE_LABEL_FORMAT).date()
    except ValueError:
        return None


def _parse_match_item(node: Tag, match_date: date, status: str) -> VLRMatchRecord | None:
    team_names = [text.get_text(" ", strip=True) for text in node.select(".match-item-vs-team-name .text-of")]
    scores = [_parse_int(text.get_text(" ", strip=True)) for text in node.select(".match-item-vs-team-score")]
    event_title = node.select_one(".match-item-event")
    event_stage = node.select_one(".match-item-event-series")

    if len(team_names) < 2 or event_title is None:
        return None

    event_text = event_title.get_text(" ", strip=True)
    event_name, region = extract_tier1_metadata(event_text)
    if event_name is None or region is None:
        return None

    match_id = node.get("href", "").strip("/").split("/")[0]
    if not match_id:
        return None

    team_a_maps = scores[0] if len(scores) > 0 and scores[0] is not None else 0
    team_b_maps = scores[1] if len(scores) > 1 and scores[1] is not None else 0
    if status == "completed" and max(team_a_maps, team_b_maps) > 5:
        return None
    best_of = 5 if max(team_a_maps, team_b_maps) >= 3 or (team_a_maps + team_b_maps) >= 5 else 3

    return VLRMatchRecord(
        match_id=match_id,
        event_name=event_name,
        region=region,
        match_date=match_date,
        team_a=team_names[0],
        team_b=team_names[1],
        event_stage=event_stage.get_text(" ", strip=True) if event_stage else None,
        team_a_maps_won=team_a_maps,
        team_b_maps_won=team_b_maps,
        best_of=best_of,
        status=status,
        source_url=f"{BASE_URL}{node.get('href')}",
    )


def _parse_match_detail_page(record: VLRMatchRecord, html: str) -> VLRMatchDetails | None:
    soup = BeautifulSoup(html, "lxml")
    game_nodes = [node for node in soup.select("div.vm-stats-game") if node.get("data-game-id") != "all"]
    maps: list[VLRMapRecord] = []
    player_stats: list[VLRPlayerStatLine] = []

    for order_index, game_node in enumerate(game_nodes, start=1):
        game_id = game_node.get("data-game-id")
        if not game_id:
            continue
        header = game_node.select_one("div.vm-stats-game-header")
        table = game_node.select_one("table.wf-table-inset.mod-overview")
        if header is None or table is None:
            continue

        map_record = _parse_map_header(record, header, game_id, order_index)
        if map_record is None:
            continue
        maps.append(map_record)

        for row in table.select("tbody tr"):
            parsed_row = _parse_player_row(record, map_record, row)
            if parsed_row is not None:
                player_stats.append(parsed_row)

    if not maps:
        return None

    updated_match = record.model_copy(
        update={
            "team_a_maps_won": sum(1 for map_record in maps if map_record.winner_team == record.team_a),
            "team_b_maps_won": sum(1 for map_record in maps if map_record.winner_team == record.team_b),
            "best_of": max(record.best_of, len(maps)),
        }
    )
    return VLRMatchDetails(match=updated_match, maps=maps, player_stats=player_stats)


def _parse_map_header(record: VLRMatchRecord, header: Tag, game_id: str, order_index: int) -> VLRMapRecord | None:
    team_nodes = header.select("div.team")
    score_nodes = header.select("div.score")
    map_node = header.select_one("div.map")
    if len(team_nodes) < 2 or len(score_nodes) < 2 or map_node is None:
        return None

    team_a_name = _clean_text(team_nodes[0].select_one(".team-name").get_text(" ", strip=True))
    team_b_name = _clean_text(team_nodes[1].select_one(".team-name").get_text(" ", strip=True))
    map_name = _extract_map_name(map_node)
    duration_text = _clean_text(map_node.select_one(".map-duration").get_text(" ", strip=True)) if map_node.select_one(".map-duration") else None
    picked_node = map_node.select_one(".picked")
    picked_by = None
    if picked_node is not None:
        picked_by = team_a_name if "mod-1" in picked_node.get("class", []) else team_b_name if "mod-2" in picked_node.get("class", []) else None

    team_a_rounds = _parse_int(score_nodes[0].get_text(" ", strip=True)) or 0
    team_b_rounds = _parse_int(score_nodes[1].get_text(" ", strip=True)) or 0
    winner_team = team_a_name if team_a_rounds > team_b_rounds else team_b_name if team_b_rounds > team_a_rounds else None

    return VLRMapRecord(
        map_id=f"{record.match_id}:{game_id}",
        match_id=record.match_id,
        map_name=map_name,
        team_a=team_a_name,
        team_b=team_b_name,
        team_a_rounds=team_a_rounds,
        team_b_rounds=team_b_rounds,
        picked_by=picked_by,
        duration_text=duration_text,
        winner_team=winner_team,
        order_index=order_index,
    )


def _parse_player_row(record: VLRMatchRecord, map_record: VLRMapRecord, row: Tag) -> VLRPlayerStatLine | None:
    player_cell = row.select_one("td.mod-player")
    if player_cell is None:
        return None
    player_name_node = player_cell.select_one(".text-of")
    team_name_node = player_cell.select_one(".ge-text-light")
    if player_name_node is None or team_name_node is None:
        return None
    team_name = _resolve_team_name(_clean_text(team_name_node.get_text(" ", strip=True)), map_record.team_a, map_record.team_b)

    opponent_team = map_record.team_b if team_name == map_record.team_a else map_record.team_a
    player_name = _clean_text(player_name_node.get_text(" ", strip=True))
    agents = [img.get("title") or img.get("alt") for img in row.select(".mod-agents img")]
    stats = [_extract_both_value(cell) for cell in row.select("td.mod-stat")]
    rating = _parse_float(stats[0]) if len(stats) > 0 else None
    acs = _parse_float(stats[1]) if len(stats) > 1 else None
    kills = _parse_int(stats[2]) or 0 if len(stats) > 2 else 0
    deaths = _parse_int(stats[3]) or 0 if len(stats) > 3 else 0
    assists = _parse_int(stats[4]) or 0 if len(stats) > 4 else 0

    return VLRPlayerStatLine(
        match_id=record.match_id,
        map_id=map_record.map_id,
        map_name=map_record.map_name,
        team_name=team_name,
        opponent_team=opponent_team,
        player_name=player_name,
        agent_name=agents[0] if agents else None,
        rating=rating,
        acs=acs,
        kills=kills,
        deaths=deaths,
        assists=assists,
    )


def _resolve_team_name(raw_team_name: str, team_a: str, team_b: str) -> str:
    if raw_team_name in {team_a, team_b}:
        return raw_team_name
    candidates = [team_a, team_b]
    for candidate in candidates:
        if raw_team_name.lower() in candidate.lower() or candidate.lower() in raw_team_name.lower():
            return candidate
    abbreviation = "".join(part[0] for part in raw_team_name.split() if part)
    for candidate in candidates:
        candidate_abbreviation = "".join(part[0] for part in candidate.split() if part)
        if abbreviation and abbreviation.lower() == candidate_abbreviation.lower():
            return candidate
        if raw_team_name.lower() == candidate_abbreviation.lower():
            return candidate
    return team_a if raw_team_name.lower()[:1] == team_a.lower()[:1] else team_b


def _extract_both_value(cell: Tag) -> str:
    both = cell.select_one(".mod-both, .side.mod-side.mod-both, .side.mod-both")
    if both is not None:
        return _clean_text(both.get_text(" ", strip=True))
    return _clean_text(cell.get_text(" ", strip=True))


def _extract_map_name(map_node: Tag) -> str:
    span = map_node.select_one("div[style*='font-weight: 700'] span")
    if span is None:
        return _clean_text(map_node.get_text(" ", strip=True))
    text = span.get_text(" ", strip=True)
    return _clean_text(text.replace("PICK", "").strip())


def _parse_int(value: str | None) -> int | None:
    if value is None:
        return None
    digits = re.sub(r"[^\d-]", "", value)
    return int(digits) if digits else None


def _parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    match = re.search(r"-?\d+(?:\.\d+)?", value)
    return float(match.group(0)) if match else None


def _clean_text(text: str) -> str:
    return " ".join(text.split())


def extract_tier1_metadata(event_text: str) -> tuple[str | None, str | None]:
    normalized = " ".join(event_text.split())
    region = None
    for candidate in ("Americas", "EMEA", "Pacific", "China"):
        if candidate.lower() in normalized.lower():
            region = candidate
            break

    if "Masters" in normalized:
        event_name = "Masters"
    elif "Champions" in normalized:
        event_name = "Champions"
    elif "Kickoff" in normalized:
        event_name = "Kickoff"
    elif "Stage 2" in normalized or "Split 2" in normalized:
        event_name = "Split 2"
    elif "Stage 1" in normalized or "Split 1" in normalized:
        event_name = "Split 1"
    else:
        event_name = None

    if event_name is None:
        return None, None

    if region is None and event_name in {"Masters", "Champions"}:
        settings = get_settings()
        if settings.enable_international_events and "International" in settings.tier1_regions:
            region = "International"

    if region is None:
        return None, None
    return event_name, region


def _record_to_fixture(record: VLRMatchRecord) -> MatchFixture:
    return MatchFixture(
        match_id=record.match_id,
        region=record.region,
        event_name=record.event_name,
        event_stage=record.event_stage,
        team_a=record.team_a,
        team_b=record.team_b,
        match_date=record.match_date,
        best_of=max(3, record.best_of),
    )


def filter_tier1_records(records: list[VLRMatchRecord]) -> list[VLRMatchRecord]:
    return [
        record
        for record in records
        if is_tier1_event(record.event_name) and is_tier1_region(record.region)
    ]


def filter_tier1_fixtures(fixtures: list[MatchFixture]) -> list[MatchFixture]:
    return [
        fixture
        for fixture in fixtures
        if is_tier1_event(fixture.event_name) and is_tier1_region(fixture.region)
    ]
