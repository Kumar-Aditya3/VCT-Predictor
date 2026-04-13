from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from app.models.vlr import VLRMatchDetails, VLRPlayerStatLine


def validate_match_details(details_list: list[VLRMatchDetails]) -> dict[str, Any]:
    clean_details: list[VLRMatchDetails] = []
    scrape_issues: list[dict[str, Any]] = []
    training_exclusions: list[dict[str, Any]] = []
    exclusion_counts = {
        "excluded_matches": 0,
        "excluded_maps": 0,
        "excluded_player_rows": 0,
    }

    for details in details_list:
        clean_maps = []
        clean_player_rows: list[VLRPlayerStatLine] = []
        rows_by_map: dict[str, list[VLRPlayerStatLine]] = defaultdict(list)
        for row in details.player_stats:
            rows_by_map[row.map_id].append(row)

        match_excluded = False
        for map_record in details.maps:
            map_rows = rows_by_map.get(map_record.map_id, [])
            map_result = _validate_map_rows(details.match.match_id, map_record.map_id, map_record.team_a, map_record.team_b, map_rows)
            scrape_issues.extend(map_result["issues"])
            training_exclusions.extend(map_result["exclusions"])

            if map_result["drop_map"]:
                exclusion_counts["excluded_maps"] += 1
                exclusion_counts["excluded_player_rows"] += len(map_rows)
                continue

            clean_maps.append(map_record)
            clean_player_rows.extend(map_result["rows"])

        if not clean_maps:
            match_excluded = True
            exclusion_counts["excluded_matches"] += 1
            training_exclusions.append(
                {
                    "scope": "match",
                    "reference_id": details.match.match_id,
                    "reason": "no_valid_maps_after_integrity_filter",
                    "details": {},
                }
            )

        if not match_excluded:
            clean_match = details.match.model_copy(
                update={
                    "team_a_maps_won": sum(1 for row in clean_maps if row.winner_team == details.match.team_a),
                    "team_b_maps_won": sum(1 for row in clean_maps if row.winner_team == details.match.team_b),
                    "best_of": max(details.match.best_of, len(clean_maps)),
                }
            )
            clean_details.append(
                VLRMatchDetails(
                    match=clean_match,
                    maps=clean_maps,
                    player_stats=clean_player_rows,
                )
            )

    return {
        "details": clean_details,
        "scrape_issues": scrape_issues,
        "training_exclusions": training_exclusions,
        "exclusion_counts": exclusion_counts,
    }


def _validate_map_rows(
    match_id: str,
    map_id: str,
    team_a: str,
    team_b: str,
    rows: list[VLRPlayerStatLine],
) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    clean_rows: list[VLRPlayerStatLine] = []
    team_names = {team_a, team_b}

    if team_a == team_b:
        issues.append(_issue("map", map_id, "map_same_team_names", {"team_a": team_a, "team_b": team_b}))
        exclusions.append(_exclusion("map", map_id, "map_same_team_names"))
        return {"rows": [], "issues": issues, "exclusions": exclusions, "drop_map": True}

    player_to_teams: dict[str, set[str]] = defaultdict(set)
    raw_team_counter: Counter[str] = Counter()
    invalid_rows = 0

    for row in rows:
        raw_team_counter[row.team_name] += 1
        if row.team_name not in team_names or row.opponent_team not in team_names or row.team_name == row.opponent_team:
            invalid_rows += 1
            issues.append(
                _issue(
                    "player_row",
                    f"{map_id}:{row.player_name}",
                    "player_team_mismatch",
                    {"team_name": row.team_name, "opponent_team": row.opponent_team, "expected_teams": [team_a, team_b]},
                )
            )
            exclusions.append(_exclusion("player_row", f"{map_id}:{row.player_name}", "player_team_mismatch"))
            continue
        player_to_teams[row.player_name].add(row.team_name)
        clean_rows.append(row)

    duplicated = [player_name for player_name, teams in player_to_teams.items() if len(teams) > 1]
    if duplicated:
        issues.append(_issue("map", map_id, "player_on_both_sides", {"players": duplicated}))
        exclusions.append(_exclusion("player_rows", map_id, "player_on_both_sides"))
        return {"rows": [], "issues": issues, "exclusions": exclusions, "drop_map": False}

    clean_counter = Counter(row.team_name for row in clean_rows)
    if len(clean_counter) != 2:
        issues.append(_issue("map", map_id, "map_missing_both_teams", {"team_counts": dict(clean_counter)}))
        exclusions.append(_exclusion("player_rows", map_id, "map_missing_both_teams"))
        return {"rows": [], "issues": issues, "exclusions": exclusions, "drop_map": False}

    for team_name, count in clean_counter.items():
        if count < 4:
            issues.append(_issue("map", map_id, "team_row_count_too_small", {"team_name": team_name, "count": count}))
            exclusions.append(_exclusion("player_rows", map_id, "team_row_count_too_small"))
            return {"rows": [], "issues": issues, "exclusions": exclusions, "drop_map": False}
        if count > 5:
            issues.append(_issue("map", map_id, "team_row_count_capped", {"team_name": team_name, "count": count}))

    if any(count > 5 for count in clean_counter.values()):
        capped_rows: list[VLRPlayerStatLine] = []
        seen_per_team: dict[str, int] = Counter()
        seen_player_names: set[tuple[str, str]] = set()
        for row in clean_rows:
            key = (row.team_name, row.player_name)
            if key in seen_player_names or seen_per_team[row.team_name] >= 5:
                exclusions.append(_exclusion("player_row", f"{map_id}:{row.player_name}", "team_row_count_capped"))
                continue
            seen_player_names.add(key)
            seen_per_team[row.team_name] += 1
            capped_rows.append(row)
        clean_rows = capped_rows

    if invalid_rows:
        issues.append(_issue("map", map_id, "partial_player_row_exclusion", {"invalid_rows": invalid_rows, "raw_team_counts": dict(raw_team_counter)}))

    return {"rows": clean_rows, "issues": issues, "exclusions": exclusions, "drop_map": False}


def _issue(scope: str, reference_id: str, issue_type: str, details: dict[str, Any]) -> dict[str, Any]:
    return {
        "scope": scope,
        "reference_id": reference_id,
        "issue_type": issue_type,
        "details": details,
    }


def _exclusion(scope: str, reference_id: str, reason: str) -> dict[str, Any]:
    return {
        "scope": scope,
        "reference_id": reference_id,
        "reason": reason,
        "details": {},
    }
