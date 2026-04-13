from app.core.config import EventName, Region, get_settings


def is_tier1_region(region: str) -> bool:
    settings = get_settings()
    return region in settings.tier1_regions


def is_tier1_event(event_name: str) -> bool:
    settings = get_settings()
    return event_name in settings.tier1_events


def assert_tier1_scope(region: str, event_name: str) -> None:
    if not is_tier1_region(region):
        raise ValueError(
            f"Region '{region}' is out of scope. Allowed regions: {', '.join(get_settings().tier1_regions)}"
        )
    if not is_tier1_event(event_name):
        raise ValueError(
            f"Event '{event_name}' is out of scope. Allowed events: {', '.join(get_settings().tier1_events)}"
        )


def whitelisted_scope() -> dict[str, tuple[Region | EventName, ...]]:
    settings = get_settings()
    return {
        "regions": settings.tier1_regions,
        "events": settings.tier1_events,
    }
