from datetime import date

from app.models.schemas import MatchFixture
from app.services.pipeline import predict_fixtures


if __name__ == "__main__":
    fixtures = [
        MatchFixture(
            match_id="sample-1",
            region="Pacific",
            event_name="Masters",
            team_a="Team Alpha",
            team_b="Team Beta",
            match_date=date.today(),
            best_of=3,
        )
    ]
    print(predict_fixtures(fixtures).model_dump_json(indent=2))
