from app.services.pipeline import run_weekly_update


if __name__ == "__main__":
    result = run_weekly_update()
    print(
        "weekly_update "
        f"status={result.status} "
        f"run_at={result.run_at} "
        f"prediction_mode={result.prediction_mode} "
        f"model_version={result.model_version} "
        f"compared_matches={result.compared_matches} "
        f"winner_accuracy={result.winner_accuracy:.4f} "
        f"artifact_path={result.artifact_path}"
    )
