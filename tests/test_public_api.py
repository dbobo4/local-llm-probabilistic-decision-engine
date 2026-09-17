import llm_decision_engine


def test_v0_1_0_public_api_surface():
    assert llm_decision_engine.__all__ == [
        "DecisionEngine",
        "Choice",
        "Boolean",
        "Rating",
        "ChoiceResult",
        "BooleanResult",
        "RatingResult",
        "DecisionResult",
    ]


def test_evaluation_helpers_are_not_top_level_public_api():
    evaluation_names = {
        "BinaryEvaluationResult",
        "CalibrationBin",
        "BinaryCalibrationResult",
        "binary_brier_score",
        "negative_log_likelihood",
        "evaluate_binary_predictions",
        "evaluate_binary_calibration",
    }

    assert evaluation_names.isdisjoint(llm_decision_engine.__all__)
