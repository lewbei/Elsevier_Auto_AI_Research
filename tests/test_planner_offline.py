from agents_planner import make_plan_offline


def test_make_plan_offline_shape():
    novelty = {"themes": [{"name": "Contrastive augmentation"}]}
    plan = make_plan_offline(novelty)
    assert isinstance(plan, dict)
    for k in ["objective", "hypotheses", "success_criteria", "datasets", "baselines", "novelty_focus", "stopping_rules"]:
        assert k in plan
    assert plan["datasets"][0]["path"].startswith("data/")

