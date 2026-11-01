from irsys.cli import build_parser


def test_parses_search_with_k():
    args = build_parser().parse_args(["search", "vitamin d", "-k", "3"])
    assert (args.command, args.query, args.k) == ("search", "vitamin d", 3)


def test_search_defaults_to_five_results():
    assert build_parser().parse_args(["search", "q"]).k == 5


def test_parses_pipeline_commands():
    assert build_parser().parse_args(["evaluate"]).command == "evaluate"
    assert build_parser().parse_args(["rescore"]).command == "rescore"
    assert build_parser().parse_args(["analyze"]).command == "analyze"


def test_parses_demo_command():
    assert build_parser().parse_args(["demo"]).command == "demo"


def test_demo_includes_the_llm_unless_switched_off():
    assert build_parser().parse_args(["demo"]).llm is True
    assert build_parser().parse_args(["demo", "--no-llm"]).llm is False
