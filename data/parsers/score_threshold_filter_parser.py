"""
Parser for preprocess.py that drops samples below a score threshold.

Required config:
{
  "score_column": "edu_int_score",
  "minimum_score": 3
}
"""
PARSER_NAME = "score-threshold-filter"


def setup(args):
    if not isinstance(args.parser_config, dict):
        raise ValueError("parser_config must be a JSON object.")
    if "score_column" not in args.parser_config:
        raise ValueError("parser_config must include 'score_column'.")
    if "minimum_score" not in args.parser_config:
        raise ValueError("parser_config must include 'minimum_score'.")


def resolve_subsets(args):
    return [args.default_subset_name]


def parse_document(doc, args):
    score_column = args.parser_config["score_column"]
    minimum_score = args.parser_config["minimum_score"]
    score = doc.metadata.get(score_column)

    if score is None or score < minimum_score:
        return None

    return args.default_subset_name
