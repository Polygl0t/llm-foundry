"""
Parser for preprocess.py that adds a UUID to every sample.

Optional config:
{
  "uuid_column": "uuid"
}
"""
import uuid

PARSER_NAME = "add-uuid"


def setup(args):
    if not isinstance(args.parser_config, dict):
        raise ValueError("parser_config must be a JSON object.")


def resolve_subsets(args):
    return [args.default_subset_name]


def parse_document(doc, args):
    uuid_column = args.parser_config.get("uuid_column", "uuid")
    return {
        "subset": args.default_subset_name,
        "metadata": {uuid_column: str(uuid.uuid4())},
    }
