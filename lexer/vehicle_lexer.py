from pygments.lexer import RegexLexer, words
from pygments.token import Keyword, Name, Comment, Literal, Operator


class VehicleLexer(RegexLexer):
    name = "Vehicle"
    aliases = ["vehicle"]
    filenames = ["*.vcl"]

    tokens = {
        "root": [
            # Declaration_keywords
            (words(("network", "dataset", "type"), suffix=r"\b"), Keyword.Declaration),
            (words(("implicit", "parameter"), suffix=r"\b"), Keyword.Declaration),
            (r"\b(implicit\s+)?parameter\b", Keyword.Declaration),
            # Control_keywords
            (
                words(
                    ("forall", "exists", "foreach", "let", "in", "if", "then", "else"),
                    suffix=r"\b",
                ),
                Keyword.Control,
            ),
            # Value_builtins
            (
                words(("()", "True", "False", "not", "map", "fold"), suffix=r"\b"),
                Keyword.Constant,
            ),
            # Type_builtins
            (
                r"\b(forallT|Unit|Bool|Nat|Int|Rat|Vector|Tensor|List|Index|Type(\s+[0-9]+))\b",
                Keyword.Declaration,
            ),
            # Operator_builtins
            (
                r"\b(->|\.|:|\\|=>|==|!=|<=|<|>=|>|\\+|/|\\*|-|::|!|=|and|or)\b",
                Operator,
            ),
            # Comments
            (r"--.*$", Comment.Single),
            (r"\{-", Comment.Multiline, "block_comment"),
            # Numeric Literals
            (r"\b[0-9]+\.[0-9]+\b", Literal.Number),
            (r"\b[0-9]+\b", Literal.Number),
            # Variables (Holes)
            (r"(\?[A-Za-z0-9_]*)\b", Name.Variable),
        ],
        "block_comment": [
            (r".*?\-}", Comment.Multiline, "#pop"),
            (r".+", Comment.Multiline),
        ],
    }
