import operator

try:
    from typing import Literal
except:
    from typing_extensions import Literal

# TH - Type Hint

TH_ASSIGN_OP = Literal[
    "+=",
    "-=",
    "*=",
    "/=",
    "//=",
    "%=",
    "**=",
    "<<=",
    ">>=",
    "&=",
    "^=",
    "|=",
]

MAP_ASSIGN_OP = {
    "+=": operator.iadd,
    "-=": operator.isub,
    "*=": operator.imul,
    "/=": operator.itruediv,
    "//=": operator.ifloordiv,
    "%=": operator.imod,
    "**=": operator.ipow,
    "<<=": operator.ilshift,
    ">>=": operator.irshift,
    "&=": operator.iand,
    "^=": operator.ixor,
    "|=": operator.ior,
}

TH_ROUND = Literal[
    "ROUND_DOWN",
    "ROUND_HALF_UP",
    "ROUND_HALF_EVEN",
    "ROUND_CEILING",
    "ROUND_FLOOR",
    "ROUND_UP",
    "ROUND_HALF_DOWN",
    "ROUND_05UP",
]


# TODO: 整理する

operators = Literal[
    # 比較系
    "__lt__",
    "__le__",
    "__eq__",
    "__ne__",
    "__gt__",
    "__ge__",
    # 算術系（左辺）
    "__add__",
    "__sub__",
    "__mul__",
    "__truediv__",
    "__floordiv__",
    "__mod__",
    "__pow__",
    "__lshift__",
    "__rshift__",
    "__and__",
    "__or__",
    "__xor__",
    # 算術系（右辺）
    "__radd__",
    "__rsub__",
    "__rmul__",
    "__rtruediv__",
    "__rfloordiv__",
    "__rmod__",
    "__rpow__",
    "__rlshift__",
    "__rrshift__",
    "__rand__",
    "__rxor__",
    "__ror__",
    # 算術代入演算子
    "__iadd__",
    "__isub__",
    "__imul__",
    "__itruediv__",
    "__ifloordiv__",
    "__imod__",
    "__ipow__",
    "__ilshift__",
    "__irshift__",
    "__iand__",
    "__ixor__",
    "__ior__",
]

"__matmul__"
"__pos__"
"__neg__"
"__invert__"  # ~obj

op_mappings = {
    # 比較系
    "<": "__lt__",
    "<=": "__le__",
    "==": "__eq__",
    "!=": "__ne__",
    ">": "__gt__",
    ">=": "__ge__",
    # 算術系（左辺）
    "+": "__add__",
    "-": "__sub__",
    "*": "__mul__",
    "/": "__truediv__",
    "//": "__floordiv__",
    "%": "__mod__",
    "**": "__pow__",
    "<<": "__lshift__",
    ">>": "__rshift__",
    "&": "__and__",
    "^": "__or__",
    "|": "__xor__",
    # 算術系（右辺）
    # "+": "__radd__",
    # "-": "__rsub__",
    # "*": "__rmul__",
    # "/": "__rtruediv__",
    # "//": "__rfloordiv__",
    # "%": "__rmod__",
    # "**": "__rpow__",
    # "<<": "__rlshift__",
    # ">>": "__rrshift__",
    # "&": "__rand__",
    # "^": "__rxor__",
    # "|": "__ror__",
    # 算術代入演算子
    "+=": "__iadd__",
    "-=": "__isub__",
    "*=": "__imul__",
    "/=": "__itruediv__",
    "//=": "__ifloordiv__",
    "%=": "__imod__",
    "**=": "__ipow__",
    "<<=": "__ilshift__",
    ">>=": "__irshift__",
    "&=": "__iand__",
    "^=": "__ixor__",
    "|=": "__ior__",
}
