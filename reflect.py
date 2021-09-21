from collections import defaultdict


def transpose(mapping):

    tmp = defaultdict(list)

    for left, right in mapping.items():
        if isinstance(right, str):
            tmp[left].append(right)
        elif isinstance(right, list):
            tmp[left] = right
        elif isinstance(right, tuple):
            tmp[left] = right
        elif isinstance(right, set):
            tmp[left] = right
        else:
            raise TypeError(f"{v} is not a valid mapping")

    # output属性 - 元の属性（複数の場合あり）
    target = defaultdict(list)

    for k, outputs in tmp.items():
        for out in outputs:
            target[out].append(k)

    return target


def split_single_multi(dic):
    single = {}
    multi = {}
    for k, v in dic.items():
        if len(v) > 1:
            multi[k] = v
        else:
            single[k] = v[0]

    return single, multi


def build_selector(single, multi, attr: bool = False):
    template = {}
    for k in multi.keys():
        template[k] = []

    if attr:

        def reflector(x):
            result = {}
            for k, v in single.items():
                result[k] = x[v]

            for k, fields in multi.items():
                result[k] = []
                for f in fields:
                    result[k].append(x[f])

            return result

    else:

        def reflector(x):
            result = {}
            for k, v in single.items():
                result[k] = x[v]

            for k, fields in multi.items():
                result[k] = []
                for f in fields:
                    result[k].append(x[f])

            return result

    return reflector


transposed = transpose(
    {
        "id": ["id"],
        "name": {"name", "searchable"},
        "kana": {"kana", "searchable"},
        "note": ["searchable"],
    }
)

single, multi = split_single_multi(transposed)

assert single == {
    "id": "id",
    "name": "name",
    "kana": "kana",
}

assert multi == {
    "searchable": ["name", "kana", "note"],
}

selector = build_selector(single, multi)
print(selector)

print(selector({"id": 1, "name": "山田", "kana": "yamada", "note": "hogehoge~~~"}))
