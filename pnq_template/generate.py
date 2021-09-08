from jinja2 import Environment, FileSystemLoader


class Info:
    def __init__(self, name, *typevars: str, is_index: bool = False):
        self.name = name
        self.typevars = typevars
        self.is_index = is_index

    def get_typevars(self):
        typevars = ",".join(self.typevars)
        if typevars:
            typevars = f"[{typevars}]"

        return typevars

    @property
    def cls(self):
        typevars = self.get_typevars()
        if typevars:
            typevars = "(Generic" + typevars + ")"
        return self.name + typevars

    @property
    def str(self):
        typevars = self.get_typevars()
        return self.name + typevars

    @property
    def row(self):
        if len(self.typevars) == 1:
            return self.typevars[0]
        elif len(self.typevars) == 2:
            return "Tuple[" + ",".join(self.typevars) + "]"
        else:
            return "##### exception #####"

    @property
    def to_dict(self):
        if len(self.typevars) == 1:
            return "DictEx[Any,Any]"
        elif len(self.typevars) == 2:
            return f"DictEx[{self.typevars[0]},{self.typevars[1]}]"
        else:
            return "DictEx[Any,Any]"

    @property
    def to_list(self):
        if len(self.typevars) == 1:
            return f"ListEx[{self.typevars[0]}]"
        elif len(self.typevars) == 2:
            return f"ListEx[Tuple[{self.typevars[0]},{self.typevars[1]}]]"
        else:
            return "ListEx"

    @property
    def K(self):
        if len(self.typevars) == 2:
            return self.typevars[0]
        else:
            return "##### exception #####"

    @property
    def V(self):
        if len(self.typevars) == 2:
            return self.typevars[1]
        else:
            return "##### exception #####"

    @property
    def value(self):
        if len(self.typevars) == 2:
            return self.typevars[1]
        else:
            return self.typevars[0]

    @property
    def selector(self):
        if len(self.typevars) == 2:
            return f"lambda x: x[0] "
        else:
            return "lambda x: x"


Query = Info("Query", "T")
PairQuery = Info("PairQuery", "K", "V")
IndexQuery = Info("IndexQuery", "K", "V", is_index=True)


data = {
    "str": str,
    "sequence": Query,
    "pair": PairQuery,
    "IndexQuery": IndexQuery,
    "queries": [Query, PairQuery, IndexQuery],
}


def generate(input, output):
    env = Environment(loader=FileSystemLoader("./", encoding="utf8"))
    tpl = env.get_template(input)
    s = tpl.render(data)

    with open(output, "w") as f:
        f.write(s)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="雛形からソースコードを自動生成します。")
    parser.add_argument("-i", help="入力とするテンプレートファイルを指定します")
    parser.add_argument("-o", help="出力するファイル名を指定します")

    args = parser.parse_args()

    generate(args.i, args.o)
    print("generated file:" + args.o)
