from jinja2 import Environment, FileSystemLoader


class Info:
    def __init__(self, name, *typevars: str, inherit: str = "", is_pair: bool = False):
        self.SELF__ = name
        self.typevars = typevars
        self.inherit = inherit
        self.is_pair = is_pair

    def get_typevars(self):
        typevars = ",".join(self.typevars)
        if typevars:
            typevars = f"[{typevars}]"

        return typevars

    @property
    def CLS(self):
        typevars = self.get_typevars()
        mro = []
        if typevars:
            generic = "Generic" + typevars
            mro.append(generic)

        if self.inherit:
            mro.append(self.inherit)

        return self.SELF__ + f"({','.join(mro)})"

    @property
    def SELF_T(self):
        typevars = self.get_typevars()
        return self.SELF__ + typevars

    @property
    def T(self):
        if len(self.typevars) == 1:
            return self.typevars[0]
        elif len(self.typevars) == 2:
            return "Tuple[" + ",".join(self.typevars) + "]"
        else:
            return "##### exception #####"

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
    def selector(self):
        if len(self.typevars) == 2:
            return f"lambda x: x[1] "
        else:
            return "lambda x: x"


"""
templateで利用するプロパティは次のような文字が返る
query.CLS = "PairQuery(Generic[K, V])"
query.SELF__ = "PairQuery"
query.SELF_T = "PairQuery[K, V]"
query.T = "Tuple[K, V]"
query.K = "K"
query.V = "K, V"
query.selector = "lambda k, v: (k, v)"
"""


Query = Info("Query", "T")
PairQuery = Info("PairQuery", "K", "V", inherit="Query[Tuple[K, V]]", is_pair=True)


data = {
    "sequence": Query,
    "pair": PairQuery,
    "queries": [Query, PairQuery],
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
