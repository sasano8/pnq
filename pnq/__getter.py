import operator

undefined = object()
ignore = object()


class Getter2:
    def __init__(self, name):
        self.name = name

    @staticmethod
    def _iterate(name, *args):
        try:
            yield args[0]
            yield name
            yield from args[1:]
        except:
            pass

    def __call__(self, *args):
        return getattr(*self._iterate(self.name, *args))

    def eq(self, value, default=undefined):
        if default is undefined:
            args = []
        else:
            args = [default]

        return lambda target: self(target, *args) == value


class BinaryOperator:
    def __init__(self, getter, op, other):
        self.getter = getter
        self.op = op
        self.other = other

        if op == "==":
            self.func = lambda target: getter(target) == other
        elif op == "!=":
            self.func = lambda target: getter(target) != other
        elif op == "and":
            self.func = lambda target: getter(target) and other
        elif op == "or":
            self.func = lambda target: getter(target) or other

    def __call__(self, *args):
        # return self.op(*args)
        return self.func(*args)

    def __str__(self):
        return f"{self.getter} {self.op} {self.other}"

    @staticmethod
    def _check_type(other):
        if not isinstance(other, BinaryOperator):
            raise TypeError(
                "BinaryOperator can't be chained with another BinaryOperator"
            )

    def __and__(self, other):
        self._check_type(other)
        print(22)
        if isinstance(other, IgnoreOperator):
            return self
        else:
            # return BinaryOperator(lambda target: self(target) and other(target))
            return BinaryOperator(self, "and", other)

    def __or__(self, other):
        self._check_type(other)
        print(11)
        if isinstance(other, IgnoreOperator):
            return self
        else:
            # return BinaryOperator(lambda target: self(target) or other(target))
            return BinaryOperator(self, "or", other)


class IgnoreOperator(BinaryOperator):
    def __init__(self):
        self.op = None

    def __str__(self):
        return ""

    def __call__(self, *args):
        raise RuntimeError("IgnoreOperator can't be called.")

    def __and__(self, other):
        self._check_type(other)
        return other

    def __or__(self, other):
        self._check_type(other)
        return other


class Getter:
    def __init__(self, name):
        print(name)
        self.name = name
        self.op = operator.attrgetter(name)

    def __str__(self):
        return f"x.{self.name}"

    def __call__(self, *args):
        return self.op(*args)

    @classmethod
    def _check_type(cls, other):
        if isinstance(other, cls):
            raise TypeError("Getter can't be chained with another Getter")

    def __eq__(self, other):
        self._check_type(other)
        if other is ignore:
            return IgnoreOperator()
        else:
            # return BinaryOperator(lambda target: self(target) == other)
            return BinaryOperator(self, "==", other)

    def __ne__(self, other):
        self._check_type(other)
        if other is ignore:
            return IgnoreOperator()
        else:
            # return BinaryOperator(lambda target: self(target) != other)
            return BinaryOperator(self, "!=", other)


class Operator:
    def __getattr__(self, name):
        return Getter(name)


op = Operator()


class Obj:
    def __init__(self, id, name):
        self.id = id
        self.name = name


obj = Obj(1, "bob")


comp1 = op.id == 1
comp2 = op.name == ignore
print(str(comp1 or comp2))
print(str(comp2 or comp1))
