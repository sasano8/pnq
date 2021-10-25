# type: ignore
import pnq


class Obj:
    def __init__(self, **kwargs) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)


class TestItem:
    @classmethod
    def setup_class(cls):
        cls.new = dict
        cls._ = pnq.operators.item

    def test_init(self):
        self.new = dict
        self._ = pnq.operators.item

    def test_eq(self):
        func = self._.count == 1
        assert not func(self.new(count=0))
        assert func(self.new(count=1))

    def test_ne(self):
        func = self._.count != 1
        assert func(self.new(count=0))
        assert not func(self.new(count=1))


class TestAttr:
    @classmethod
    def setup_class(cls):
        cls.new = Obj
        cls._ = pnq.operators.attr

    def test_init(self):
        self.new = Obj
        self._ = pnq.operators.attr

    def test_eq(self):
        func = self._.count == 1
        assert not func(self.new(count=0))
        assert func(self.new(count=1))

    def test_ne(self):
        func = self._.count != 1
        assert func(self.new(count=0))
        assert not func(self.new(count=1))
