def test_rx():
    from pnq.rx import Distributer, Subscriber, MyExitStack

    result_next = []
    result_err = []

    sub1 = Subscriber()
    @sub1.callback_on_next
    def on_next(value):
        result_next.append(value)

    @sub1.callback_on_err
    def on_error(err):
        result_err.append(str(err))

    source1 = Distributer()
    source2 = Distributer()

    with MyExitStack(source1, source2) as stack:
        assert len(source1._subscribers) == 0
        assert len(source2._subscribers) == 0

        # merge や union
        source1.distribuite(sub1)
        source2.distribuite(sub1)

        assert len(source1._subscribers) == 1
        assert len(source2._subscribers) == 1

        source1.on_next(1)
        assert result_next == [1]
        assert result_err == []

        source2.on_next(2)
        assert result_next == [1, 2]
        assert result_err == []

        source1.on_error(Exception("err1"))
        assert result_next == [1, 2]
        assert result_err == ["err1"]

        source2.on_error(Exception("err2"))
        assert result_next == [1, 2]
        assert result_err == ["err1", "err2"]
