from pnq.exceptions import NotFoundError


def test_notfounderror():
    """例外をいくつか継承した時、継承した任意の例外をキャッチできるか確認する"""
    msg = "not found"
    expect = "'not found'"
    try:
        raise NotFoundError(msg)
    except Exception as e:
        assert str(e) == expect

    try:
        raise NotFoundError(msg)
    except KeyError as e:
        assert str(e) == expect

    try:
        raise NotFoundError(msg)
    except IndexError as e:
        assert str(e) == expect

    try:
        raise NotFoundError(msg)
    except NotFoundError as e:
        assert str(e) == expect

    try:
        raise NotFoundError(msg)
    except (KeyError, IndexError, NotFoundError) as e:
        assert str(e) == expect

    try:
        raise NotFoundError(msg)
    except KeyError as e:
        assert str(e) == expect

    except IndexError:
        raise Exception()

    except NotFoundError:
        raise Exception()
