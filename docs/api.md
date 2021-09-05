# Developer Interface

## Helper Functions

---

!!! note
    Only use these functions if you're testing HTTPX in a console
    or making a small number of requests. Using a `Client` will
    enable HTTP/2 and connection pooling for more efficient and
    long-lived connections.

---

::: pnq.Query
    :docstring:
    :members: filter map unpack unpack_kw select select_attr cast