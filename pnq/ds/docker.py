import requests_unixsocket

from pnq import query

# https://docs.docker.com/registry/spec/api/


# IF ERROR
"""
{
    "errors:" [{
            "code": <error identifier>,
            "message": <message describing condition>,
            "detail": <unstructured>
        },
        ...
    ]
}
"""


class DockerClient:
    def __init__(
        self, schema="http+unix://%2Fvar%2Frun%2Fdocker.sock", version="v1.40"
    ):
        self.schema = schema
        self.version = version

    def _make_url(self, endpoint):
        schema = self.schema
        version = f"/{self.version}"
        return schema + version + endpoint

    def commands(self):
        return query(
            [
                "images",
                "containers",
                "networks",
                "volumes",
                "nodes",
                "services",
                "tasks",
                "secrets",
                "configs",
            ]
        )

    def images(self):
        url = self._make_url("/images/json")
        return query(requests_unixsocket.get(url).json())

    def containers(self):
        url = self._make_url("/containers/json")
        return query(requests_unixsocket.get(url).json())

    def networks(self):
        url = self._make_url("/networks/json")
        return query(requests_unixsocket.get(url).json())

    def volumes(self):
        url = self._make_url("/volumes/json")
        return query(requests_unixsocket.get(url).json())

    # swarm
    def nodes(self):
        url = self._make_url("/nodes/json")
        return query(requests_unixsocket.get(url).json())

    def services(self):
        url = self._make_url("/services/json")
        return query(requests_unixsocket.get(url).json())

    def tasks(self):
        url = self._make_url("/tasks/json")
        return query(requests_unixsocket.get(url).json())

    def secrets(self):
        url = self._make_url("/secrets/json")
        return query(requests_unixsocket.get(url).json())

    def configs(self):
        url = self._make_url("/configs/json")
        return query(requests_unixsocket.get(url).json())

    def plugins(self):
        url = self._make_url("/plugins/json")
        return query(requests_unixsocket.get(url).json())


Client = DockerClient
