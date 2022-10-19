import requests
import pnq
import urllib.parse


class K8sClient:
    def __init__(self, url):
        """minikube: kubectl proxy"""
        self.url = url

    def paths(self):
        res = requests.get(self.url)
        res.raise_for_status()
        return pnq.query(res.json()["paths"])

    def resources(self):
        def scan():
            for path in self.paths():
                res = self.get(path).json()
                if "resources" in res:
                    yield path

        return pnq.query(list(scan()))

    def apiv1(self):
        url = urllib.parse.urljoin(self.url, "/api/v1")
        res = requests.get(url)
        res.raise_for_status()
        return pnq.query("/api/v1/" + x["name"] for x in res.json()["resources"])

    def get(self, path, key=None):
        url = urllib.parse.urljoin(self.url, path)
        res = requests.get(url)
        res.raise_for_status()
        if key is None:
            return res.json()
        else:
            return res.json()[key]

    def query(self, path, key=None):
        data = self.get(path, key)
        return pnq.query(data)


Client = K8sClient


"""
from pnq.ds.k8s import Client
c = Client("http://localhost:8001")
c.query("/api/v1/pods", "items").select("metadata").select("name").each(print)
"""
