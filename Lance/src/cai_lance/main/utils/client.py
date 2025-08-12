import json
from typing import Optional, Dict, Union, Any, List

import grpc
import requests
import websocket

class HttpClient:
    def __init__(
            self,
            base_url: str,
            auth: Optional[Dict[str, str]] = None,
            api_key: Optional[str] = None,
            basic: Optional[tuple] = None,
            cert: Optional[tuple] = None,
            verify: Union[bool, str] = True,
    ):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        if auth and "bearer" in auth:
            self.session.headers.update({"Authorization": f"Bearer {auth['bearer']}"})
        if basic:
            self.session.auth = basic
        if api_key:
            self.session.headers.update({"X-Api-Key": api_key})
        if cert:
            self.session.cert = cert
        self.session.verify = verify

    def post(self, path: str, json_body: dict) -> dict:
        url = f"{self.base_url}/{path.lstrip('/')}"
        r = self.session.post(url, json=json_body)
        r.raise_for_status()
        return r.json()

class GrpcClient:
    def __init__(self, target: str, creds: Optional[grpc.ChannelCredentials] = None):
        self.channel = grpc.secure_channel(target, creds) if creds else grpc.insecure_channel(target)
        self.stub = DatasetServiceStub(self.channel)

    def write(self, table_ipc: bytes, mode: str, options: Dict[str, Any]) -> None:
        req = WriteRequest(mode=mode, table_ipc=table_ipc, options=json.dumps(options))
        self.stub.WriteDataset(req)

    def read(self, params: Dict[str, Any]) -> bytes:
        req = ReadRequest(params=json.dumps(params))
        res = self.stub.ReadDataset(req)
        return res.table_ipc

    def list_versions(self) -> List[Dict[str, Any]]:
        req = VersionsRequest()
        res = self.stub.ListVersions(req)
        return json.loads(res.versions_json)

    def rollback(self, version: int) -> None:
        req = RollbackRequest(version=version)
        self.stub.RollbackDataset(req)

    def optimize(self, opts: Dict[str, Any]) -> None:
        req = OptimizeRequest(options=json.dumps(opts))
        self.stub.OptimizeDataset(req)

    def create_vector_index(self, column: str, opts: Dict[str, Any]) -> None:
        req = IndexRequest(column=column, options=json.dumps(opts))
        self.stub.CreateIndex(req)

    def query_sql(self, sql: str) -> Dict[str, Any]:
        req = SQLRequest(sql=sql)
        res = self.stub.QuerySQL(req)
        return {"data": json.loads(res.data_json), "columns": list(res.columns)}

class WsClient:
    def __init__(self, ws_url: str):
        self.ws = websocket.create_connection(ws_url)

    def send(self, action: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        msg = json.dumps({"action": action, "payload": payload})
        self.ws.send(msg)
        resp = self.ws.recv()
        return json.loads(resp)