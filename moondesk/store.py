"""Store immutable source images and revisioned local inspection snapshots."""

from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from threading import RLock
from uuid import uuid4

from werkzeug.exceptions import Conflict, NotFound

from .frame import encode


def stamp():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_bytes(path, payload):
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_bytes(payload)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_document(path, data):
    write_bytes(path, json.dumps(data, ensure_ascii=False, separators=(",", ":")).encode())


def read_document(path):
    if not path.is_file():
        raise NotFound("找不到该月面观测或检查记录。")
    return json.loads(path.read_text())


def identifier(value):
    if not isinstance(value, str) or len(value) != 32 or any(c not in "0123456789abcdef" for c in value):
        raise NotFound("找不到该月面观测或检查记录。")
    return value


class Library:
    def __init__(self, root):
        self.root = Path(root)
        self.panels = self.root / "panels"
        self.blobs = self.root / "images"
        self.reports = self.root / "reports"
        for directory in (self.panels, self.blobs, self.reports):
            directory.mkdir(parents=True, exist_ok=True)
        self.lock = RLock()

    def panel_path(self, panel_id):
        return self.panels / f"{identifier(panel_id)}.json"

    def panel(self, panel_id):
        return read_document(self.panel_path(panel_id))

    def source(self, record, role):
        try:
            return self.blobs / f"{record['sources'][role]['hash']}.png"
        except KeyError:
            raise NotFound("这份月面观测没有该图层。") from None

    def blob(self, image, name):
        payload = encode(image)
        fingerprint = sha256(payload).hexdigest()
        path = self.blobs / f"{fingerprint}.png"
        if not path.exists():
            write_bytes(path, payload)
        return {"hash": fingerprint, "name": name}

    def list(self):
        entries = (read_document(path) for path in self.panels.glob("*.json"))
        return sorted(entries, key=lambda record: (record["created"], record["id"]), reverse=True)

    def create(self, title, inputs, settings, synthetic=False):
        width, height = inputs["a"][0].size
        record = {
            "id": uuid4().hex, "title": title, "width": width, "height": height,
            "created": stamp(), "revision": 1, "settings": deepcopy(settings), "reports": [],
            "synthetic": synthetic,
        }
        with self.lock:
            record["sources"] = {role: self.blob(image, name) for role, (image, name) in inputs.items()}
            write_document(self.panel_path(record["id"]), record)
        return record

    @staticmethod
    def require_revision(record, expected):
        if not isinstance(expected, int) or isinstance(expected, bool) or expected != record["revision"]:
            raise Conflict("这份月面观测已被更新。请重新打开后再保存。")

    def revise(self, panel_id, expected, settings):
        with self.lock:
            record = self.panel(panel_id)
            self.require_revision(record, expected)
            record["settings"] = settings
            record["revision"] += 1
            write_document(self.panel_path(panel_id), record)
        return record

    def set_companion(self, panel_id, expected, image, filename, provenance):
        with self.lock:
            record = self.panel(panel_id)
            self.require_revision(record, expected)
            record["sources"]["b"] = {**self.blob(image, filename), "prepared": provenance}
            record["revision"] += 1
            write_document(self.panel_path(panel_id), record)
        return record

    def retain(self, panel_id, expected, result):
        with self.lock:
            record = self.panel(panel_id)
            self.require_revision(record, expected)
            saved = {
                **result, "id": uuid4().hex, "panel_id": panel_id, "title": record["title"],
                "sources": record["sources"], "created": stamp(), "source_revision": expected,
                "width": record["width"], "height": record["height"],
            }
            write_document(self.reports / f"{saved['id']}.json", saved)
            record["reports"].insert(0, {key: saved[key] for key in ("id", "created", "mode", "heading")})
            record["revision"] += 1
            write_document(self.panel_path(panel_id), record)
            return record, saved

    def report(self, panel_id, report_id):
        record = self.panel(panel_id)
        report_id = identifier(report_id)
        if not any(entry["id"] == report_id for entry in record["reports"]):
            raise NotFound("找不到这份检查记录。")
        return read_document(self.reports / f"{report_id}.json")
