"""Durable business execution positions and observable request attempts.

This module records what the application can prove; it neither executes a
request nor infers completion from a body file or an in-memory checkpoint.
"""

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
from uuid import uuid4

from lineage_agent.agent.record_codec import decode_record, encode_record


def _error(code="archive_integrity_error"):
    from lineage_agent.archive.store import ArchiveError
    return ArchiveError(code)


def _require(condition, code="archive_integrity_error"):
    if not condition:
        raise _error(code)


def _identity(value):
    _require(type(value) is str and 0 < len(value) <= 512)


def _encode(value):
    try:
        data = encode_record(value)
    except Exception:
        raise _error("archive_write_error") from None
    return data, hashlib.sha256(data).hexdigest()


def _decode(data, digest):
    _require(type(data) is bytes and type(digest) is str and hashlib.sha256(data).hexdigest() == digest)
    try:
        return decode_record(data)
    except Exception:
        raise _error() from None


def _now():
    return datetime.now(timezone.utc).isoformat()


class ExecutionArchiveMixin:
    """Private implementation shared by the one conversation archive store."""

    def _execution_writable(self, run_id):
        run = self._lifecycle(run_id)
        _require(run is not None)
        _require(type(run["revision"]) is int and run["revision"] >= 0)
        _require(run["status"] == "started", "archive_publication_conflict")
        _require(self._conversation()["current_snapshot_id"] == run["source_snapshot_id"],
                 "archive_publication_conflict")
        return run

    def _execution_revision(self, run_id):
        self._connection.execute("UPDATE run_lifecycle SET revision=revision+1 WHERE run_id=?", (run_id,))

    def _reconnect_execution(self):
        """Discard transaction-local state before resolving an uncertain commit."""
        from lineage_agent.archive.store import ArchiveStore
        _require(self._operation_depth == 0 and not self._closed)
        if self._connection is not None:
            self._connection.close()
        self._connection = None
        replacement = ArchiveStore(self.directory, create=False, read_only=self.read_only)
        self._connection = replacement._connection
        replacement._connection = None
        replacement._closed = True

    def _execution_transaction(self, write, resolve):
        try:
            with self._operation(write=True, immediate=True):
                value = write()
            return value
        except Exception as error:
            if getattr(error, "code", None) not in {"archive_write_error", "archive_read_error"}:
                raise
            # The caller allocated one identity before entering this function.
            # Resolve that identity; never mint another attempt on a COMMIT error.
            try:
                with self._lock:
                    self._reconnect_execution()
                    with self._operation():
                        value = resolve()
            except Exception:
                if self._connection is None:
                    self._closed = True
                raise _error("archive_publication_unknown") from None
            if value is not None:
                return value
            raise error

    def _execution_dependencies(self, value):
        """Validate explicitly registered candidate dependencies without adopting."""
        dependencies = value.get("dependencies", {})
        _require(type(dependencies) is dict)
        files = dependencies.get("record_files", value.get("record_files", {}))
        self._files(files)
        for ref in dependencies.get("artifacts", []):
            self._artifact(self._object("artifact", ref))
        if type(value.get("artifact")) is dict and (ref := value["artifact"].get("artifact_ref")):
            self._artifact(self._object("artifact", ref))
        for identity in dependencies.get("summaries", []):
            self._summary(self._object("summary", identity))
        for ref in dependencies.get("history_scopes", []):
            self._scope(self._object("scope", ref))
        if view_id := dependencies.get("view_id"):
            self._verify_view(view_id, False)

    def _execution_compaction(self, state, current_view):
        """Validate private H/R candidates separately from ordinary view access."""
        compaction = state.get("compaction") or {}
        _require(type(compaction) is dict)
        snapshot = compaction.get("snapshot")
        if snapshot is None:
            return
        _require(type(snapshot) is dict and snapshot.get("run_id") == state["run_id"])
        source_view = self._verify_view(snapshot.get("resource_view_id"), True)
        _require(snapshot.get("record_files") == source_view["record_files"])
        cursor, seen = current_view["view_id"], set()
        while cursor != source_view["view_id"]:
            _require(cursor is not None and cursor not in seen)
            seen.add(cursor)
            cursor = self._view(cursor, confirmed=False)["parent_id"]
        # This is a business position identity. Historical checkpoint IDs are
        # diagnostic only and are deliberately not consulted.
        if source_id := snapshot.get("source_execution_id"):
            row = self._connection.execute("SELECT payload,sha256,confirmed FROM execution_points WHERE id=?",
                                           (source_id,)).fetchone()
            _require(row is not None and row[2] == 1)
            source = _decode(row[0], row[1])
            _require(source.get("point_id") == source_id and source.get("run_id") == state["run_id"]
                     and source.get("view_id") == source_view["view_id"])
            previous = source.get("state", {})
            for key in ("input", "context_risks", "record_files", "record_revision", "active_context"):
                _require(snapshot.get(key) == previous.get(key))
            _require(snapshot.get("active_version_id") == previous.get("active_context", {}).get("version_id"))
        lanes = compaction.get("lanes")
        _require(type(lanes) is dict and set(lanes).issubset({"H", "R"}))
        for lane, selected in lanes.items():
            _require(type(selected) is dict)
            if selected.get("status") == "empty":
                continue
            scope = selected.get("scope")
            _require(type(scope) is dict)
            registered = self._object("scope", scope.get("history_ref"))
            _require(registered == scope)
            self._scope(registered)
            _require(scope.get("source_view_id") == source_view["view_id"])
            for run, watermark in scope["record_files"].items():
                _require(run in source_view["record_files"])
                self._file_extension(run, watermark, source_view["record_files"][run])
            previous = selected.get("previous", [])
            _require(type(previous) is list)
            for reference in previous:
                _require(type(reference) is dict and reference.get("summary_id") in source_view["summaries"])
                old = self._object("summary", reference["summary_id"])
                self._summary(old)
                for key in ("lane", "history_ref", "ranges"):
                    _require(reference.get(key) == old.get(key))
            candidate = selected.get("candidate_input")
            if candidate is not None:
                _require(type(candidate) is dict)
                old = self._object("summary", candidate.get("summary_id"))
                self._summary(old)
                _require(old["body"] == candidate.get("body") and old["run_id"] == state["run_id"]
                         and old["lane"] == lane and old["history_ref"] == scope["history_ref"])
            if selected.get("status") in {"generated", "reused"}:
                saved = self._object("summary", selected.get("summary_id"))
                self._summary(saved)
                expected_previous = ([candidate["summary_id"]] if candidate is not None else
                                     [ref["summary_id"] for ref in previous])
                _require(saved.get("run_id") == state["run_id"] and saved.get("lane") == lane
                         and saved.get("body") == selected.get("body")
                         and saved.get("history_ref") == scope["history_ref"] and saved.get("ranges") == scope["ranges"]
                         and saved.get("previous", []) == expected_previous
                         and saved.get("source_refs", []) == ([] if candidate is not None else selected.get("source_refs", [])))

    def _execution_row(self, point_id, *, confirmed=True, validate_operation=True):
        row = self._connection.execute(
            "SELECT run_id,sequence,parent_id,view_id,payload,sha256,confirmed,checkpoint_id "
            "FROM execution_points WHERE id=?", (point_id,)).fetchone()
        _require(row is not None and (not confirmed or row[6] == 1))
        value = _decode(row[4], row[5])
        _require(value.get("point_id") == point_id and value.get("conversation_id") == self.conversation_id
                 and tuple(value.get(k) for k in ("run_id", "sequence", "parent_id", "view_id")) == row[:4])
        from lineage_agent.archive.store import RECOVERY_VERSION
        _require(value.get("recovery_version") == RECOVERY_VERSION, "archive_version_error")
        _require(type(value.get("phase")) is str and bool(value["phase"]))
        view = self._verify_view(value["view_id"], confirmed)
        state = value.get("state")
        _require(type(state) is dict and state.get("run_id") == value["run_id"]
                 and state.get("conversation_id") == self.conversation_id
                 and state.get("resource_view_id") == value["view_id"]
                 and state.get("record_files", {}) == view["record_files"]
                 and state.get("history_scopes", {}) == view["history_scopes"])
        execution = state.get("execution", {})
        _require(type(execution) is dict)
        if execution.get("point_id") is not None:
            _require(execution["point_id"] == point_id)
        if "parent_id" in execution:
            _require(execution["parent_id"] == value["parent_id"])
        if execution.get("phase") is not None:
            _require(execution["phase"] == value["phase"])
        operation_id = execution.get("operation_id")
        if operation_id is not None:
            if validate_operation:
                _require(self._operation_row(operation_id)["run_id"] == value["run_id"])
            else:
                binding = self._connection.execute("SELECT run_id,kind FROM operations WHERE id=?",
                                                   (operation_id,)).fetchone()
                _require(binding is not None and binding[0] == value["run_id"] and binding[1] != "finalization")
        self._execution_compaction(state, view)
        state["record_files"] = {run: deepcopy(view["record_files"][run]) for run in view["record_order"]}
        return {**value, "confirmed": bool(row[6]), "checkpoint_id": row[7]}

    def prepare_execution(self, run_id, state, view_id, phase, parent_id, *, point_id=None):
        """Stage an immutable candidate; only confirm_execution makes it current."""
        identity = point_id or str(uuid4())
        _identity(identity)
        _identity(phase)

        def existing():
            if self._connection.execute("SELECT 1 FROM execution_points WHERE id=?", (identity,)).fetchone() is None:
                return None
            value = self._execution_row(identity, confirmed=False)
            _require(value["run_id"] == run_id and value["state"] == state and value["view_id"] == view_id
                     and value["phase"] == phase and value["parent_id"] == parent_id)
            return identity

        def write():
            if result := existing():
                return result
            run = self._execution_writable(run_id)
            _require(run["current_execution_id"] == parent_id, "archive_publication_conflict")
            sequence = 1
            if parent_id is not None:
                parent = self._execution_row(parent_id)
                _require(parent["run_id"] == run_id)
                sequence = parent["sequence"] + 1
                cursor, seen = view_id, set()
                while cursor != parent["view_id"]:
                    _require(cursor is not None and cursor not in seen)
                    seen.add(cursor)
                    cursor = self._view(cursor, confirmed=False)["parent_id"]
            from lineage_agent.archive.store import RECOVERY_VERSION
            value = {"point_id": identity, "conversation_id": self.conversation_id,
                     "run_id": run_id, "sequence": sequence, "parent_id": parent_id,
                     "source_snapshot_id": run["source_snapshot_id"], "view_id": view_id,
                     "phase": phase, "state": deepcopy(state), "recovery_version": RECOVERY_VERSION,
                     "created_at": _now()}
            data, digest = _encode(value)
            self._connection.execute("INSERT INTO execution_points(id,run_id,sequence,parent_id,view_id,payload,sha256) "
                                     "VALUES (?,?,?,?,?,?,?)", (identity, run_id, sequence, parent_id, view_id, data, digest))
            self._execution_row(identity, confirmed=False)
            return identity

        return self._execution_transaction(write, existing)

    def confirm_execution(self, point_id, checkpoint_id, expected_parent):
        """Confirm the view and business position together after memory save."""
        _identity(checkpoint_id)

        def existing():
            row = self._connection.execute("SELECT confirmed FROM execution_points WHERE id=?", (point_id,)).fetchone()
            if row is None or not row[0]:
                return None
            value = self._execution_row(point_id)
            _require(value["parent_id"] == expected_parent and value["checkpoint_id"] == checkpoint_id)
            run = self._lifecycle(value["run_id"])
            _require(run is not None)
            return {**value, "revision": run["revision"]}

        def write():
            value = self._execution_row(point_id, confirmed=False)
            run = self._execution_writable(value["run_id"])
            if result := existing():
                return result
            _require(value["parent_id"] == expected_parent == run["current_execution_id"],
                     "archive_publication_conflict")
            self._confirm_view_rows(value["view_id"], checkpoint_id)
            self._validate_snapshot_lineage(value["run_id"], value["view_id"], run["source_snapshot_id"])
            self._connection.execute("UPDATE execution_points SET confirmed=1,checkpoint_id=? WHERE id=?",
                                     (checkpoint_id, point_id))
            self._connection.execute("UPDATE run_lifecycle SET current_execution_id=?,revision=revision+1 WHERE run_id=?",
                                     (point_id, value["run_id"]))
            return existing()

        return self._execution_transaction(write, existing)

    def execution(self, point_id, *, confirmed=True):
        """Read a fixed position for trusted inspection, never permission to run."""
        with self._operation():
            return self._execution_row(point_id, confirmed=confirmed)

    def read_execution(self, run_id):
        """Read the only resumable head; completed/abandoned runs are rejected."""
        with self._operation():
            run = self._execution_writable(run_id)
            _require(run["current_execution_id"] is not None, "archive_no_execution")
            value = self._execution_row(run["current_execution_id"])
            count, latest = self._connection.execute(
                "SELECT COUNT(*),MAX(sequence) FROM execution_points WHERE run_id=? AND confirmed=1", (run_id,)).fetchone()
            _require(count == latest == value["sequence"])
            seen, cursor, sequence = set(), value, value["sequence"]
            while True:
                _require(cursor["point_id"] not in seen and cursor["run_id"] == run_id
                         and cursor["sequence"] == sequence
                         and cursor["source_snapshot_id"] == run["source_snapshot_id"])
                seen.add(cursor["point_id"])
                if cursor["parent_id"] is None:
                    _require(sequence == 1)
                    break
                sequence -= 1
                cursor = self._execution_row(cursor["parent_id"])
            self._validate_snapshot_lineage(run_id, value["view_id"], run["source_snapshot_id"])
            return {**value, "revision": run["revision"]}

    def _operation_row(self, operation_id):
        row = self._connection.execute("SELECT run_id,sequence,kind,payload,sha256 FROM operations WHERE id=?",
                                       (operation_id,)).fetchone()
        _require(row is not None)
        envelope = _decode(row[3], row[4])
        _require(envelope.get("operation_id") == operation_id and envelope.get("run_id") == row[0]
                 and envelope.get("sequence") == row[1] and envelope.get("kind") == row[2])
        payload = envelope.get("payload")
        _require(type(payload) is dict)
        self._execution_dependencies(payload)
        if row[2] == "finalization":
            state = payload.get("final_state")
            _require(type(state) is dict and type(state.get("execution")) is dict)
            progress = state["execution"]
            _identity(progress.get("final_record_id"))
            # Validate the exact staged terminal transition even before it is
            # confirmed. Do not recurse through its operation binding or gate
            # historical inspection on the current lifecycle head.
            candidate = self._execution_row(progress.get("point_id"), confirmed=False, validate_operation=False)
            _require(candidate["run_id"] == row[0] and candidate["phase"] == "finalizing"
                     and _encode(candidate["state"]) == _encode(state))
            view = self._verify_view(candidate["view_id"], False)
            _require(type(payload.get("fact_end")) is int
                     and payload["fact_end"] == view["fact_ends"].get(row[0], 0))
        return {"operation_id": operation_id, "run_id": row[0], "sequence": row[1], "kind": row[2],
                "payload": payload}

    def save_operation(self, run_id, operation_id, kind, payload):
        _identity(operation_id)
        _identity(kind)
        data, digest = _encode(payload)

        def existing():
            if self._connection.execute("SELECT 1 FROM operations WHERE id=?", (operation_id,)).fetchone() is None:
                return None
            value = self._operation_row(operation_id)
            _require(value["run_id"] == run_id and value["kind"] == kind and _encode(value["payload"]) == (data, digest))
            return value

        def write():
            if value := existing():
                return value
            self._execution_writable(run_id)
            self._execution_dependencies(payload)
            sequence = self._connection.execute("SELECT COALESCE(MAX(sequence),0)+1 FROM operations WHERE run_id=?",
                                                 (run_id,)).fetchone()[0]
            envelope, envelope_hash = _encode({"operation_id": operation_id, "run_id": run_id,
                                               "sequence": sequence, "kind": kind, "payload": payload})
            self._connection.execute("INSERT INTO operations VALUES (?,?,?,?,?,?)",
                                     (operation_id, run_id, sequence, kind, envelope, envelope_hash))
            self._execution_revision(run_id)
            return existing()

        return self._execution_transaction(write, existing)

    def operation(self, operation_id):
        with self._operation():
            return self._operation_row(operation_id)

    def operations(self, run_id):
        with self._operation():
            values = [self._operation_row(row[0]) for row in self._connection.execute(
                "SELECT id FROM operations WHERE run_id=? ORDER BY sequence", (run_id,)).fetchall()]
            _require([item["sequence"] for item in values] == list(range(1, len(values) + 1)))
            return values

    def _operation_result(self, operation_id, slot):
        self._operation_row(operation_id)
        row = self._connection.execute("SELECT payload,sha256 FROM operation_results WHERE operation_id=? AND slot=?",
                                       (operation_id, slot)).fetchone()
        if row is None:
            return None
        envelope = _decode(*row)
        _require(envelope.get("operation_id") == operation_id and envelope.get("slot") == slot)
        value = envelope.get("result")
        _require(type(value) is dict)
        self._execution_dependencies(value)
        return value

    def save_operation_result(self, operation_id, result, *, slot="processed"):
        _identity(slot)
        data, digest = _encode(result)

        def existing():
            value = self._operation_result(operation_id, slot)
            if value is not None:
                _require(_encode(value) == (data, digest))
            return value

        def write():
            value = existing()
            if value is not None:
                return value
            operation = self._operation_row(operation_id)
            self._execution_writable(operation["run_id"])
            self._execution_dependencies(result)
            envelope, envelope_hash = _encode({"operation_id": operation_id, "slot": slot, "result": result})
            self._connection.execute("INSERT INTO operation_results VALUES (?,?,?,?)",
                                     (operation_id, slot, envelope, envelope_hash))
            self._execution_revision(operation["run_id"])
            return deepcopy(result)

        return self._execution_transaction(write, existing)

    def operation_result(self, operation_id, *, slot="processed"):
        with self._operation():
            return self._operation_result(operation_id, slot)

    def _attempt_row(self, attempt_id):
        row = self._connection.execute("SELECT operation_id,sequence,payload,sha256,result,result_sha256 FROM attempts WHERE id=?",
                                       (attempt_id,)).fetchone()
        _require(row is not None)
        value = _decode(row[2], row[3])
        _require(value.get("attempt_id") == attempt_id and value.get("operation_id") == row[0]
                 and value.get("sequence") == row[1])
        reservation = value.get("reservation")
        _require(type(reservation) is dict and all(type(k) is str and type(v) is int and v >= 0 for k, v in reservation.items()))
        _require((row[4] is None) == (row[5] is None))
        envelope = _decode(row[4], row[5]) if row[4] is not None else None
        result = None
        if envelope is not None:
            _require(envelope.get("attempt_id") == attempt_id)
            result = envelope.get("result")
            _require(type(result) is dict)
        if result is not None:
            self._execution_dependencies(result)
            used = result.get("reservation_used", reservation)
            _require(type(used) is dict and set(used).issubset(reservation)
                     and all(type(v) is int and 0 <= v <= reservation[k] for k, v in used.items()))
            if type(result.get("artifact")) is dict and (ref := result["artifact"].get("artifact_ref")):
                saved = self._object("artifact", ref)
                operation = self._operation_row(row[0])
                origin = saved["metadata"].get("origin", {})
                _require(origin.get("run_id") == operation["run_id"] and origin.get("operation_id") == row[0]
                         and origin.get("attempt_id") == attempt_id and origin.get("purpose") == "execution_result")
                if "sha256" in result["artifact"]:
                    _require(result["artifact"]["sha256"] == saved["sha256"])
        return {**value, "status": "completed" if result is not None else "started", "result": result}

    def start_attempt(self, operation_id, reservation, *, attempt_id=None):
        identity = attempt_id or str(uuid4())
        _identity(identity)
        _require(type(reservation) is dict and all(type(k) is str and type(v) is int and v >= 0 for k, v in reservation.items()))

        def existing():
            if self._connection.execute("SELECT 1 FROM attempts WHERE id=?", (identity,)).fetchone() is None:
                return None
            value = self._attempt_row(identity)
            _require(value["operation_id"] == operation_id and value["reservation"] == reservation)
            return value

        def write():
            if value := existing():
                return value
            operation = self._operation_row(operation_id)
            self._execution_writable(operation["run_id"])
            previous = [self._attempt_row(row[0]) for row in self._connection.execute(
                "SELECT id FROM attempts WHERE operation_id=? ORDER BY sequence", (operation_id,)).fetchall()]
            _require(len(previous) < 2 and all(item["result"] is None for item in previous), "archive_attempt_limit")
            _require(self._operation_result(operation_id, "processed") is None, "archive_publication_conflict")
            value = {"attempt_id": identity, "operation_id": operation_id, "sequence": len(previous) + 1,
                     "reservation": deepcopy(reservation), "started_at": _now()}
            data, digest = _encode(value)
            self._connection.execute("INSERT INTO attempts(id,operation_id,sequence,payload,sha256) VALUES (?,?,?,?,?)",
                                     (identity, operation_id, value["sequence"], data, digest))
            self._execution_revision(operation["run_id"])
            return existing()

        return self._execution_transaction(write, existing)

    def finish_attempt(self, attempt_id, result):
        data, digest = _encode(result)

        def existing():
            attempt = self._attempt_row(attempt_id)
            if attempt["result"] is None:
                return None
            _require(_encode(attempt["result"]) == (data, digest))
            return attempt

        def write():
            if value := existing():
                return value
            attempt = self._attempt_row(attempt_id)
            operation = self._operation_row(attempt["operation_id"])
            self._execution_writable(operation["run_id"])
            _require(self._connection.execute(
                "SELECT 1 FROM attempts WHERE operation_id=? AND id<>? AND result IS NOT NULL",
                (attempt["operation_id"], attempt_id)).fetchone() is None, "archive_publication_conflict")
            used = result.get("reservation_used", attempt["reservation"])
            _require(type(used) is dict and set(used).issubset(attempt["reservation"])
                     and all(type(v) is int and 0 <= v <= attempt["reservation"][k] for k, v in used.items()))
            self._execution_dependencies(result)
            envelope, envelope_hash = _encode({"attempt_id": attempt_id, "result": result})
            self._connection.execute("UPDATE attempts SET result=?,result_sha256=? WHERE id=?",
                                     (envelope, envelope_hash, attempt_id))
            self._execution_revision(operation["run_id"])
            return existing()

        return self._execution_transaction(write, existing)

    def attempts(self, operation_id):
        with self._operation():
            self._operation_row(operation_id)
            values = [self._attempt_row(row[0]) for row in self._connection.execute(
                "SELECT id FROM attempts WHERE operation_id=? ORDER BY sequence", (operation_id,)).fetchall()]
            _require([item["sequence"] for item in values] == list(range(1, len(values) + 1)))
            _require(sum(item["result"] is not None for item in values) <= 1)
            return values

    def budget_projection(self, run_id):
        """Each persisted attempt counts once; an unknown attempt keeps its hold."""
        with self._operation():
            reserved, used, charged_total, unknown, kinds = {}, {}, {}, [], {}
            rows = self._connection.execute("SELECT a.id,o.id FROM attempts a JOIN operations o ON a.operation_id=o.id "
                                            "WHERE o.run_id=? ORDER BY o.sequence,a.sequence", (run_id,)).fetchall()
            for attempt_id, operation_id in rows:
                kind = self._operation_row(operation_id)["kind"]
                attempt = self._attempt_row(attempt_id)
                result = attempt["result"]
                charged = result.get("reservation_used", attempt["reservation"]) if result is not None else attempt["reservation"]
                if result is None:
                    unknown.append(attempt_id)
                bucket = kinds.setdefault(kind, {"reserved": {}, "used": {}, "charged": {}, "attempt_count": 0})
                bucket["attempt_count"] += 1
                for key, amount in attempt["reservation"].items():
                    actual = charged.get(key, amount)
                    target, category = (reserved, "reserved") if result is None else (used, "used")
                    target[key] = target.get(key, 0) + actual
                    bucket[category][key] = bucket[category].get(key, 0) + actual
                    charged_total[key] = charged_total.get(key, 0) + actual
                    bucket["charged"][key] = bucket["charged"].get(key, 0) + actual
            return {"reserved": reserved, "used": used, "charged": charged_total, "unknown_attempt_ids": unknown,
                    "attempt_count": len(rows), "by_kind": kinds}

    def accept_resume(self, run_id, point_id, revision, *, choice_id=None):
        """Accept one exact inspected head, without executing or creating a Run."""
        identity = choice_id or str(uuid4())
        _identity(identity)
        _require(type(revision) is int and revision >= 0)

        def existing():
            row = self._connection.execute("SELECT payload,sha256 FROM recovery_choices WHERE id=?", (identity,)).fetchone()
            if row is None:
                return None
            value = _decode(*row)
            _require(value.get("kind") == "resume" and value.get("run_id") == run_id
                     and value.get("point_id") == point_id and value.get("revision") == revision)
            return identity

        def write():
            if value := existing():
                return value
            run = self._execution_writable(run_id)
            _require(run["current_execution_id"] == point_id and run["revision"] == revision,
                     "archive_publication_conflict")
            point = self._execution_row(point_id)
            _require(point["run_id"] == run_id)
            value = {"choice_id": identity, "kind": "resume", "run_id": run_id, "point_id": point_id,
                     "revision": revision, "created_at": _now()}
            data, digest = _encode(value)
            self._connection.execute("INSERT INTO recovery_choices VALUES (?,?,?)", (identity, data, digest))
            self._execution_revision(run_id)
            return identity

        return self._execution_transaction(write, existing)

    def run_status(self, run_id):
        """Inspect lifecycle metadata without granting execution permission."""
        with self._operation():
            self._conversation()
            return self._lifecycle(run_id)

    def resume_choice(self, choice_id):
        """Read an exact acceptance receipt so a repeated API call can report it."""
        with self._operation():
            row = self._connection.execute("SELECT payload,sha256 FROM recovery_choices WHERE id=?",
                                           (choice_id,)).fetchone()
            if row is None:
                return None
            value = _decode(*row)
            _require(value.get("choice_id") == choice_id and value.get("kind") == "resume")
            return value
