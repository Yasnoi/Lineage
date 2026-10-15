"""Business recovery positions, explicit result ownership and uncertain commits."""

from copy import deepcopy
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from uuid import uuid4

from lineage_agent.archive import ArchiveError, ArchiveStore
from lineage_agent.agent.record_journal import RecordJournal
from lineage_agent.agent.record_codec import encode_record
from lineage_agent.agent.records import make_record_batch
from lineage_agent.storage import conversation_paths


class ExecutionArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory(prefix="execution-archive-")
        self.addCleanup(self.temporary.cleanup)
        self.paths = conversation_paths(Path(self.temporary.name), "conversation")
        self.archive = ArchiveStore(self.paths.directory)
        self.addCleanup(self.archive.close)
        self.journal = RecordJournal(self.paths.history)
        self.addCleanup(self.journal.close)
        self.archive.register_conversation({"settings": {"model": "test"}}, [])
        self.archive.begin_run("run", None)

    def point(self, identity="p1", parent=None, *, confirm=True):
        previous = self.archive.execution(parent) if parent else None
        _, watermark = self.journal.append("run", make_record_batch("run", 0, [{"role": "user", "content": "question"}]))
        view = self.archive.prepare_view(record_files={"run": watermark}, history_scopes={}, summaries=[], artifacts=[],
                                         parent_id=previous["view_id"] if previous else None)
        state = {"run_id": "run", "conversation_id": "conversation", "resource_view_id": view,
                 "record_files": {"run": watermark}, "history_scopes": {}, "input": [{"role": "user", "content": "question"}],
                 "history_end": 1, "active_context": {"turn_start": 0}, "context_risks": [None],
                 "execution": {"point_id": identity, "parent_id": parent, "phase": "model_ready"}}
        point = self.archive.prepare_execution("run", state, view, "model_ready", parent, point_id=identity)
        if confirm:
            self.archive.confirm_execution(point, f"cp-{identity}", parent)
        return point, view, state

    def operation(self, identity="op", kind="model"):
        return self.archive.save_operation("run", identity, kind, {"request": {"input": ["fixed"]}})

    def test_first_run_has_execution_point_without_final_snapshot(self):
        self.point()
        current = self.archive.read_execution("run")
        self.assertEqual(current["point_id"], "p1")
        self.assertEqual(current["phase"], "model_ready")
        self.assertEqual(current["state"]["input"][0]["content"], "question")
        self.assertIsNone(self.archive.conversation()["current_snapshot_id"])
        with self.assertRaises(ArchiveError):
            self.archive.read_run("run")
        self.archive.close()
        with ArchiveStore(self.paths.directory, create=False) as reopened:
            self.assertEqual(reopened.read_execution("run")["state"], current["state"])

    def test_many_points_do_not_publish_final_snapshot(self):
        self.point()
        self.point("p2", "p1")
        self.point("p3", "p2")
        self.assertEqual(self.archive.read_execution("run")["sequence"], 3)
        self.assertEqual(self.archive.execution("p1")["sequence"], 1)
        self.assertIsNone(self.archive.metadata()["head"])
        self.assertEqual(self.archive.confirm_execution("p2", "cp-p2", "p1")["point_id"], "p2")
        self.assertEqual(self.archive.read_execution("run")["point_id"], "p3")

    def test_prepared_point_and_view_stay_unconfirmed_until_joint_commit(self):
        point, view, _ = self.point(confirm=False)
        with self.assertRaises(ArchiveError) as error:
            self.archive.read_execution("run")
        self.assertEqual(error.exception.code, "archive_no_execution")
        with self.assertRaises(ArchiveError):
            self.archive.view(view)
        native = self.archive._confirm_view_rows

        def fail_after_view(*args):
            native(*args)
            raise OSError("fail before position commit")

        with patch.object(self.archive, "_confirm_view_rows", side_effect=fail_after_view), self.assertRaises(ArchiveError):
            self.archive.confirm_execution(point, "cp", None)
        self.assertFalse(self.archive.execution(point, confirmed=False)["confirmed"])
        with self.assertRaises(ArchiveError):
            self.archive.view(view)
        self.archive.confirm_execution(point, "cp", None)
        self.assertEqual(self.archive.read_execution("run")["point_id"], point)

    def test_commit_after_success_is_resolved_by_original_point_identity(self):
        point, _, _ = self.point(confirm=False)
        native = self.archive._commit

        def uncertain():
            native()
            raise OSError("ack lost")

        with patch.object(self.archive, "_commit", side_effect=uncertain):
            result = self.archive.confirm_execution(point, "cp", None)
        self.assertEqual(result["point_id"], point)
        self.assertEqual(self.archive.read_execution("run")["sequence"], 1)

    def test_stale_parent_and_forged_state_point_are_rejected(self):
        _, view, state = self.point()
        with self.assertRaises(ArchiveError):
            self.archive.prepare_execution("run", state, view, "model_ready", None, point_id="forged")
        candidate = deepcopy(state)
        candidate["execution"] = {"point_id": "p2", "parent_id": "p1", "phase": "model_ready"}
        self.archive.prepare_execution("run", candidate, view, "model_ready", "p1", point_id="p2")
        with self.assertRaises(ArchiveError):
            self.archive.confirm_execution("p2", "cp2", None)
        self.assertEqual(self.archive.read_execution("run")["point_id"], "p1")

    def test_corrupted_head_cannot_silently_rewind_confirmed_progress(self):
        self.point()
        self.point("p2", "p1")
        self.archive._connection.execute("UPDATE run_lifecycle SET current_execution_id='p1' WHERE run_id='run'")
        self.archive._connection.commit()
        with self.assertRaises(ArchiveError):
            self.archive.read_execution("run")

    def test_new_execution_point_cannot_regress_its_parent_resource_view(self):
        _, old_view, state = self.point()
        self.point("p2", "p1")
        state["execution"] = {"point_id": "p3", "parent_id": "p2", "phase": "model_ready"}
        with self.assertRaises(ArchiveError):
            self.archive.prepare_execution("run", state, old_view, "model_ready", "p2", point_id="p3")
        self.assertEqual(self.archive.read_execution("run")["point_id"], "p2")

    def test_operation_and_result_slots_are_immutable_and_precisely_bound(self):
        operation = self.operation()
        self.assertEqual(self.operation(), operation)
        with self.assertRaises(ArchiveError):
            self.archive.save_operation("run", "op", "model", {"request": "changed"})
        result = {"usage": None, "response": {"text": "known"}}
        self.assertEqual(self.archive.save_operation_result("op", result), result)
        self.assertEqual(self.archive.save_operation_result("op", result), result)
        self.assertEqual(self.archive.operation_result("op"), result)
        self.assertIsNone(self.archive.operation_result("op", slot="plan"))
        with self.assertRaises(ArchiveError):
            self.archive.save_operation_result("op", {"usage": 0})
        self.operation("other")
        self.archive._connection.execute("UPDATE operation_results SET operation_id='other' WHERE operation_id='op'")
        self.archive._connection.commit()
        with self.assertRaises(ArchiveError):
            self.archive.operation_result("other")

    def test_unknown_attempts_reserve_once_and_only_one_retry_is_allowed(self):
        self.operation()
        first = self.archive.start_attempt("op", {"model_calls": 1}, attempt_id="a1")
        self.assertEqual(self.archive.start_attempt("op", {"model_calls": 1}, attempt_id="a1"), first)
        self.archive.start_attempt("op", {"model_calls": 1}, attempt_id="a2")
        with self.assertRaises(ArchiveError) as error:
            self.archive.start_attempt("op", {"model_calls": 1}, attempt_id="a3")
        self.assertEqual(error.exception.code, "archive_attempt_limit")
        budget = self.archive.budget_projection("run")
        self.assertEqual(budget["used"], {})
        self.assertEqual(budget["reserved"], {"model_calls": 2})
        self.assertEqual(budget["charged"], {"model_calls": 2})
        self.assertEqual(budget["unknown_attempt_ids"], ["a1", "a2"])
        self.assertEqual(budget["attempt_count"], 2)

    def test_complete_attempt_reuses_usage_and_settles_only_known_bytes(self):
        self.operation(kind="tool")
        self.archive.start_attempt("op", {"tool_calls": 1, "read_bytes": 100}, attempt_id="a1")
        result = {"output": "body", "usage": None, "reservation_used": {"read_bytes": 40}}
        completed = self.archive.finish_attempt("a1", result)
        self.assertEqual(self.archive.finish_attempt("a1", result), completed)
        self.assertIsNone(completed["result"]["usage"])
        self.assertEqual(self.archive.budget_projection("run")["used"], {"tool_calls": 1, "read_bytes": 40})
        self.assertEqual(self.archive.budget_projection("run")["reserved"], {})
        self.assertEqual(self.archive.budget_projection("run")["unknown_attempt_ids"], [])
        with self.assertRaises(ArchiveError):
            self.archive.start_attempt("op", {"tool_calls": 1})
        with self.assertRaises(ArchiveError):
            self.archive.finish_attempt("a1", {"output": "different"})

    def test_attempt_commit_uncertainty_never_allocates_another_attempt(self):
        self.operation()
        native = self.archive._commit

        def uncertain():
            native()
            raise OSError("commit reply lost")

        with patch.object(self.archive, "_commit", side_effect=uncertain):
            started = self.archive.start_attempt("op", {"model_calls": 1}, attempt_id="a1")
            finished = self.archive.finish_attempt("a1", {"usage": None, "response": "saved"})
        self.assertEqual(started["attempt_id"], finished["attempt_id"])
        self.assertEqual(len(self.archive.attempts("op")), 1)

    def test_result_dependency_corruption_is_not_treated_as_unknown_attempt(self):
        _, _, state = self.point()
        self.operation()
        self.archive.start_attempt("op", {"model_calls": 1}, attempt_id="a1")
        self.archive.finish_attempt("a1", {"response": "known", "dependencies": {"record_files": state["record_files"]}})
        self.journal.path("run").write_bytes(b"corrupted\n")
        with self.assertRaises(ArchiveError):
            self.archive.attempts("op")
        with self.assertRaises(ArchiveError):
            self.archive.start_attempt("op", {"model_calls": 1})

    def test_tool_candidate_is_bound_to_its_exact_attempt_before_adoption(self):
        self.operation(kind="tool")
        self.archive.start_attempt("op", {"tool_calls": 1}, attempt_id="a1")
        ref = "artifact://tool/" + "a" * 32
        raw = b"trusted tool result"
        path = self.paths.artifacts / ("a" * 32 + ".raw")
        path.write_bytes(raw)
        digest = hashlib.sha256(raw).hexdigest()
        metadata = {"artifact_ref": ref, "raw_bytes": len(raw), "total_chars": len(raw),
                    "tool_name": "custom", "view_kind": "raw_result", "origin": {
                        "run_id": "run", "operation_id": "op", "attempt_id": "a1", "purpose": "execution_result"}}
        self.archive.save_artifact({"artifact_ref": ref, "path": "artifacts/" + path.name,
                                    "sha256": digest, "metadata": metadata})
        result = {"artifact": {**metadata, "sha256": digest}}
        self.archive.finish_attempt("a1", result)
        self.operation("other", kind="tool")
        self.archive.start_attempt("other", {"tool_calls": 1}, attempt_id="a2")
        with self.assertRaises(ArchiveError):
            self.archive.finish_attempt("a2", result)
        self.assertIsNone(self.archive.attempts("other")[0]["result"])
        path.write_bytes(b"changed")
        with self.assertRaises(ArchiveError):
            self.archive.attempts("op")

    def test_resume_choice_requires_exact_head_and_ledger_revision(self):
        self.point()
        inspected = self.archive.read_execution("run")
        self.operation()
        with self.assertRaises(ArchiveError):
            self.archive.accept_resume("run", "p1", inspected["revision"])
        inspected = self.archive.read_execution("run")
        choice = self.archive.accept_resume("run", "p1", inspected["revision"], choice_id="choice")
        self.assertEqual(self.archive.accept_resume("run", "p1", inspected["revision"], choice_id="choice"), choice)
        self.assertEqual(len(self.archive.recovery_choices()), 1)
        self.assertEqual(self.archive.read_execution("run")["point_id"], "p1")
        self.assertEqual(self.archive.operations("run")[0]["operation_id"], "op")

    def test_fixed_final_identity_and_publication_close_execution(self):
        _, view, state = self.point()
        state["stop_reason"] = "completed"
        payload = {"final_state": state, "fact_end": 0}
        identity = self.archive.stage_run("run", payload, record_id="final")
        self.assertEqual(self.archive.stage_run("run", payload, record_id="final"), identity)
        candidate = self.archive.staged_run(identity)
        self.assertEqual(candidate["payload"], payload)
        self.assertFalse(candidate["sealed"])
        with self.assertRaises(ArchiveError):
            self.archive.read_run("run")
        snapshot = self.archive.publish_snapshot(identity, view, "cp-p1", None)
        self.assertTrue(self.archive.staged_run(identity)["sealed"])
        self.assertEqual(snapshot["snapshot_id"], "final")
        self.assertEqual(self.archive.execution("p1")["point_id"], "p1")
        for action in (lambda: self.archive.read_execution("run"),
                       lambda: self.archive.accept_resume("run", "p1", 1),
                       lambda: self.archive.confirm_execution("p1", "cp-p1", None)):
            with self.assertRaises(ArchiveError):
                action()
        self.assertEqual(self.archive.publish_snapshot(identity, view, "cp-p1", None), snapshot)

    def test_staged_final_candidate_rechecks_bodies_and_never_fakes_sealing(self):
        _, _, state = self.point()
        self.archive.stage_run("run", {"final_state": state, "fact_end": 0}, record_id="final")
        self.assertFalse(self.archive.staged_run("final")["sealed"])
        self.journal.path("run").write_bytes(b"damaged")
        with self.assertRaises(ArchiveError):
            self.archive.staged_run("final")

    def final_operation(self):
        _, parent_view, state = self.point()
        _, watermark = self.journal.append("run", make_record_batch(
            "run", 1, [{"stop_reason": "completed"}], kind="outcome"))
        view = self.archive.prepare_view(record_files={"run": watermark}, history_scopes={},
                                         summaries=[], artifacts=[], parent_id=parent_view)
        state.update(resource_view_id=view, record_files={"run": watermark})
        state.update(stop_reason="completed", execution={"point_id": "final-point", "parent_id": "p1",
                                                        "phase": "finalizing", "final_record_id": "final-record"})
        self.archive.prepare_execution("run", state, view, "finalizing", "p1", point_id="final-point")
        payload = {"final_state": state, "fact_end": 0}
        self.archive.save_operation("run", "final-operation", "finalization", payload)
        return state, view, payload

    def test_unconfirmed_final_operation_validates_candidate_and_bodies_during_inspection(self):
        _, _, payload = self.final_operation()
        self.assertEqual(self.archive.operation("final-operation")["payload"], payload)
        current = self.archive.read_execution("run")
        self.assertEqual(current["point_id"], "p1")
        confirmed_end = current["state"]["record_files"]["run"]["byte_end"]
        path = self.journal.path("run")
        path.write_bytes(path.read_bytes()[:confirmed_end])
        self.assertEqual(self.archive.read_execution("run")["point_id"], "p1")
        with self.assertRaises(ArchiveError):
            self.archive.operations("run")

    def test_final_operation_rejects_different_state_or_fact_boundary(self):
        _, _, payload = self.final_operation()
        for changed in ({**payload, "fact_end": 1},
                        {**payload, "final_state": {**payload["final_state"], "stop_reason": "error"}}):
            with self.assertRaises(ArchiveError):
                self.archive.save_operation("run", str(uuid4()), "finalization", changed)

    def test_final_operation_remains_readable_after_publication(self):
        _, view, payload = self.final_operation()
        self.archive.confirm_execution("final-point", "cp-final", "p1")
        self.archive.stage_run("run", payload, record_id="final-record")
        self.archive.publish_snapshot("final-record", view, "cp-final", None)
        self.assertEqual(self.archive.operation("final-operation")["payload"], payload)

    def compact_candidate(self):
        _, source_view, state = self.point()
        scope = {"history_ref": f"history://scope/{uuid4()}", "source_view_id": source_view,
                 "ranges": [{"run_id": "run", "start": 0, "end": 1}], "record_files": state["record_files"]}
        self.archive.save_scope(scope)
        summary = {"summary_id": "candidate", "run_id": "run", "lane": "R", "body": "fixed candidate",
                   "history_ref": scope["history_ref"], "ranges": scope["ranges"], "previous": [],
                   "source_refs": ["history://record/run/0"]}
        self.archive.save_summary(summary)
        view = self.archive.prepare_view(record_files=state["record_files"], history_scopes={scope["history_ref"]: scope},
                                         summaries=[], artifacts=[], parent_id=source_view)
        snapshot = {key: deepcopy(state[key]) for key in ("run_id", "input", "context_risks", "active_context", "record_files")}
        snapshot.update(resource_view_id=source_view, source_execution_id="p1", active_version_id=None)
        state.update(resource_view_id=view, history_scopes={scope["history_ref"]: scope}, compaction={
            "snapshot": snapshot, "lanes": {"H": {"status": "empty"}, "R": {
                "status": "generated", "summary_id": "candidate", "body": "fixed candidate", "scope": scope,
                "previous": [], "source_refs": ["history://record/run/0"]}}})
        state["execution"] = {"point_id": "p2", "parent_id": "p1", "phase": "validate_candidate"}
        return state, view, summary

    def test_private_compaction_candidate_is_validated_without_view_adoption(self):
        state, view, summary = self.compact_candidate()
        self.archive.prepare_execution("run", state, view, "validate_candidate", "p1", point_id="p2")
        self.archive.confirm_execution("p2", "cp2", "p1")
        self.assertEqual(self.archive.read_execution("run")["state"]["compaction"]["lanes"]["R"]["body"], "fixed candidate")
        with self.assertRaises(ArchiveError):
            self.archive.summary("candidate", view)
        summary["body"] = "different candidate"
        encoded = encode_record(summary)
        self.archive._connection.execute("UPDATE objects SET payload=?,sha256=? WHERE kind='summary' AND id='candidate'",
                                         (encoded, hashlib.sha256(encoded).hexdigest()))
        self.archive._connection.commit()
        with self.assertRaises(ArchiveError):
            self.archive.read_execution("run")

    def test_compaction_source_must_match_the_fixed_business_position(self):
        state, view, _ = self.compact_candidate()
        state["compaction"]["snapshot"]["input"] = [{"role": "user", "content": "changed source"}]
        with self.assertRaises(ArchiveError):
            self.archive.prepare_execution("run", state, view, "validate_candidate", "p1", point_id="p2")


if __name__ == "__main__":
    unittest.main()
