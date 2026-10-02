"""Offline v2 configuration, scoped tracing and technical gate contracts."""
from copy import deepcopy
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from benchmarks.core import configuration, runner
from benchmarks.core.budgets import RequestBudget, WorkerAbort
from benchmarks.core.contracts import known, not_applicable
from benchmarks.core.executors import agent
from benchmarks.core.files import ROOT, sha256_file
from benchmarks.core.diagnostics import ConfigurationError
from benchmarks.core.registry import load_suite
from benchmarks.tests.test_research_batch import MODEL, SUITE


V2 = {"schema_version": "2", "enabled": True, "agent_source": "project_env",
      "source_repository": "fixture-source", "model_overrides": {}, "units": ["fixture"],
      "repeats": 1, "gate_first_sample": True,
      "limits": {"requests": 200, "execution_seconds": 5400, "sample_seconds": 1200}}


class ResearchConfigurationTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory(prefix="research-config-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.env_text = (
            "BASE_URL=https://model.invalid/v1\nMODEL_NAME=file-model\nAPI_KEY=file-secret-marker\n"
            "MAX_MODEL_CALLS=25\nLANGSMITH_TRACING=true\nLANGSMITH_API_KEY=trace-secret-marker\n"
            "LANGSMITH_ENDPOINT=https://trace.invalid\nLANGSMITH_PROJECT=fixture-project\n"
        )
        (self.root / ".env").write_text(self.env_text, encoding="utf-8")

    def test_source_precedence_snapshot_and_environment_restoration(self):
        config = deepcopy(V2)
        config["model_overrides"] = {"max_model_calls": 9}
        initial = {"MODEL_NAME": "environment-model", "API_KEY": "environment-secret-marker"}
        with patch.dict(os.environ, initial, clear=True):
            safe, connection = configuration.resolve_run_config(config, self.root)
            self.assertEqual(dict(os.environ), initial)
        self.assertEqual(safe["model"]["model_name"], "environment-model")
        self.assertEqual(safe["model"]["max_model_calls"], 9)
        self.assertEqual(connection["API_KEY"], "environment-secret-marker")
        self.assertEqual(safe["field_sources"]["model_name"], "environment")
        self.assertEqual(safe["field_sources"]["base_url"], "project_env")
        self.assertEqual(safe["field_sources"]["timeout_seconds"], "default")
        self.assertEqual(safe["field_sources"]["max_model_calls"], "model_override")
        self.assertEqual(safe["source_snapshot"]["settings"]["max_model_calls"], 25)
        self.assertEqual(safe["source_snapshot_sha256"], configuration.configuration_hash(safe["source_snapshot"]))
        serialized = json.dumps(safe)
        self.assertNotIn("secret-marker", serialized)
        self.assertNotIn("API_KEY", serialized)
        self.assertNotIn("LANGSMITH_API_KEY", connection)
        self.assertFalse(safe["tracing"]["enabled"])

    def test_retired_run_configuration_is_rejected_and_judge_still_uses_environment(self):
        with patch.object(configuration, "load_settings", side_effect=AssertionError("must not load .env")):
            with self.assertRaises(ConfigurationError) as error:
                configuration.resolve_run_config({**V2, "schema_version": "1"}, self.root, {})
            judge = configuration.load_connection({"model": MODEL, "credential_env": "JUDGE_KEY"},
                                                  {"JUDGE_KEY": "judge-secret"})
        self.assertEqual(error.exception.diagnostic["path"], "schema_version")
        self.assertEqual(judge["API_KEY"], "judge-secret")

    def test_real_run_example_matches_registered_tasks_and_validates_without_connections(self):
        example = json.loads((ROOT / "benchmarks/entrypoints/run.example.json").read_text(encoding="utf-8"))
        suite = load_suite("wechat-reporting")
        self.assertIs(example["enabled"], False)
        self.assertIs(example["tracing"]["enabled"], False)
        with patch.object(configuration, "load_settings", side_effect=AssertionError("no connection lookup")):
            checked = configuration.checked_config({**example, "enabled": True})
            planned = agent.plan(suite, checked)
        self.assertEqual([task["definition_unit_id"] for task in planned], example["units"] * example["repeats"])
        self.assertTrue(set(example["tracing"]["unit_ids"]) <= set(example["units"]))

    def test_field_diagnostics_never_include_supplied_values_or_unknown_keys(self):
        marker = "private-marker-DO-NOT-PRINT"
        cases = [({**V2, "enabled": False}, "enabled"),
                 ({**V2, "limits": {**V2["limits"], "requests": 0}}, "limits.requests"),
                 ({**V2, "limits": {**V2["limits"], "requests": 10 ** 400}}, "limits.requests"),
                 ({**V2, "repeats": True}, "repeats"),
                 ({**V2, "units": [marker, marker]}, "units[1]"),
                 ({**V2, "model_overrides": {marker: marker}}, "model_overrides"),
                 ({**V2, "tracing": {"enabled": True, "unit_ids": [marker]}}, "tracing.unit_ids[0]")]
        for config, field in cases:
            with self.subTest(field=field), self.assertRaises(ConfigurationError) as error:
                configuration.checked_config(config)
            self.assertEqual(error.exception.diagnostic["path"], field)
            self.assertEqual(set(error.exception.diagnostic), {"path", "code", "constraint", "hint"})
            self.assertNotIn(marker, str(error.exception))
        with self.assertRaises(ConfigurationError) as error:
            agent.plan(SUITE, {**V2, "units": [marker]})
        self.assertEqual(error.exception.diagnostic["code"], "unknown_unit_id")
        self.assertNotIn(marker, str(error.exception))

    def test_blocked_results_keep_actionable_diagnostics_without_network(self):
        for changed, field in (({**V2, "enabled": False}, "enabled"),
                               ({**V2, "limits": {**V2["limits"], "requests": 0}}, "limits.requests"),
                               ({**V2, "schema_version": "1"}, "schema_version")):
            with self.subTest(field=field), patch.object(runner, "collect_baseline", return_value={}), \
                    patch.object(agent, "execute_sample") as execute:
                result, _ = runner.run_benchmark("second-project", config=changed, root=self.root,
                                                 registry={"second-project": lambda: SUITE})
            execute.assert_not_called()
            self.assertTrue(all(unit["execution_status"] == "blocked" for unit in result["units"]))
            self.assertEqual(result["units"][0]["issues"][0]["diagnostic"]["path"], field)

    def test_model_and_credential_diagnostics_identify_safe_fields(self):
        marker = "private-model-value-do-not-print"
        with self.assertRaises(ConfigurationError) as error:
            configuration.checked_model({**MODEL, "timeout_seconds": marker}, path="model_overrides")
        self.assertEqual(error.exception.diagnostic["path"], "model_overrides.timeout_seconds")
        self.assertNotIn(marker, str(error.exception))
        with self.assertRaises(ConfigurationError) as error:
            configuration.checked_model({**MODEL, "base_url": f"https://{marker}@model.invalid"})
        self.assertEqual(error.exception.diagnostic["path"], "model.base_url")
        self.assertNotIn(marker, str(error.exception))
        (self.root / ".env").write_text(self.env_text + f"TIMEOUT_SECONDS={marker}\n", encoding="utf-8")
        with self.assertRaises(ConfigurationError) as error:
            configuration.resolve_run_config(V2, self.root, {})
        self.assertEqual(error.exception.diagnostic["path"], "project_env.timeout_seconds")
        self.assertNotIn(marker, str(error.exception))
        with self.assertRaises(ConfigurationError) as error:
            configuration.load_connection({"model": MODEL, "credential_env": "JUDGE_KEY"}, {})
        self.assertEqual(error.exception.diagnostic["path"], "credential_env")
        self.assertIn("不会自动加载 .env", str(error.exception))

    def test_trace_requires_selected_opt_in_and_rejects_sdk_overrides(self):
        config = {**deepcopy(V2), "tracing": {"enabled": True, "unit_ids": ["fixture"]}}
        safe, connection = configuration.resolve_run_config(config, self.root, {})
        self.assertTrue(safe["tracing"]["enabled"])
        self.assertEqual(safe["tracing"]["endpoint"], "https://trace.invalid")
        self.assertEqual(safe["tracing"]["field_sources"], {"enabled": "benchmark", "unit_ids": "benchmark",
                                                          "endpoint": "project_env", "project": "project_env",
                                                          "workspace_id": "default"})
        self.assertEqual(safe["source_snapshot"]["tracing"], safe["tracing"])
        self.assertEqual(connection["LANGSMITH_API_KEY"], "trace-secret-marker")
        with patch.dict(os.environ, {"LANGCHAIN_TRACING_V2": "true", "UNRELATED_SECRET": "private"}, clear=True):
            child = agent.research_environment(connection)
        self.assertEqual(child["LANGSMITH_TRACING"], "true")
        self.assertEqual(child["LANGCHAIN_TRACING_V2"], "false")
        self.assertNotIn("UNRELATED_SECRET", child)
        disabled = agent.connection_for(connection, {"enabled": False})
        self.assertEqual(set(disabled), configuration.CONNECTION_KEYS)
        for name in ("LANGSMITH_RUNS_ENDPOINTS", "LANGCHAIN_FAILED_TRACES_DIR", "LANGSMITH_USE_PYO3_CLIENT"):
            initial = {name: ""}
            with self.subTest(name=name), patch.dict(os.environ, initial, clear=True):
                with self.assertRaises(ValueError):
                    configuration.resolve_run_config(config, self.root)
                self.assertEqual(dict(os.environ), initial)
        for traced in (["unknown"], ["fixture", "fixture"], []):
            with self.subTest(traced=traced), self.assertRaises(ValueError):
                configuration.checked_config({**config, "tracing": {"enabled": True, "unit_ids": traced}})

    def test_invalid_source_and_overrides_restore_environment_and_block(self):
        for extra in ("TIMEOUT_SECONDS=invalid\n", "CONTEXT_TRIGGER_PERCENT=90\n"):
            (self.root / ".env").write_text(self.env_text + extra, encoding="utf-8")
            with self.subTest(extra=extra), patch.dict(os.environ, {}, clear=True):
                with self.assertRaises(ValueError):
                    configuration.resolve_run_config(V2, self.root)
                self.assertEqual(dict(os.environ), {})
        for overrides in ({"API_KEY": "private"}, {"log_dir": "elsewhere"}, {"project_root": "elsewhere"}):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                configuration.checked_config({**V2, "model_overrides": overrides})
        (self.root / ".env").write_text(self.env_text, encoding="utf-8")
        with patch.dict(os.environ, {}, clear=True), patch.object(runner, "collect_baseline", return_value={}), \
                patch.object(agent, "atomic_write_json", side_effect=OSError("injected configuration save failure")), \
                patch.object(agent, "prepare_materials") as materials, patch.object(agent, "execute_sample") as execute:
            result, _ = runner.run_benchmark("second-project", config=V2, root=self.root,
                                             registry={"second-project": lambda: SUITE})
        materials.assert_not_called()
        execute.assert_not_called()
        self.assertTrue(all(unit["execution_status"] == "blocked" for unit in result["units"]))
        self.assertTrue(all(any(item["code"] == "configuration_recording_unavailable" for item in unit["issues"])
                            for unit in result["units"]))
        self.assertFalse(any(item["kind"] == "resolved_configuration" for unit in result["units"] for item in unit["evidence"]))

    def test_ordered_selection_and_explicit_unselected_scope(self):
        suite = deepcopy(SUITE)
        suite["units"] = [{**deepcopy(suite["units"][0]), "unit_id": name} for name in ("first", "second", "third")]
        config = {**V2, "units": ["third", "first"], "repeats": 2}
        planned = agent.plan(suite, config)
        self.assertEqual([task["unit_id"] for task in planned], ["third.r1", "first.r1", "third.r2", "first.r2"])
        for names in (["unknown"], ["first", "first"], []):
            with self.subTest(names=names), self.assertRaises(ValueError):
                agent.plan(suite, {**config, "units": names})
        with patch.object(agent, "prepare", side_effect=lambda suite, config, directory, result, **kw:
                          [unit.update(execution_status="blocked") for unit in result["units"]] and None), \
                patch.object(runner, "collect_baseline", return_value={}):
            result, _ = runner.run_benchmark("second-project", config=config, root=self.root,
                                             registry={"second-project": lambda: suite})
        self.assertEqual(result["unselected_units"], ["second"])

    def test_sample_deadline_reaches_1200_without_raising_batch_caps(self):
        safe = configuration.checked_config(V2)
        self.assertEqual(safe["limits"]["sample_seconds"], 1200)
        path = self.root / "requests.jsonl"
        path.touch()
        with patch("benchmarks.core.budgets.time.monotonic", return_value=100):
            budget = RequestBudget(path, "run", "case", remaining_seconds=1200)
        with patch("benchmarks.core.budgets.time.monotonic", return_value=401):
            budget.reserve(1, "business")
        with patch("benchmarks.core.budgets.time.monotonic", return_value=1300), self.assertRaises(WorkerAbort):
            budget.reserve(1, "summary")
        for name, value in (("requests", 401), ("execution_seconds", 7201), ("sample_seconds", 1201)):
            with self.subTest(name=name), self.assertRaises(ValueError):
                configuration.checked_config({**V2, "limits": {**V2["limits"], name: value}})

    def test_per_task_sources_scope_and_first_only_gate(self):
        result = self.run_fixture_batch(first_good=True)
        unit = result["units"][0]
        agent._accept_receipt(unit, {"run_id": unit["sample_run_id"], "started_at": result["started_at"],
                                    "ended_at": result["ended_at"], "execution_status": "completed",
                                    "evidence": deepcopy(unit["evidence"])}, failed=False)
        self.assertEqual(sum(item["kind"] == "resolved_configuration" for item in unit["evidence"]), 1)
        turn = unit["turns"][0]
        turn.update(session_id=known("session"), checkpoint_id=known("checkpoint"), checkpoint_complete=True)
        process_path = Path(next(item["path"] for item in turn["process_evidence"] if item["kind"] == "process"))
        process = json.loads(process_path.read_text(encoding="utf-8"))
        process.update(session_id="session", checkpoint_id=known("checkpoint"), checkpoint_complete=True)
        process_path.write_text(json.dumps(process), encoding="utf-8")
        self.assertTrue(agent.first_sample_completed(unit))
        for bad in (None, False):
            with self.subTest(checkpoint_complete=bad):
                turn["checkpoint_complete"] = bad
                self.assertFalse(agent.first_sample_completed(unit))
        turn["checkpoint_complete"] = True
        turn["checkpoint_id"] = {"availability": "unknown", "reason": "not confirmed"}
        self.assertFalse(agent.first_sample_completed(unit))
        turn["checkpoint_id"] = known("checkpoint")
        process.pop("checkpoint_complete")
        process_path.write_text(json.dumps(process), encoding="utf-8")
        self.assertFalse(agent.first_sample_completed(unit))

    def test_failed_first_technical_gate_stops_without_changing_quality(self):
        self.run_fixture_batch(first_good=False)

    def run_fixture_batch(self, *, first_good):
        suite = deepcopy(SUITE)
        suite["units"] = [{**deepcopy(suite["units"][0]), "unit_id": name, "turns": ["one turn"],
                           "settings_overrides": {"context_budget_tokens": 65536}}
                          for name in ("first", "second", "third")]
        config = {**deepcopy(V2), "units": ["second", "first", "third"],
                  "model_overrides": {"context_budget_tokens": 100000},
                  "tracing": {"enabled": True, "unit_ids": ["second"]}}
        manifests, calls = {}, []

        def prepare(repo, source, path, definition):
            manifest = {"material_root": str(path / "material"), "files": [], "material_version": "fixture"}
            manifests[str(path)] = manifest
            return manifest

        def execute(unit, spec, directory, connection, **kwargs):
            frozen = json.loads((directory.parent / "resolved-configuration.json").read_text(encoding="utf-8"))
            self.assertEqual(frozen["run_id"], spec["batch_id"])
            self.assertNotIn("secret-marker", json.dumps(frozen))
            recorded = next(item for item in frozen["units"] if item["unit_id"] == unit["unit_id"])
            self.assertEqual(recorded["configuration"]["settings"], spec["profile"])
            self.assertTrue(any(item["kind"] == "resolved_configuration" and item["availability"] == "available"
                                for item in unit["evidence"]))
            calls.append((deepcopy(spec), deepcopy(connection), kwargs["timeout"]))
            good = first_good if len(calls) == 1 else len(calls) != 2
            outcome = "completed" if good else "error"
            unit.update(execution_status="completed", agent_outcome=known(outcome))
            turn = unit["turns"][0]
            run_id = spec["run_id"]
            answer = directory / "answer.md"
            answer.write_text("Synthetic output; not a semantic quality judgment.", encoding="utf-8")
            process, log = directory / "process.json", directory / "agent.jsonl"
            process.write_text(json.dumps({"run_id": run_id, "log_complete": True}), encoding="utf-8")
            log.write_text(json.dumps({"run_id": run_id, "event": "run.ended"}) + "\n", encoding="utf-8")
            turn.update(execution_status="completed", run_id=known(run_id), agent_outcome=known(outcome),
                        session_id=not_applicable("independent run"), checkpoint_id=not_applicable("independent run"),
                        answer_sha256=sha256_file(answer), answer_evidence={"kind": "answer", "path": str(answer), "availability": "available"},
                        process_evidence=[{"kind": kind, "path": str(path), "availability": "available"}
                                          for kind, path in (("process", process), ("agent_log", log))],
                        tracing={"requested": spec["tracing"]["enabled"], "traces": []})
            return False

        with patch.dict(os.environ, {}, clear=True), patch.object(runner, "collect_baseline", return_value={}), \
                patch.object(agent, "prepare_materials", side_effect=prepare), \
                patch.object(agent, "instantiate_materials", side_effect=lambda prepared, path, task: prepare(None, None, path, None)), \
                patch.object(agent, "verify_materials", side_effect=lambda path: manifests[str(path)]), \
                patch("benchmarks.core.materials.verify_materials", side_effect=lambda path: manifests[str(path)]), \
                patch.object(agent, "execute_sample", side_effect=execute), \
                patch("benchmarks.core.trace_evidence.verify_sample_traces") as trace_query:
            result, _ = runner.run_benchmark("second-project", config=config, root=self.root,
                                             registry={"second-project": lambda: suite})
        self.assertEqual(len(calls), 3 if first_good else 1)
        self.assertEqual(result["batch"]["first_sample_gate"]["passed"], first_good)
        self.assertTrue(all(unit["verdict"] == "pending_review" for unit in result["units"]))
        self.assertEqual(trace_query.call_count, 1)
        self.assertGreater(trace_query.call_args.kwargs["timeout_seconds"], 0)
        self.assertLessEqual(trace_query.call_args.kwargs["timeout_seconds"], 30)
        for index, (spec, connection, timeout) in enumerate(calls):
            self.assertEqual(timeout, 1200)
            self.assertEqual(spec["profile"]["context_budget_tokens"], 65536)
            self.assertEqual("LANGSMITH_API_KEY" in connection, index == 0)
            self.assertEqual(spec["tracing"]["enabled"], index == 0)
        details = result["units"][0]["conditions"]["configuration"]["value"]
        self.assertEqual(details["field_sources"]["context_budget_tokens"], "task_override")
        self.assertEqual(details["effective_settings_sha256"], configuration.configuration_hash(details["settings"]))
        self.assertEqual(details["paths"]["source"], "benchmark")
        self.assertNotIn("secret-marker", json.dumps(result))
        self.assertNotIn("secret-marker", json.dumps([item[0] for item in calls]))
        return result


if __name__ == "__main__":
    unittest.main()
