"""Public selection and runtime boundaries, independent of current project cases."""
from copy import deepcopy
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from benchmarks.core import registry
from benchmarks.core.contracts import validate_result
from benchmarks.core.files import ROOT
from benchmarks.core.processes import offline_network, safe_environment
from benchmarks.core.executors import control, control_worker
from benchmarks.suites.agent_contracts.suite import load_catalog
from benchmarks.reporting.render import render_report


class ArchitectureTests(unittest.TestCase):
    def test_registry_rejects_unknown_duplicate_or_mismatched_suites(self):
        for value in ({'suite_id': 'wrong'}, {'suite_id': 'fixture', 'executor': 'agent', 'version': '1',
                                            'units': [{'unit_id': 'same'}, {'unit_id': 'same'}]}):
            with self.assertRaises(ValueError):
                registry.load_suite('fixture', registry={'fixture': lambda: value})
        with self.assertRaises(ValueError):
            registry.load_suite('unknown')

    def test_shared_framework_has_no_project_import_or_case_routing(self):
        for folder in ('core', 'entrypoints', 'evaluators', 'reporting'):
            for path in (ROOT / 'benchmarks' / folder).rglob('*.py'):
                if path.name == 'registry.py':
                    continue  # explicit data registry is the only project binding
                source = path.read_text(encoding='utf-8')
                self.assertNotIn('wechat', source.lower(), str(path))
                self.assertNotIn('scripts/check_', source, str(path))
        self.assertFalse(list((ROOT / 'scripts').glob('check_*.py')))

    def test_offline_guard_rejects_external_socket_before_connect(self):
        with offline_network(), socket.socket() as connection:
            with self.assertRaisesRegex(AssertionError, 'external network'):
                connection.connect(('198.51.100.1', 443))
        # Asyncio's local socket pair remains possible on Windows.
        with offline_network():
            one, two = socket.socketpair()
            one.close()
            two.close()

    def test_missing_entry_and_failed_start_are_not_assertion_verdicts(self):
        definition = deepcopy(load_catalog()[0]['execution.react'])
        with TemporaryDirectory() as temp:
            root = Path(temp)
            unit = control.make_unit(definition, root)
            self.assertFalse(control.execute_unit(unit, definition, 'missing', root, root=root))
            self.assertEqual(unit['execution_status'], 'blocked')
            path = root / definition['script']
            path.parent.mkdir(parents=True)
            path.write_text('# launch fixture', encoding='utf-8')
            unit = control.make_unit(definition, root)
            with patch.object(control.subprocess, 'Popen', side_effect=OSError('launch failure')):
                control.execute_unit(unit, definition, 'failed-start', root, root=root)
            self.assertEqual(unit['execution_status'], 'failed')
            self.assertEqual(unit['verdict'], 'pending_review')

    def test_optimized_worker_refuses_before_any_group(self):
        with TemporaryDirectory() as temp:
            receipt = Path(temp) / 'receipt.json'
            process = subprocess.run([sys.executable, '-O', '-m', 'benchmarks.core.executors.control_worker',
                                      '--run-id', 'optimized-fixture', '--unit-id', 'execution.react', '--receipt', str(receipt)],
                                     cwd=ROOT, env=safe_environment(dict(os.environ)), capture_output=True, timeout=20)
            self.assertEqual(process.returncode, 2, process.stderr)
            data = json.loads(receipt.read_text(encoding='utf-8'))
            self.assertEqual(data['outcome'], 'error')
            self.assertEqual(data['events_count'], 0)

    def test_foreign_check_module_cannot_run(self):
        definition = load_catalog()[0]['execution.react']
        fake = SimpleNamespace(__file__='C:/outside/react.py', execute=unittest.mock.Mock())
        with patch.object(control_worker, 'import_module', return_value=fake), self.assertRaises(ImportError):
            control_worker._execute_entry(definition)
        fake.execute.assert_not_called()

    def test_current_real_model_report_does_not_invent_control_groups(self):
        from benchmarks.tests.test_research_review import ReviewTest
        fixture = ReviewTest()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        result = deepcopy(fixture.result)
        validate_result(result)
        rendered = render_report(result)
        self.assertIn('pending_review', rendered)
        self.assertNotIn('groups', result['units'][0])
