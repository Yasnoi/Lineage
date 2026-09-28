"""Offline evaluator conformance; synthetic judgments do not establish quality."""
import asyncio
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from benchmarks.core.contracts import known
from benchmarks.core.files import atomic_write_json
from benchmarks.core.evidence import usage_fields
from benchmarks.core.executors.agent_worker import usage_metrics
from benchmarks.evaluators import checklist, rubric


class EvaluatorTests(unittest.TestCase):
    def setUp(self):
        from benchmarks.tests.test_research_review import ReviewTest, RUBRIC
        self.fixture = ReviewTest()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.directory = self.fixture.directory
        self.rules = deepcopy(RUBRIC)
        self.fixture.result['units'][0]['conditions']['rubric'] = known(self.rules)
        from benchmarks.core.materials import prepare_materials
        with patch('benchmarks.core.materials._read_blob', return_value=(b'Synthetic source facts', 'fixture-blob')):
            manifest = prepare_materials(self.directory, self.directory, self.directory / 'materials', {
                'version': 'fixture-v1', 'sources': [{'prefix': 'code', 'commit': 'fixture', 'paths': ['README.md']}],
                'index': 'Synthetic provenance',
            })
        self.fixture.result['units'][0]['conditions']['materials'] = known(manifest)
        atomic_write_json(self.fixture.path, self.fixture.result)
        self.config = {
            'enabled': True, 'model': {'base_url': 'https://judge.invalid', 'model_name': 'judge-fixture'},
            'credential_env': 'JUDGE_FIXTURE_KEY', 'max_requests': 1,
            'max_input_bytes': 50000, 'execution_seconds': 10,
        }

    def run_judge(self, value, *, config=None):
        seen = []
        directory = self.directory

        class Client:
            def __init__(self, **kwargs):
                self.responses = SimpleNamespace(create=self.create)
                self.options = kwargs

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_):
                pass

            async def create(self, **kwargs):
                journal = next(directory.glob('judge-proposals/*/judge.json'))
                record = json.loads(journal.read_text(encoding='utf-8'))
                assert len(record['requests']) == 1  # reservation precedes real transport seam
                seen.append(kwargs)
                if isinstance(value, Exception):
                    raise value
                return SimpleNamespace(output_text=value, usage={'input_tokens': 10, 'output_tokens': 3, 'total_tokens': 13})

        with patch.dict('os.environ', {'JUDGE_FIXTURE_KEY': 'JUDGE_SECRET_MARKER'}):
            outcome = asyncio.run(rubric._grade(directory, config or self.config, client_factory=Client))
        return outcome, seen

    def test_checklist_missing_na_and_severe_error(self):
        self.assertEqual(checklist.aggregate([]), 'pending_review')
        self.assertEqual(checklist.aggregate([{'verdict': 'passed'}, {'verdict': 'failed', 'applicable': False}]), 'passed')
        self.assertEqual(checklist.aggregate([{'verdict': 'passed'}], [{'present': None}]), 'pending_review')
        self.assertEqual(checklist.aggregate([{'verdict': 'passed'}], [{'present': True}]), 'failed')

    def test_cached_usage_missing_not_zero_or_hit_rate(self):
        self.assertIsNone(usage_fields({'usage': {'input_tokens': 10}})['cached_input_tokens'])
        self.assertEqual(usage_fields({'usage': {'input_tokens_details': {'cached_tokens': 4}}})['cached_input_tokens'], 4)
        observer = SimpleNamespace(request_events=[{'purpose': 'business'}], request_usage=[
            {'input_tokens': 10, 'output_tokens': 2, 'total_tokens': 12, 'cached_input_tokens': 4}], request_durations=[0.25])
        metrics = usage_metrics(observer)
        self.assertEqual(metrics['cache_input_fraction']['value'], 0.4)
        self.assertEqual(metrics['mean_request_seconds']['value'], 0.25)
        observer.request_events.append({'purpose': 'summary'})
        metrics = usage_metrics(observer)
        self.assertEqual(metrics['cache_input_fraction']['availability'], 'unknown')
        self.assertEqual(metrics['total_tokens']['value'], 12)
        self.assertEqual(metrics['total_tokens_missing_requests']['value'], 1)

    def test_llm_proposal_is_bound_and_never_changes_execution_verdict(self):
        original = self.fixture.path.read_bytes()
        review = self.fixture.completed_review()
        outcome, requests = self.run_judge(json.dumps(review['assessments'][0]))
        self.assertEqual(outcome['exit_code'], 3)
        self.assertEqual(self.fixture.path.read_bytes(), original)
        self.assertEqual(len(requests), 1)
        self.assertIn('untrusted data', requests[0]['instructions'])
        facts = json.loads(outcome['saved']['judge'].read_text(encoding='utf-8'))
        self.assertEqual(facts['requests'][0]['usage']['total_tokens'], 13)
        proposal = json.loads(outcome['saved']['proposal'].read_text(encoding='utf-8'))
        self.assertEqual(proposal['reviewer'], '')
        self.assertEqual(proposal['binding']['run_id'], self.fixture.result['run_id'])
        self.assertNotIn('JUDGE_SECRET_MARKER', ''.join(p.read_text(encoding='utf-8') for p in outcome['saved'].values()))

    def test_malformed_judgment_retains_usage_without_retry(self):
        outcome, requests = self.run_judge('not valid JSON')
        self.assertEqual(outcome['exit_code'], 2)
        self.assertEqual(len(requests), 1)
        facts = json.loads(outcome['saved']['judge'].read_text(encoding='utf-8'))
        self.assertEqual(facts['requests'][0]['status'], 'failed')
        self.assertEqual(facts['requests'][0]['usage']['input_tokens'], 10)

    def test_llm_insufficient_support_cannot_pass(self):
        review = self.fixture.completed_review()
        review['assessments'][0]['conditions'][0]['evidence_support'] = 'insufficient'
        outcome, _ = self.run_judge(json.dumps(review['assessments'][0]))
        self.assertEqual(outcome['exit_code'], 2)
        proposal = json.loads(outcome['saved']['proposal'].read_text(encoding='utf-8'))
        self.assertEqual(proposal['assessments'][0]['conditions'][0]['verdict'], 'pending_review')

    def test_input_limit_blocks_without_transport(self):
        outcome, requests = self.run_judge('{}', config={**self.config, 'max_input_bytes': 1})
        self.assertEqual(outcome['exit_code'], 2)
        self.assertEqual(requests, [])

    def test_disabled_judge_configuration_never_constructs_client(self):
        factory = unittest.mock.Mock()
        with self.assertRaises(ValueError):
            asyncio.run(rubric._grade(self.directory, {**self.config, 'enabled': False}, client_factory=factory))
        factory.assert_not_called()

    def test_unexecuted_sample_is_not_sent_to_judge(self):
        from benchmarks.core.contracts import summarize, unknown
        unit = self.fixture.result['units'][0]
        unit['execution_status'] = 'not_run'
        for turn in unit['turns']:
            turn.update(execution_status='not_run', run_id=unknown('not executed'), answer_evidence=[])
        self.fixture.result['summary'] = summarize([unit])
        atomic_write_json(self.fixture.path, self.fixture.result)
        outcome, requests = self.run_judge('{}')
        self.assertEqual(requests, [])
        self.assertEqual(outcome['exit_code'], 2)

    def test_judge_accounting_failure_stops_before_transport(self):
        write = rubric.atomic_write_json

        def fail(path, value):
            if path.name == 'judge.json' and value['requests']:
                raise OSError('ledger disk failure')
            return write(path, value)

        with patch.object(rubric, 'atomic_write_json', side_effect=fail), self.assertRaises(OSError):
            self.run_judge('{}')
        facts = json.loads(next(self.directory.glob('judge-proposals/*/judge.json')).read_text(encoding='utf-8'))
        self.assertEqual(facts['requests'], [])

    def test_historical_review_is_read_only(self):
        from benchmarks.evaluators.review import make_review_packet
        result = deepcopy(self.fixture.result)
        result['schema_version'] = '2'
        with self.assertRaises(ValueError):
            make_review_packet(result, self.directory, rubric=self.rules)
