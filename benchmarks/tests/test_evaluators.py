"""Offline evaluator conformance; synthetic judgments do not establish quality."""
import asyncio
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from benchmarks.core.contracts import known
from benchmarks.core.diagnostics import ConfigurationError
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
        rule = self.rules['cases']['fixture']
        rule['conditions'][0]['judge_scope'] = 'model_suggestion'
        rule['severe_errors'][0]['judge_scope'] = 'human_only'
        rule['mechanism_review'][0]['judge_scope'] = 'human_only'
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

    def candidate(self):
        return {
            'unit_id': self.fixture.result['units'][0]['unit_id'],
            'conditions': [{'id': 'factual', 'verdict': 'passed', 'basis': 'Fixture source supports this answer.',
                            'evidence_refs': ['answer:turn:1:1', 'source:code/README.md#facts'],
                            'evidence_support': 'sufficient'}],
            'severe_errors': [], 'mechanism_review': [],
        }

    def save_rules(self):
        self.fixture.result['units'][0]['conditions']['rubric'] = known(self.rules)
        atomic_write_json(self.fixture.path, self.fixture.result)

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
        outcome, requests = self.run_judge(json.dumps(self.candidate()))
        self.assertEqual(outcome['exit_code'], 3)
        self.assertEqual(self.fixture.path.read_bytes(), original)
        self.assertEqual(len(requests), 1)
        self.assertIn('untrusted data', requests[0]['instructions'])
        facts = json.loads(outcome['saved']['judge'].read_text(encoding='utf-8'))
        self.assertEqual(facts['requests'][0]['usage']['total_tokens'], 13)
        self.assertEqual(facts['policy_version'], rubric.POLICY_VERSION)
        proposal = json.loads(outcome['saved']['proposal'].read_text(encoding='utf-8'))
        self.assertEqual(proposal['reviewer'], '')
        self.assertEqual(proposal['binding']['run_id'], self.fixture.result['run_id'])
        assessment = proposal['assessments'][0]
        self.assertEqual(assessment['conditions'][0]['verdict'], 'passed')
        self.assertEqual(assessment['mechanism_review'][0], {
            'id': 'no_recent_leak', 'verdict': 'pending_review', 'basis': '',
            'evidence_refs': [], 'evidence_support': 'pending'})
        self.assertEqual(assessment['severe_errors'][0], {
            'id': 'invented', 'present': None, 'basis': '', 'evidence_refs': []})
        self.assertNotIn('JUDGE_SECRET_MARKER', ''.join(p.read_text(encoding='utf-8') for p in outcome['saved'].values()))

    def test_malformed_judgment_retains_usage_without_retry(self):
        outcome, requests = self.run_judge('not valid JSON')
        self.assertEqual(outcome['exit_code'], 2)
        self.assertEqual(len(requests), 1)
        facts = json.loads(outcome['saved']['judge'].read_text(encoding='utf-8'))
        self.assertEqual(facts['requests'][0]['status'], 'failed')
        self.assertEqual(facts['requests'][0]['usage']['input_tokens'], 10)

    def test_llm_insufficient_support_cannot_pass(self):
        candidate = self.candidate()
        candidate['conditions'][0]['evidence_support'] = 'insufficient'
        outcome, _ = self.run_judge(json.dumps(candidate))
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

    def test_payload_contains_only_model_items_and_answer_source_references(self):
        outcome, requests = self.run_judge(json.dumps(self.candidate()))
        self.assertEqual(outcome['exit_code'], 3)
        payload = json.loads(requests[0]['input'][0]['content'])
        self.assertEqual([item['id'] for item in payload['rubric']['conditions']], ['factual'])
        self.assertEqual(payload['rubric']['severe_errors'], [])
        self.assertEqual(payload['rubric']['mechanism_review'], [])
        self.assertEqual(set(payload['observed_evidence']), {'answer:turn:1:1', 'source:code/README.md#facts'})
        self.assertEqual(payload['assessment']['mechanism_review'], [])
        self.assertNotIn('coverage:summary_used', payload['observed_evidence'])

    def test_human_only_scope_saves_pending_without_credentials_or_client(self):
        self.rules['cases']['fixture']['conditions'][0]['judge_scope'] = 'human_only'
        self.save_rules()
        factory = unittest.mock.Mock()
        with patch.dict('os.environ', {}, clear=True), patch.object(rubric, 'load_connection') as connection:
            outcome = asyncio.run(rubric._grade(self.directory, self.config, client_factory=factory))
        connection.assert_not_called()
        factory.assert_not_called()
        self.assertEqual(outcome['exit_code'], 3)
        facts = json.loads(outcome['saved']['judge'].read_text(encoding='utf-8'))
        self.assertEqual(facts['requests'], [])
        self.assertEqual(facts['issues'], [])
        self.assertEqual(facts['skipped_units'], [{'unit_id': self.candidate()['unit_id'], 'reason': 'human_only'}])
        proposal = json.loads(outcome['saved']['proposal'].read_text(encoding='utf-8'))
        self.assertEqual(proposal['assessments'][0]['conditions'][0]['verdict'], 'pending_review')

    def test_missing_or_invalid_scope_is_rejected_before_connection(self):
        for value in (None, 'UNSAFE_SCOPE_MARKER', [], {}):
            with self.subTest(value=value):
                item = self.rules['cases']['fixture']['conditions'][0]
                item.pop('judge_scope', None)
                if value is not None:
                    item['judge_scope'] = value
                self.save_rules()
                factory = unittest.mock.Mock()
                with patch.object(rubric, 'load_connection') as connection, self.assertRaises(ConfigurationError) as caught:
                    asyncio.run(rubric._grade(self.directory, self.config, client_factory=factory))
                connection.assert_not_called()
                factory.assert_not_called()
                self.assertEqual(caught.exception.diagnostic['path'], 'rubric.cases[0].conditions[0].judge_scope')
                self.assertNotIn('UNSAFE_SCOPE_MARKER', str(caught.exception))
        self.assertFalse((self.directory / 'judge-proposals').exists())

    def test_mechanism_cannot_opt_into_model_scope(self):
        self.rules['cases']['fixture']['mechanism_review'][0]['judge_scope'] = 'model_suggestion'
        self.save_rules()
        factory = unittest.mock.Mock()
        with patch.object(rubric, 'load_connection') as connection, self.assertRaises(ConfigurationError):
            asyncio.run(rubric._grade(self.directory, self.config, client_factory=factory))
        connection.assert_not_called()
        factory.assert_not_called()

    def test_reviewed_result_is_rejected_before_connection(self):
        from benchmarks.evaluators.review import _template, apply_review
        review = _template(self.fixture.result, self.fixture.path, self.rules)
        review.update(reviewer='test human', reviewed_at='2026-09-27T01:00:00+00:00')
        derived = apply_review(self.fixture.result, review, self.fixture.path, rubric=self.rules)
        atomic_write_json(self.fixture.path, derived)
        factory = unittest.mock.Mock()
        with patch.object(rubric, 'load_connection') as connection, self.assertRaises(ConfigurationError) as caught:
            asyncio.run(rubric._grade(self.directory, self.config, client_factory=factory))
        self.assertEqual(caught.exception.diagnostic['code'], 'already_reviewed')
        connection.assert_not_called()
        factory.assert_not_called()

    def test_invalid_result_is_rejected_before_connection(self):
        self.fixture.result['summary'] = {}
        atomic_write_json(self.fixture.path, self.fixture.result)
        factory = unittest.mock.Mock()
        with patch.object(rubric, 'load_connection') as connection, self.assertRaises(ConfigurationError) as caught:
            asyncio.run(rubric._grade(self.directory, self.config, client_factory=factory))
        self.assertEqual(caught.exception.diagnostic['code'], 'invalid_result')
        connection.assert_not_called()
        factory.assert_not_called()

    def test_judge_config_has_safe_field_diagnostics(self):
        cases = [('enabled', False), ('max_requests', 0), ('max_input_bytes', 0),
                 ('execution_seconds', 0), ('credential_env', 'UNSAFE=SECRET')]
        for field, value in cases:
            with self.subTest(field=field), self.assertRaises(ConfigurationError) as caught:
                rubric.checked_judge_config({**self.config, field: value})
            self.assertEqual(caught.exception.diagnostic['path'], field)
            self.assertNotIn('UNSAFE=SECRET', str(caught.exception))

    def test_real_judge_example_is_disabled_and_valid_when_explicitly_enabled(self):
        from benchmarks.core.files import ROOT
        config = json.loads((ROOT / 'benchmarks/entrypoints/judge.example.json').read_text(encoding='utf-8'))
        factory = unittest.mock.Mock()
        with self.assertRaises(ConfigurationError) as caught:
            asyncio.run(rubric._grade(self.directory, config, client_factory=factory))
        self.assertEqual(caught.exception.diagnostic['path'], 'enabled')
        factory.assert_not_called()
        checked = rubric.checked_judge_config({**config, 'enabled': True})
        self.assertEqual(checked['credential_env'], 'BENCHMARK_JUDGE_KEY')
        self.assertEqual(checked['max_requests'], config['max_requests'])

    def test_out_of_scope_response_keeps_usage_and_does_not_promote_manual_items(self):
        mutations = (
            lambda value: value['mechanism_review'].append({
                'id': 'no_recent_leak', 'verdict': 'passed', 'basis': 'unsupported mechanism claim',
                'evidence_refs': ['answer:turn:1:1'], 'evidence_support': 'sufficient'}),
            lambda value: value['severe_errors'].append({
                'id': 'invented', 'present': False, 'basis': 'unsupported process claim',
                'evidence_refs': ['answer:turn:1:1']}),
            lambda value: value['conditions'][0].update(evidence_refs=['execution:turn:1']),
            lambda value: value['conditions'][0].update(evidence_refs=['coverage:summary_used']),
        )
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                candidate = self.candidate()
                mutate(candidate)
                outcome, calls = self.run_judge(json.dumps(candidate))
                self.assertEqual(len(calls), 1)
                self.assertEqual(outcome['exit_code'], 2)
                facts = json.loads(outcome['saved']['judge'].read_text(encoding='utf-8'))
                self.assertEqual(facts['requests'][0]['status'], 'failed')
                self.assertEqual(facts['requests'][0]['usage']['total_tokens'], 13)
                proposal = json.loads(outcome['saved']['proposal'].read_text(encoding='utf-8'))
                assessment = proposal['assessments'][0]
                self.assertEqual(assessment['conditions'][0]['verdict'], 'pending_review')
                self.assertEqual(assessment['mechanism_review'][0]['verdict'], 'pending_review')
                self.assertIsNone(assessment['severe_errors'][0]['present'])


class JudgeScopeDefinitionTests(unittest.TestCase):
    def test_real_rubric_explicitly_separates_answer_and_process_judgments(self):
        from benchmarks.suites.wechat_reporting.suite import load_rubric
        rules = load_rubric()
        counts = {'conditions': [0, 0], 'severe_errors': [0, 0], 'mechanism_review': [0, 0]}
        for case_id, rule in rules['cases'].items():
            for group in counts:
                for item in rule.get(group, []):
                    scope = item['judge_scope']
                    self.assertIn(scope, {'model_suggestion', 'human_only'})
                    counts[group][scope == 'human_only'] += 1
                    if group == 'conditions':
                        manual = item['id'] in {f'{case_id}-evidence', f'{case_id}-execution', 'read-limit-1'}
                    elif group == 'severe_errors':
                        manual = item['id'] != 'unsupported-certainty'
                    else:
                        manual = True
                    self.assertEqual(scope, 'human_only' if manual else 'model_suggestion', item['id'])
        self.assertEqual(counts, {'conditions': [37, 19], 'severe_errors': [9, 18], 'mechanism_review': [0, 2]})

    def test_scope_metadata_does_not_change_existing_scoring_standards(self):
        from benchmarks.suites.wechat_reporting.suite import HERE, load_rubric
        archived = json.loads((HERE / 'history/rubric-v2.json').read_text(encoding='utf-8'))
        current = load_rubric()
        self.assertEqual(archived['version'], 'wechat-rubric-v2')
        self.assertEqual(current['version'], 'wechat-rubric-v3')
        current['version'] = archived['version']
        for rule in current['cases'].values():
            for group in rubric.ASSESSMENT_GROUPS:
                for item in rule.get(group, []):
                    item.pop('judge_scope')
        self.assertEqual(current, archived)
