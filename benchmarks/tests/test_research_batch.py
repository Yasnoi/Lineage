"""Dataset-independent dispatch with a second synthetic suite and real artifacts."""
from contextlib import redirect_stdout,redirect_stderr
from copy import deepcopy
import io,json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from benchmarks.entrypoints import cli
from benchmarks.core import runner
from benchmarks.core.executors import agent
from benchmarks.core.configuration import checked_config,checked_model,load_connection
from benchmarks.core.contracts import known,exit_code,validate_result
from benchmarks.core.files import sha256_file

CONFIG={'schema_version':'2','enabled':True,'source_repository':'fixture-source','agent_source':'project_env',
        'model_overrides':{},'units':['fixture'],'repeats':2,
        'limits':{'requests':6,'execution_seconds':30,'sample_seconds':5}}
MODEL={'base_url':'https://model.invalid','model_name':'test-model'}
ENV={'BASE_URL':MODEL['base_url'],'MODEL_NAME':MODEL['model_name'],'API_KEY':'test-secret'}
SUITE={'suite_id':'second-project','version':'fixture-v1','executor':'agent','network':True,'materials':{},
       'rubric':{'version':'fixture-v1','dimensions':[{'id':'facts','description':'facts'}],
                 'cases':{'fixture':{'conditions':[{'id':'facts','dimension':'facts','description':'source support'}],'severe_errors':[]}}},
       'units':[{'unit_id':'fixture','case_id':'fixture','version':'fixture-v1','turns':['first','follow up'],
                 'dimensions':['facts'],'mechanism':None,'settings_overrides':{}}]}

class ResearchDispatchTests(unittest.TestCase):
    def test_list_and_help_never_execute(self):
        with patch.object(cli,'run_benchmark') as run,redirect_stdout(io.StringIO()):
            self.assertEqual(cli.main([]),0);self.assertEqual(cli.main(['list']),0)
        run.assert_not_called()

    def test_unknown_or_retired_cli_never_executes(self):
        for args in (['run','unknown'],['--suite','control-all'],['run','wechat-reporting','--research-profile','wechat-v1']):
            with patch.object(cli,'run_benchmark') as run,redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:cli.main(args)
                self.assertEqual(error.exception.code,2);run.assert_not_called()

    def test_explicit_config_and_filtered_credentials(self):
        checked_config(CONFIG);env={'TEST_MODEL_KEY':'secret-marker','LANGSMITH_API_KEY':'tracing-marker'}
        connection=load_connection({'model':MODEL,'credential_env':'TEST_MODEL_KEY'},env)
        self.assertEqual(env['LANGSMITH_API_KEY'],'tracing-marker')
        with patch.dict('os.environ',{'LANGSMITH_API_KEY':'tracing-marker','PYTHONOPTIMIZE':'2'}):child=agent.research_environment(connection)
        self.assertEqual(child['LANGSMITH_TRACING'],'false');self.assertNotIn('PYTHONOPTIMIZE',child)
        self.assertNotIn('tracing-marker',json.dumps(child))
        for changed in (None,{**CONFIG,'enabled':False},{**CONFIG,'API_KEY':'must not persist'},{**CONFIG,'schema_version':'1'}):
            with self.assertRaises(ValueError):checked_config(changed)
        with self.assertRaises(ValueError):checked_model({**MODEL,'base_url':'https://secret@model.invalid'})
        with self.assertRaises(ValueError):load_connection({'model':MODEL,'credential_env':'TEST_MODEL_KEY'},{})

    def test_missing_configuration_is_blocked_without_worker(self):
        with TemporaryDirectory() as temp,patch.object(agent,'execute_sample') as execute,patch.object(runner,'collect_baseline',return_value={}):
            result,saved=runner.run_benchmark('second-project',root=Path(temp),registry={'second-project':lambda:deepcopy(SUITE)})
        execute.assert_not_called();self.assertEqual(exit_code(result),2)
        self.assertTrue(all(u['execution_status']=='blocked' for u in result['units']))
        self.assertEqual({p.name for p in saved},{'result.json','report.md','review-template.json','review.md'})

    def test_second_project_and_repeats_use_same_runner_without_project_branch(self):
        with TemporaryDirectory() as temp:
            root=Path(temp);manifests={};specs=[]
            def prepare(repo,source,path,definition):
                value={'material_root':str(path/'material'),'files':[],'material_version':'fixture'};manifests[str(path)]=value;return value
            def instantiate(prepared,path,task):
                return prepare(None,None,path,None)
            def execute(unit,spec,directory,connection,**kwargs):
                specs.append(spec);unit.update(execution_status='completed',agent_outcome=known('completed'))
                for turn in unit['turns']:
                    answer=directory/f"turn-{turn['turn']}.md";answer.write_text('Synthetic adapter output; no real model',encoding='utf-8')
                    turn.update(execution_status='completed',run_id=known(spec['run_id']+str(turn['turn'])),agent_outcome=known('completed'),
                                answer_sha256=sha256_file(answer),answer_evidence={'kind':'answer','path':str(answer),'source':'fixture','availability':'available'})
                return False
            with patch.object(runner,'collect_baseline',return_value={}),patch.dict('os.environ',ENV,clear=True),\
                 patch.object(agent,'prepare_materials',side_effect=prepare),patch.object(agent,'instantiate_materials',side_effect=instantiate),\
                 patch.object(agent,'verify_materials',side_effect=lambda p:manifests[str(p)]),\
                 patch('benchmarks.core.materials.verify_materials',side_effect=lambda p:manifests[str(p)]),\
                 patch.object(agent,'execute_sample',side_effect=execute):
                result,saved=runner.run_benchmark('second-project',config=CONFIG,root=root,registry={'second-project':lambda:deepcopy(SUITE)})
            validate_result(result)
            self.assertEqual(exit_code(result),3);self.assertFalse(result['framework_errors'])
            self.assertEqual(len(specs),2);self.assertNotEqual(specs[0]['material_root'],specs[1]['material_root'])
            self.assertNotEqual(specs[0]['run_id'],specs[1]['run_id'])
            self.assertEqual(result['batch']['planned_turns'],4)
            self.assertTrue(all(p.is_file() for p in saved))

    def test_unavailable_plan_directory_does_not_start(self):
        with TemporaryDirectory() as temp,patch.object(Path,'mkdir',side_effect=PermissionError()),\
             patch.object(agent,'prepare') as prepare,patch.object(runner,'collect_baseline',return_value={}):
            result,saved=runner.run_benchmark('second-project',root=Path(temp),registry={'second-project':lambda:deepcopy(SUITE)})
        prepare.assert_not_called();self.assertEqual(saved,[]);self.assertEqual(exit_code(result),2)
