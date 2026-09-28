"""Frozen material provenance, variants and task coverage; no endpoint calls."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from benchmarks.core.files import ROOT
from benchmarks.core import materials
from benchmarks.suites.wechat_reporting.suite import definition

class DefinitionTests(unittest.TestCase):
    def test_tasks_cover_research_workflow_with_matching_rubrics(self):
        suite=definition()
        self.assertEqual(len(suite['units']),9)
        self.assertEqual(sum(len(t['turns']) for t in suite['units']),19)
        self.assertEqual({t['case_id'] for t in suite['units']},set(suite['rubric']['cases']))
        for task in suite['units']:
            rule=suite['rubric']['cases'][task['case_id']]
            self.assertEqual(set(task['dimensions']),{c['dimension'] for c in rule['conditions']})
            self.assertTrue(rule['severe_errors'])
        self.assertEqual([len(s['paths']) for s in suite['materials']['sources']],[50,3])
        self.assertTrue(all('project.config.json' not in s['paths'] for s in suite['materials']['sources']))

    def test_task_loads_are_independent_and_no_fixed_model(self):
        first=definition();first['units'][0]['turns'][0]='changed'
        self.assertNotEqual(definition()['units'][0]['turns'][0],'changed')
        self.assertNotIn('profile',definition())
        self.assertNotIn('deepseek',json.dumps(definition()))

    def test_transcript_ambiguity_preserved(self):
        base=ROOT/'benchmarks/suites/wechat_reporting/transcripts'
        self.assertIn('数据库里加密？小程序前端展示需要脱敏？',(base/'简览.md').read_text(encoding='utf-8'))
        later=(base/'信息采集小程序需求简报.md').read_text(encoding='utf-8')
        self.assertIn('数据库里加密，小程序前端展示需要脱敏，',later)
        self.assertIn('## 第 2 页',later)

class MaterialTests(unittest.TestCase):
    def setUp(self):
        temp=TemporaryDirectory();self.addCleanup(temp.cleanup);self.root=Path(temp.name)
        self.suite=definition();self.definition=deepcopy(self.suite['materials'])
        self.pdf=b'reviewed PDF fixture'
        for item in self.definition['transcripts']:item['sha256']=hashlib.sha256(self.pdf).hexdigest()
        self.source=self.root/'source';self.source.mkdir()
        (self.source/'.env').write_text('CREDENTIAL MARKER')
        self.calls=[]
        def blob(source,commit,path):
            self.calls.append((commit,path))
            content=self.pdf if path.endswith('.pdf') else b'revision: draft.revision + 1\n  delete next.signature;\n' if path.endswith('domain/form.ts') else f'frozen {path}'.encode()
            return content,hashlib.sha1(content).hexdigest()
        patcher=patch.object(materials,'_read_blob',side_effect=blob);patcher.start();self.addCleanup(patcher.stop)
        self.prepared=self.root/'prepared'
        self.manifest=materials.prepare_materials(ROOT,self.source,self.prepared,self.definition)

    def task(self,name):return next(t for t in self.suite['units'] if t['case_id']==name)

    def test_frozen_files_only_and_small_index_without_grading(self):
        self.assertEqual(len(self.calls),55)
        self.assertEqual(len(self.manifest['files']),56)
        self.assertEqual(materials.verify_materials(self.prepared),self.manifest)
        root=Path(self.manifest['material_root'])
        self.assertLess((root/'SOURCE_INDEX.md').stat().st_size,2048)
        self.assertFalse(any(p.name in {'.env','research.json','manifest.json','project.config.json'} for p in root.rglob('*')))
        self.assertEqual((self.source/'.env').read_text(),'CREDENTIAL MARKER')

    def test_isolation_variants_and_fixture_provenance(self):
        instances={name:materials.instantiate_materials(self.prepared,self.root/name,self.task(name))
                   for name in ('defect-reference','defect-changed','compaction-continuation')}
        before=(Path(instances['defect-reference']['material_root'])/'code/miniprogram/domain/form.ts').read_text()
        after=(Path(instances['defect-changed']['material_root'])/'code/miniprogram/domain/form.ts').read_text()
        self.assertIn('revision: draft.revision + 1',before);self.assertIn('delete next.signature',before)
        self.assertNotIn('revision: draft.revision + 1',after);self.assertNotIn('delete next.signature',after)
        self.assertEqual((Path(instances['compaction-continuation']['material_root'])/'evaluation-fixtures/background.txt').stat().st_size,6144)
        self.assertEqual(materials.verify_materials(self.prepared),self.manifest)
        for name,instance in instances.items():self.assertEqual(materials.verify_materials(self.root/name),instance)
        with self.assertRaises(materials.MaterialError):materials.instantiate_materials(self.root/'defect-changed',self.root/'nested',self.task('defect-reference'))

    def test_repeat_destination_and_mutations_rejected(self):
        with self.assertRaises(FileExistsError):materials.prepare_materials(ROOT,self.source,self.prepared,self.definition)
        path=Path(self.manifest['material_root'])/'unexpected.txt';path.write_text('extra')
        with self.assertRaises(materials.MaterialError):materials.verify_materials(self.prepared)
        path.unlink();(Path(self.manifest['material_root'])/'code/README.md').write_text('changed')
        with self.assertRaises(materials.MaterialError):materials.instantiate_materials(self.prepared,self.root/'copy',self.task('defect-reference'))

    def test_path_escape_and_changed_manifest_rejected(self):
        for relative in ('../outside','C:/outside','/outside','..\\outside'):
            with self.assertRaises(materials.MaterialError):materials._destination(self.root,relative)
        manifest=deepcopy(self.manifest);manifest['material_root']=str(self.source)
        (self.prepared/'manifest.json').write_text(json.dumps(manifest))
        with self.assertRaises(materials.MaterialError):materials.verify_materials(self.prepared)

    def test_transcript_mismatch_or_save_failure_does_not_claim_manifest(self):
        with patch.object(materials,'_read_blob',return_value=(b'unexpected','blob')):
            with self.assertRaises(materials.MaterialError):materials.prepare_materials(ROOT,self.source,self.root/'failed',self.definition)
        self.assertFalse((self.root/'failed/manifest.json').exists())
        with patch.object(materials,'atomic_write_json',side_effect=OSError('disk fault')):
            with self.assertRaises(OSError):materials.instantiate_materials(self.prepared,self.root/'partial',self.task('defect-reference'))
        self.assertTrue((self.root/'partial/material/code/README.md').exists())
        self.assertFalse((self.root/'partial/manifest.json').exists())
