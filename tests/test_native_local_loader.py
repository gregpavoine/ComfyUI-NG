from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / 'plugins/flux/src/comfyng_flux/runtime.py'
DISPATCHER = ROOT / 'src/comfyng/api/dispatcher.py'


def test_runtime_has_no_remote_model_ids():
    text = RUNTIME.read_text(encoding='utf-8')
    forbidden = ('Z-a-o/', 'black-forest-labs/', 'snapshot_download', 'hf_hub_download')
    assert not any(item in text for item in forbidden)


def test_dispatcher_does_not_search_hf_cache():
    text = DISPATCHER.read_text(encoding='utf-8')
    assert '.cache" / "huggingface' not in text
    assert 'local_files_only": True' in text


def test_no_legacy_http_bridge():
    text = '\n'.join(p.read_text(encoding='utf-8', errors='ignore') for p in (RUNTIME, DISPATCHER))
    assert '127.0.0.1:8188' not in text
    assert 'urllib.request.urlopen' not in text
