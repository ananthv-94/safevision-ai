import pytest
from safevision.pipeline import process

def test_bad_video_fails_without_success_summary(tmp_path):
    output=tmp_path/'result'
    output.mkdir()
    (output/'summary.json').write_text('{"old_success": true}')
    with pytest.raises(ValueError,match='Cannot open'):
        process(str(tmp_path/'missing.mp4'),output)
    assert (output/'incidents.csv').exists()
    assert not (output/'summary.json').exists()
