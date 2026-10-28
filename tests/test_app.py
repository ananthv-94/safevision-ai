from pathlib import Path
from streamlit.testing.v1 import AppTest

APP=Path(__file__).resolve().parents[1]/'app.py'

def test_dashboard_loads_and_rejects_bad_zone():
    app=AppTest.from_file(str(APP),default_timeout=30).run()
    assert not app.exception
    assert app.title[0].value=='Industrial safety monitoring'
    assert any(h.value=='Model Evaluation & Benchmarking' for h in app.header)
    selector=next(x for x in app.selectbox if x.label=='Detection model / engine')
    assert selector.value=='Original YOLO11 Nano'
    app.text_area[0].set_value('[[0,0],[1,1]]')
    app.button[0].click().run()
    assert not app.exception
    assert app.error and 'three' in app.error[0].value
