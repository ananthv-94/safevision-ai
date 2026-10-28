import json
import uuid
from pathlib import Path
import pandas as pd
import streamlit as st
from safevision.events import DEFAULT_ZONE, polygon
from safevision.pipeline import process

ROOT=Path(__file__).resolve().parent
st.set_page_config(page_title='SafeVision AI',page_icon='🛡️',layout='wide')
st.markdown('''<style>.stApp{background:#0d1420} [data-testid="stMetric"]{background:#182435;padding:18px;border-radius:10px} h1{letter-spacing:-1px} </style>''',unsafe_allow_html=True)
st.caption('SAFEVISION AI  /  OPERATIONS CONSOLE')
st.title('Industrial safety monitoring')
st.write('Person detection · persistent tracking · restricted-zone alerts')
st.info('Demonstration only. Zones are configured for this demo; alerts do not establish actual unsafe conduct.')
with st.sidebar:
    st.header('Monitoring setup')
    mode=st.radio('Source',['Included warehouse demo','Upload video','Local webcam'])
    upload=st.file_uploader('Video file',type=['mp4','mov','avi','mkv']) if mode=='Upload video' else None
    camera=st.number_input('Camera index',0,10,0) if mode=='Local webcam' else 0
    tracker=st.selectbox('Tracker',['bytetrack','botsort'])
    confidence=st.slider('Detection confidence',.1,.95,.35,.05)
    st.caption('CPU execution · no NVIDIA GPU required')
    device='cpu'
    choices=['Original YOLO11 Nano']
    if (ROOT/'models/best.pt').exists():choices.append('Fine-tuned Nano · PyTorch CPU')
    if (ROOT/'models/best.onnx').exists():choices.append('Fine-tuned Nano · ONNX Runtime CPU')
    model_choice=st.selectbox('Detection model / engine',choices)
    engine='onnx' if 'ONNX' in model_choice else 'pytorch' if 'Fine-tuned' in model_choice else 'ultralytics'
    selected_model=ROOT/('models/best.onnx' if engine=='onnx' else 'models/best.pt' if engine=='pytorch' else 'yolo11n.pt')
    limit=st.number_input('Maximum frames (0 = full video)',0,100000,250,50)
    cooldown=st.number_input('Re-entry cooldown (seconds)',0.,120.,5.)
    track_experiment=st.checkbox('Log this run to MLflow',value=True)
    st.caption('Local experiment history. Video output is kept if logging is unavailable.')
    zone_text=st.text_area('Restricted zone: normalized [x,y] points',json.dumps(DEFAULT_ZONE),height=130)
    st.caption('Convex polygon, ordered around its edge. Bottom-center footpoint; boundary counts as inside.')
    start=st.button('Start processing',type='primary',use_container_width=True)

if upload is not None:
    with st.expander('Uploaded source preview',expanded=True):
        st.video(upload)

if start:
    try:
        zone=json.loads(zone_text)
        polygon(zone,1280,720)
        run=ROOT/'runs'/uuid.uuid4().hex[:12]
        run.mkdir(parents=True)
        source=ROOT/'demo'/'warehouse.mp4'
        if mode=='Upload video':
            if upload is None:
                raise ValueError('Choose a video first.')
            source=run/('input'+Path(upload.name).suffix.lower())
            source.write_bytes(upload.getbuffer())
        elif mode=='Local webcam':
            source=int(camera)
            if limit==0:
                raise ValueError('Set a frame limit for webcam recording.')
        elif not source.exists():
            raise ValueError('Sample unavailable. Upload your own video or run scripts/download_demo.py.')
        bar=st.progress(0,text='Loading detector…')
        preview=st.empty()
        def progress(n,total,frame):
            denominator=min(total,limit) if total>0 and limit else total or limit or n+1
            bar.progress(min(n/max(1,denominator),1.),text=f'Processed {n} frames')
            preview.image(frame,channels='BGR',use_container_width=True)
        with st.spinner('Processing locally; please keep this page open.'):
            process(str(source) if isinstance(source,Path) else source,run,tracker,confidence,zone,
                    str(selected_model),device,int(limit),progress,cooldown,
                    track_experiment=track_experiment,engine=engine,imgsz=320,threads=2)
        bar.empty()
        preview.empty()
        st.session_state['result']=str(run)
    except Exception as e:
        st.error(f'Processing could not finish: {e}')
        st.caption('Check your video, model file and device. CPU is the most portable option.')

result=Path(st.session_state.get('result',ROOT/'demo'))
if (result/'summary.json').exists():
    s=json.loads((result/'summary.json').read_text())
    if (result/'mlflow.json').exists():
        tracking=json.loads((result/'mlflow.json').read_text())
        if tracking['status']=='logged':
            local_record=ROOT/'mlruns'/str(tracking.get('experiment_id',''))/str(tracking['run_id'])/'meta.yaml'
            if tracking.get('tracking_uri')==(ROOT/'mlruns').resolve().as_uri() and local_record.is_file():
                st.success(f"MLflow run saved: {tracking['run_id']}")
                st.link_button('Open MLflow experiment history','http://127.0.0.1:5000')
                st.caption('Start START_MLFLOW.bat first if the experiment viewer is not running.')
            else:
                st.info('This result has a historical MLflow receipt. Run track_run.py on its output folder to recreate the experiment history here.')
        else:
            st.warning('Video processing succeeded, but MLflow logging was unavailable: '+tracking.get('error','Unknown error'))
    cols=st.columns(4)
    for c,title,value in zip(cols,['Unique track IDs','Zone incidents','Processing FPS','Inference latency'],
                             [s['unique_track_ids'],s['incidents'],f"{s['processing_fps']:.1f}",f"{s['inference_ms']:.1f} ms"]):
        c.metric(title,value)
    left,right=st.columns([2,1])
    with left:
        st.subheader('Annotated monitoring video')
        st.video(str(result/'annotated.mp4'))
        st.download_button('Download annotated MP4',(result/'annotated.mp4').read_bytes(),'safevision.mp4','video/mp4')
    with right:
        st.subheader('Incident history')
        st.dataframe(pd.read_csv(result/'incidents.csv'),hide_index=True,use_container_width=True)
        st.download_button('Export incident CSV',(result/'incidents.csv').read_bytes(),'incidents.csv','text/csv')
        st.caption('Timestamps are seconds from the start of the source video. IDs are tracker trajectories, not verified individuals.')
        st.caption(f"{s['tracker']} · {s['frames']} frames · {s['device']}")
    with st.expander('Source preview and measurement details'):
        if (ROOT/'demo'/'warehouse.mp4').exists():
            st.video(str(ROOT/'demo'/'warehouse.mp4'))
        st.json(s)
        st.caption('Processing FPS includes decoding, tracking, drawing and writing, excluding warmup and final H.264 encoding. Inference latency is Ultralytics model inference only.')
else:
    st.write('Upload a video or download the sample to begin. Results and CSV exports appear here after processing.')

st.header('Model Evaluation & Benchmarking')
with st.expander('Measured CPU training, evaluation and optimization results',expanded=False):
    training=ROOT/'training/cpu_person/training_summary.json'
    if training.exists():
        trained=json.loads(training.read_text())
        st.write(f"YOLO11 Nano: {trained['epochs_completed']} CPU training epochs completed.")
        st.image(str(ROOT/'training/cpu_person/loss_curves.png'),caption='Measured training and validation losses',use_container_width=True)
    evaluation=ROOT/'reports/evaluation_comparison.csv'
    if evaluation.exists():
        st.subheader('Held-out person detection')
        comparison_data=pd.read_csv(evaluation)
        st.dataframe(comparison_data,hide_index=True,use_container_width=True)
        accuracy_row=comparison_data[comparison_data['metric']=='map50_95']
        if not accuracy_row.empty and accuracy_row.iloc[0]['fine_tuned']<accuracy_row.iloc[0]['pretrained']:
            st.warning('The small fine-tuning experiment scored lower on this test set. The original pretrained model remains the default.')
        st.caption('Same grouped test split and evaluation settings. Small pilot dataset; no production accuracy claim.')
        for name in ('BoxPR_curve.png','confusion_matrix.png'):
            picture=ROOT/'reports/evaluation/fine_tuned'/name
            if picture.exists():st.image(str(picture),use_container_width=True)
    comparison=ROOT/'benchmark_results.csv'
    if comparison.exists():
        st.subheader('PyTorch vs direct ONNX Runtime · CPU')
        st.dataframe(pd.read_csv(comparison),hide_index=True,use_container_width=True)
        chart=ROOT/'reports/benchmark/comparison.png'
        if chart.exists():st.image(str(chart),use_container_width=True)
        st.caption('FP32, batch 1, two CPU threads, identical inputs and postprocessing. Peak process RAM includes libraries and inputs.')
    if not evaluation.exists() or not comparison.exists():
        st.info('Run evaluate.py --compare and benchmark.py to populate measured results.')
