"""Isolated FP32 CPU trials with identical square letterboxing and numpy NMS."""
import argparse,json,os,subprocess,sys,time,threading,platform,hashlib
from importlib.metadata import version
from pathlib import Path
import numpy as np
import cv2
import psutil
ROOT=Path(__file__).resolve().parent

def worker(a):
    from safevision.engines import TorchEngine,OnnxEngine,preprocess,decode
    cv2.setNumThreads(1)
    process=psutil.Process();peak=[process.memory_info().rss];stop=threading.Event()
    def monitor():
        while not stop.wait(.005):peak[0]=max(peak[0],process.memory_info().rss)
    monitor_thread=threading.Thread(target=monitor,daemon=True);monitor_thread.start()
    try:
        engine=TorchEngine(a.pt,a.threads,a.imgsz) if a.worker=='pytorch' else OnnxEngine(a.onnx,a.threads)
        if engine.size!=a.imgsz:raise ValueError('Image size does not match exported model')
        paths=sorted((ROOT/'datasets/safevision/images/test').glob('*.jpg'))
        images=[cv2.imread(str(p)) for p in paths]
        tensors=[preprocess(image,a.imgsz) for image in images]
        for i in range(a.warmup):engine.forward(tensors[i%len(tensors)][0])
        latency=[];total=[];prep=[];post=[]
        for i in range(a.repeats):
            image=images[i%len(images)]
            t0=time.perf_counter();tensor,meta=preprocess(image,a.imgsz);t1=time.perf_counter()
            raw=engine.forward(tensor);t2=time.perf_counter()
            decode(raw,meta,.25,.7);t3=time.perf_counter()
            prep.append((t1-t0)*1000);latency.append((t2-t1)*1000)
            post.append((t3-t2)*1000);total.append((t3-t0)*1000)
        predictions={}
        for path,(tensor,meta) in zip(paths,tensors):
            predictions[path.name]=decode(engine.forward(tensor),meta,.25,.7).tolist()
        memory=process.memory_info()
        peak[0]=max(peak[0],memory.rss,getattr(memory,'peak_wset',0))
        result=dict(engine=a.worker,latency_ms=latency,preprocess_ms=prep,postprocess_ms=post,
                    total_ms=total,peak_process_ram_mb=peak[0]/1024**2,
                    model_mb=Path(a.pt if a.worker=='pytorch' else a.onnx).stat().st_size/1024**2,
                    predictions=predictions,threads=a.threads,batch=1,precision='FP32',
                    torch_imported='torch' in sys.modules,
                    image_names=[p.name for p in paths],
                    provider='CPUExecutionProvider' if a.worker=='onnxruntime' else 'PyTorch CPU')
        if a.worker=='onnxruntime':assert not result['torch_imported']
        Path(a.output).write_text(json.dumps(result,indent=2))
    finally:stop.set();monitor_thread.join()

def consistency(torch_predictions,onnx_predictions):
    from safevision.engines import box_iou
    matched=0;torch_count=0;onnx_count=0;ious=[];score_errors=[]
    for name,rows in torch_predictions.items():
        left=np.asarray(rows,dtype=float).reshape(-1,6)
        right=np.asarray(onnx_predictions[name],dtype=float).reshape(-1,6)
        torch_count+=len(left);onnx_count+=len(right);used=set()
        for box in left:
            options=[i for i in range(len(right)) if i not in used and right[i,5]==box[5]]
            if not options:continue
            overlaps=box_iou(box[:4],right[options,:4]);best=int(np.argmax(overlaps))
            if overlaps[best]>=.95:
                index=options[best];used.add(index);matched+=1;ious.append(float(overlaps[best]))
                score_errors.append(float(abs(box[4]-right[index,4])))
    return dict(torch_detections=torch_count,onnx_detections=onnx_count,matched_at_iou95=matched,
                matched_fraction=matched/max(torch_count,onnx_count) if max(torch_count,onnx_count) else 1.0,
                mean_matched_iou=float(np.mean(ious)) if ious else None,
                maximum_confidence_error=max(score_errors) if score_errors else None,
                matching_threshold=.95,confidence=.25,nms_iou=.7)

def benchmark(a):
    out=ROOT/'reports/benchmark';out.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy()
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):env[key]=str(a.threads)
    trials={'pytorch':[],'onnxruntime':[]}
    for trial in range(a.trials):
        order=['pytorch','onnxruntime'] if trial%2==0 else ['onnxruntime','pytorch']
        for engine in order:
            output=out/f'{engine}_{trial}.json'
            command=[sys.executable,str(ROOT/'benchmark.py'),'--worker',engine,'--pt',str(Path(a.pt).resolve()),
                     '--onnx',str(Path(a.onnx).resolve()),'--imgsz',str(a.imgsz),'--threads',str(a.threads),
                     '--warmup',str(a.warmup),'--repeats',str(a.repeats),'--output',str(output)]
            subprocess.run(command,cwd=ROOT,env=env,check=True)
            trials[engine].append(json.loads(output.read_text()))
            print(f'{engine} trial {trial+1} finished',flush=True)
    results={}
    for engine,items in trials.items():
        values=np.concatenate([x['latency_ms'] for x in items])
        total=np.concatenate([x['total_ms'] for x in items])
        results[engine]=dict(average_latency_ms=float(values.mean()),median_latency_ms=float(np.median(values)),
             p95_latency_ms=float(np.percentile(values,95)),inference_fps=1000/float(values.mean()),
             detection_pipeline_fps=1000/float(total.mean()),
             preprocess_ms=float(np.mean([v for x in items for v in x['preprocess_ms']])),
             postprocess_ms=float(np.mean([v for x in items for v in x['postprocess_ms']])),
             peak_process_ram_mb=max(x['peak_process_ram_mb'] for x in items),model_mb=items[0]['model_mb'])
    agreement=consistency(trials['pytorch'][0]['predictions'],trials['onnxruntime'][0]['predictions'])
    if a.video:
        from safevision.pipeline import process
        for engine,key,model in [('pytorch','pytorch',a.pt),('onnx','onnxruntime',a.onnx)]:
            summary=process(str(Path(a.video).resolve()),out/f'video_{key}',model_path=model,
                            max_frames=a.video_frames,engine=engine,imgsz=a.imgsz,threads=a.threads)
            results[key]['video_processing_fps']=summary['processing_fps']
            results[key]['video_with_final_encoding_fps']=summary['frames']/(summary['processing_seconds']+summary['encoding_seconds'])
    speedup=results['pytorch']['average_latency_ms']/results['onnxruntime']['average_latency_ms']
    hardware=dict(platform=platform.platform(),processor=platform.processor(),logical_cpus=os.cpu_count())
    if os.name=='nt':
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as key:
            hardware['cpu_name']=winreg.QueryValueEx(key,'ProcessorNameString')[0]
    provenance=dict(hardware=hardware,versions={key:version(key) for key in ('torch','onnxruntime','numpy','opencv-python')},
                    pytorch_sha256=hashlib.sha256(Path(a.pt).read_bytes()).hexdigest(),
                    onnx_sha256=hashlib.sha256(Path(a.onnx).read_bytes()).hexdigest(),
                    dataset_version=json.loads((ROOT/'datasets/safevision/quality_report.json').read_text())['dataset_version'])
    report=dict(provenance=provenance,results=results,prediction_consistency=agreement,onnx_speedup=speedup,
                settings=dict(imgsz=a.imgsz,threads=a.threads,warmup=a.warmup,repeats=a.repeats,
                trials=a.trials,batch=1,precision='FP32',image_names=trials['pytorch'][0]['image_names'],
                preprocessing='Identical square letterbox, BGR to RGB, FP32 /255, NCHW',
                timing='Model forward excludes loading, warmup, preprocessing and NMS. Detection pipeline adds preprocessing/postprocessing, excludes disk reads.',
                memory='Separate process high-water RSS/Windows peak working set; includes Python, libraries, model, inputs and loading; not model-only memory.',
                video='Includes decode, detection, tracking, annotation and intermediate video write; excludes final H264 encoding, consistent with existing pipeline.'))
    (ROOT/'reports/benchmark_results.json').write_text(json.dumps(report,indent=2))
    import pandas as pd
    rows=[dict(metric=key,pytorch_cpu=value,onnxruntime_cpu=results['onnxruntime'][key]) for key,value in results['pytorch'].items()]
    pd.DataFrame(rows).to_csv(ROOT/'benchmark_results.csv',index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    axes[0].bar(['PyTorch','ONNX Runtime'],[results[e]['average_latency_ms'] for e in ('pytorch','onnxruntime')],color=['#39a998','#477ad2'])
    axes[0].set_ylabel('Mean model latency (ms)')
    axes[1].bar(['PyTorch','ONNX Runtime'],[results[e]['peak_process_ram_mb'] for e in ('pytorch','onnxruntime')],color=['#39a998','#477ad2'])
    axes[1].set_ylabel('Peak process RAM (MiB)')
    fig.tight_layout();fig.savefig(out/'comparison.png',dpi=140);plt.close(fig)
    table='| Metric | PyTorch CPU | ONNX Runtime CPU |\n|---|---:|---:|\n'
    table+='\n'.join(f"| {r['metric']} | {r['pytorch_cpu']:.3f} | {r['onnxruntime_cpu']:.3f} |" for r in rows)
    text=f'# CPU benchmark\n\n{table}\n\nMeasured native/ONNX latency ratio: **{speedup:.3f}x**. Greater than 1 means ONNX is faster.\n'
    text+=f'\nFP32, batch 1, {a.imgsz}px, {a.threads} CPU threads; {a.trials} isolated trials per engine, {a.repeats} measured forwards and {a.warmup} warmups each. Engine order alternates between trials. Same held-out images and shared preprocessing/NMS.\n'
    text+='\n'+report['settings']['timing']+' '+report['settings']['memory']+'\n'
    text+='\nPrediction consistency: '+json.dumps(agreement)+'.\n'
    text+='\nThe native checkpoint stores FP16 weights on disk but both engines execute FP32. The ONNX graph stores FP32 weights; file size is not memory footprint.\n'
    text+='\nDifferences can arise from graph fusion, kernel implementations and runtime overhead. No causal performance attribution is established by this single-laptop experiment. Full video throughput also includes tracking, drawing and encoding writes, so model speedup need not equal video speedup. The ONNX benchmark workers assert that PyTorch was never imported. The tracker-integrated app uses Ultralytics tracking libraries which may import torch, but ONNX model execution is always direct CPUExecutionProvider.\n'
    (ROOT/'BENCHMARK_REPORT.md').write_text(text)
    print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker',choices=['pytorch','onnxruntime'])
    p.add_argument('--pt',default='models/best.pt');p.add_argument('--onnx',default='models/best.onnx')
    p.add_argument('--imgsz',type=int,default=320);p.add_argument('--threads',type=int,default=2)
    p.add_argument('--warmup',type=int,default=10);p.add_argument('--repeats',type=int,default=100)
    p.add_argument('--trials',type=int,default=3);p.add_argument('--output',default='reports/worker.json')
    p.add_argument('--video');p.add_argument('--video-frames',type=int,default=100)
    a=p.parse_args()
    if min(a.threads,a.repeats,a.trials,a.imgsz)<=0 or a.warmup<0:p.error('Invalid benchmark settings')
    if a.worker:worker(a)
    else:benchmark(a)
