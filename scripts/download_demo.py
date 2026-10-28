"""Download licensed stock footage, resize to CPU-friendly 720p at 25 FPS."""
import sys
import urllib.request
from pathlib import Path
import subprocess
import imageio_ffmpeg

ROOT=Path(__file__).resolve().parents[1]
URL='https://videos.pexels.com/video-files/4293958/4293958-uhd_3840_2160_25fps.mp4'
if __name__=='__main__':
    target=ROOT/'demo'
    target.mkdir(exist_ok=True)
    raw=target/'download.mp4'
    try:
        req=urllib.request.Request(URL,headers={'User-Agent':'Mozilla/5.0'})
        with urllib.request.urlopen(req,timeout=120) as response,raw.open('wb') as f:
            import shutil
            shutil.copyfileobj(response,f)
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-y','-loglevel','error',
                        '-i',str(raw),'-vf','scale=1280:720','-an','-c:v','libx264',
                        '-preset','fast','-crf','23','-pix_fmt','yuv420p',
                        '-movflags','+faststart',str(target/'warehouse.mp4')],check=True)
        raw.unlink()
        print(f'Saved browser-compatible sample to {target / "warehouse.mp4"}')
    except Exception as e:
        print(f'Sample download failed: {e}. Upload a local video instead.',file=sys.stderr)
        sys.exit(1)
