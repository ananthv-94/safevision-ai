"""Recreate MLflow history from completed training outputs, without retraining."""
import argparse,json
from pathlib import Path
from safevision.training_log import log_training
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',nargs='?',default='training/cpu_person')
    args=parser.parse_args();output=Path(args.output)
    config=json.loads((output/'training_config.json').read_text())
    receipt=log_training(output,config,record_type='replayed_training_output')
    print(json.dumps(receipt,indent=2))
    raise SystemExit(0 if receipt['status']=='logged' else 1)
