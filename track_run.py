"""Log existing results without repeating model inference."""
import argparse
import json
from safevision.experiment import log_result

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', help='Completed output directory containing summary.json')
    parser.add_argument('--source', help='Original video path, for its SHA-256 fingerprint')
    parser.add_argument('--store', help='Local MLflow store directory; default: project/mlruns')
    args = parser.parse_args()
    result = log_result(args.output, source=args.source, store=args.store)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['status'] == 'logged' else 1)
