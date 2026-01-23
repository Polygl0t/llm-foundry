"""
Evaluation Results Post-processor

Converts JSON evaluation results to YAML format for Portuguese language model benchmarks.

Output Format (YAML):
```yaml
model_name: model_identifier
results:
  task_metric: value
  calame_pt_acc: 0.82
  assin2_f1: 0.75
```

Usage:
    # Convert Portuguese evaluation results
    python post_processing_portuguese.py \
        --logs_dir pt_eval_logs/ \
        --output_folder pt_results/
"""
import yaml
import json
import argparse

def main (args):
    output_folder = args.output_folder
    logs_dir = args.logs_dir
    eval_files = os.listdir(logs_dir)
    for eval_file in eval_files:
        
        path = os.path.join(logs_dir, eval_file)

        with open(path, "r") as f:
            results = json.load(f)['results']
        model_name = eval_file.split(".json")[0]

        flat_data = {}
        for key, value in results.items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    flat_data[f"{key}_{subkey.replace(',none', '')}"] = subvalue
            else:
                flat_data[key] = value
        
        flat_data = {"model_name": model_name, "results" : flat_data}

        path_to_save = os.path.join(output_folder, model_name + ".yaml")
        if os.path.exists(path_to_save):
            print(f"File {path_to_save} already exists. Skipping.")
    
        else:
            with open(path_to_save, "w") as f:
                yaml.dump(flat_data, f, default_flow_style=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Post-process JSON files.")
    parser.add_argument(
        "--logs_dir",
        type=str,
        default=None,
        help="Directory containing the JSON log files",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default=None,
        help="Output folder for the yaml files",
    )
    args = parser.parse_args()
    main(args)
