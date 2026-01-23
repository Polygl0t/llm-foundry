"""
Evaluation Results Post-processor

Converts JSON evaluation results from lm-evaluation-harness to YAML format for easier
analysis and integration. Flattens nested result structures and extracts model metadata.

Output Format (YAML):
```yaml
model_name: model_identifier
model_pretrained: path/to/model
results:
  task_metric: value
  task_acc: 0.85
  task_f1: 0.82
```

Usage:
    # Basic conversion
    python post_processing_base.py \
        --logs_dir evaluation_logs/ \
        --output_folder processed_results/
"""
import os
import yaml
import json
import argparse

def main (args):
    output_folder = args.output_folder
    logs_dir = args.logs_dir

    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    json_files = []
    for entry in os.listdir(logs_dir):
        path = os.path.join(logs_dir, entry)
        if os.path.isdir(path):
            json_files += [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".json")]
        elif entry.endswith(".json"):
            json_files.append(path)

    if not json_files:
        print(f"No JSON files found in {logs_dir}")
        return

    for file in json_files:
        try:
            with open(file, "r") as f:
                data = json.load(f)
        except Exception as e:
            print(f"Failed to read {file}: {e}")
            continue

        results = data.get("results", data)

        pretrained = None
        config = data.get("config", {})
        model_args = config.get("model_args", {}) if isinstance(config, dict) else {}
        if isinstance(model_args, dict):
            pretrained = model_args.get("pretrained")
        if pretrained:
            model_name = os.path.basename(pretrained.rstrip("/"))
        else:
            fname = os.path.basename(file)

            for prefix in ("results_", "result_", "eval_"):
                if fname.startswith(prefix):
                    fname = fname[len(prefix):]
                    break
            model_name = os.path.splitext(fname)[0]

        flat_data = {}
        if isinstance(results, dict):
            for key, value in results.items():
                if isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        clean_subkey = subkey.replace(",none", "")
                        flat_data[f"{key}_{clean_subkey}"] = subvalue
                else:
                    flat_data[key] = value
        else:
            flat_data["results_raw"] = results

        out = {"model_name": model_name, "model_pretrained": pretrained, "results": flat_data}

        path_to_save = os.path.join(output_folder, model_name + ".yaml")
        if os.path.exists(path_to_save):
            print(f"File {path_to_save} already exists. Skipping.")
        else:
            try:
                with open(path_to_save, "w") as f:
                    yaml.dump(out, f, default_flow_style=False)
            except Exception as e:
                print(f"Failed to write {path_to_save}: {e}")

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
