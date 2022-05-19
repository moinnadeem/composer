import subprocess

import wandb
import yaml
from tqdm import tqdm

api = wandb.Api(timeout=19)

entity = "mosaic-ml"
conv_project = f"bert-fused-layernorm"
bert_conv_runs = api.runs(f"{entity}/{conv_project}")
for run in tqdm(bert_conv_runs):
    if run.state != "finished":
        continue
    if 'load_path_format' in run.config:
        continue

    command_name = ['kubectl', 'get', 'configmaps', run.name, '-o', 'yaml']
    result = subprocess.run(command_name, stdout=subprocess.PIPE)
    result = yaml.safe_load(result.stdout)
    parameters = yaml.safe_load(result['data']['parameters.yaml'])
    run.config['parameters'] = parameters
    run.update()
