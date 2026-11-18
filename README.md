# Spend Analytics & Classification Pipeline

## Setup
```bat
python -m pip install -r requirements.txt
```

Python 3.10+ is recommended.

## Run
Run every configured company:
```bat
python main.py
```

Run one config and choose the output:
```bat
python main.py --config configs/config_A.yaml --output gitansh__A.csv
python main.py --config configs/config_B.yaml --output gitansh__B.csv
python main.py --config configs/config_C.yaml --output gitansh__C.csv
```

The default classifier is local and requires no API key. To explicitly classify with
OpenAI (the item descriptions will be sent to OpenAI), set your key in the current
PowerShell session and opt in:
```powershell
$env:OPENAI_API_KEY = "your-api-key"
python main.py --config configs/config_A.yaml --output gitansh__A.csv --classifier openai
python main.py --config configs/config_B.yaml --output gitansh__B.csv --classifier openai
```
Set `classifier.provider: openai` in a config to make that the config's default.
`classifier.model` and `classifier.batch_size` are configurable; repeated descriptions
are sent only once per run. API usage is billed by the provider and varies with input
size, model and batch size. API confidence is model self-reported, not calibrated.
The pipeline does not silently fall back if an API call fails; inspect the error and
rerun with `--classifier local` if desired.

Do not commit API keys or company data. The `data/` directory and generated submission
CSV names are ignored in this repository.