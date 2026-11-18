# Spend Analytics Pipeline

## Pipeline
YAML configs define paths and field mappings. The pipeline parses local numbers and
dates, applies tax/currency rules, decides exclusions, matches duplicate records,
creates supplier/item keys, and writes the required output.

## Company A matching
PO references are normalized to digit sequences and matched against ERP PO numbers.
Invoice identifiers and repeated normalized invoice numbers are also compared. The
ERP record is retained and overlapping tool records are marked `DUPLICATE`.

## Deduplication
Supplier names are normalized and, for A, matched to golden supplier keys by vendor
ID and fuzzy name. B uses vendor IDs as supplier keys. Items use character TF-IDF
nearest-neighbor matching. A and B labels seed that matching and unseen items receive
stable fallback keys.

## Classification
The taxonomy is loaded at runtime. By default, labels train a local character TF-IDF
classifier; without labels, descriptions are compared with the available taxonomy
paths. An explicit `--classifier openai` option uses the configured OpenAI model to
classify unique descriptions in batches. The API result is checked against the
taxonomy before output. This option transmits descriptions to the API and incurs
provider charges; the local path remains available.

## Validation and cost
No API key or API call is needed for the default local path. API cost depends on the
chosen model and descriptions sent; do not claim a zero-cost estimate for the opt-in
OpenAI path. Runtime depends on provider rate limits when the API option is enabled.
