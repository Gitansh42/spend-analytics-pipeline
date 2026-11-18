import argparse
import glob
import os
import pandas as pd
import yaml

from src.cleaner import clean_company_a_amounts, clean_company_b
from src.matcher import match_company_a_systems
from src.deduplicator import generate_keys
from src.classifier import classify_spend


OUTPUT_COLUMNS = [
    "record_id", "include", "exclusion_reason", "supplier_key", "item_key",
    "spend", "l1", "l2", "l3", "l4", "confidence",
]


def read_csv(path):
    return pd.read_csv(path, low_memory=False)


def run_pipeline(config_path, output_filename, classifier_provider=None):
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    mappings = config.get("mappings", {})
    classifier_config = config.get("classifier", {})
    provider = classifier_provider or classifier_config.get("provider", "local")
    classifier_options = {
        "provider": provider,
        "model": classifier_config.get("model", "gpt-4o-mini"),
        "batch_size": classifier_config.get("batch_size", 25),
    }
    company = str(config.get("company_name", ""))
    if "A" in company:
        erp = read_csv(config["files"]["erp"])
        tool = read_csv(config["files"]["tool"])
        erp, tool = clean_company_a_amounts(erp, tool)
        erp, tool = match_company_a_systems(erp, tool)

        erp["item_desc"] = erp["material_short_text"].fillna(erp.get("service_short_text"))
        tool["item_desc"] = tool["line_text"]

        combined = pd.concat([erp, tool], ignore_index=True)
        combined["supplier_name"] = combined["supplier_name"].fillna(
            combined["supplier_name_as_captured"]
        ) if "supplier_name" in combined else combined["supplier_name_as_captured"]
        training = None
        supplier_training = None
        if config["files"].get("labels") and os.path.exists(config["files"]["labels"]):
            labels = read_csv(config["files"]["labels"])
            erp_label_cols = ["record_id", "category_l1", "category_l2", "category_l3", "supplier_key", "item_key"]
            erp_labels = erp[["line_id", "material_short_text", "service_short_text", "supplier_name", "vendor_id"]].copy()
            erp_labels["record_id"] = "ERP:" + erp_labels["line_id"].astype(str)
            erp_labels["item_desc"] = erp_labels["material_short_text"].fillna(erp_labels["service_short_text"])
            labels = labels.merge(
                erp_labels[["record_id", "item_desc", "supplier_name", "vendor_id"]],
                on="record_id", how="left",
            )
            training = labels.rename(columns={"item_desc": "item_desc"})
            supplier_training = labels.rename(columns={"supplier_name": "supplier_name"})

        combined = generate_keys(
            combined,
            "item_desc",
            "supplier_name" if "supplier_name" in combined else "supplier_name_as_captured",
            training_df=training,
            supplier_training=supplier_training,
        )
        combined = classify_spend(
            combined,
            config["files"]["taxonomy"],
            "item_desc",
            label_training=training.rename(columns={"item_desc": "item_desc"}) if training is not None else None,
            **classifier_options,
        )
    else:
        invoice_key = "invoices" if "invoices" in config["files"] else next(
            k for k in ("invoice_lines", "input", "file") if k in config["files"]
        )
        df = read_csv(config["files"][invoice_key])
        fx = read_csv(config["files"]["fx_rates"]) if config["files"].get("fx_rates") else None
        seed = read_csv(config["files"]["seed_labels"]) if config["files"].get("seed_labels") else None
        df = clean_company_b(df, fx, seed)

        id_col = mappings.get("id_col", "line_id")
        supplier_col = mappings.get("supplier_col", "vendor_id")
        desc_col = mappings.get("desc_col", "line_description")
        df["record_id"] = df[id_col].astype(str)
        df["item_desc"] = df[desc_col].fillna("") if desc_col in df else ""

        training = None
        if seed is not None:
            seed_training = seed.merge(
                df[[id_col, desc_col]],
                left_on="line_id",
                right_on=id_col,
                how="left",
            )
            training = seed_training

        df = generate_keys(df, "item_desc", supplier_col)
        df = classify_spend(
            df,
            config["files"]["taxonomy"],
            "item_desc",
            label_training=training,
            label_desc_col=desc_col,
            **classifier_options,
        )
        combined = df

    for column in OUTPUT_COLUMNS:
        if column not in combined.columns:
            combined[column] = ""
    combined[OUTPUT_COLUMNS].to_csv(output_filename, index=False)
    print("Saved", len(combined), "rows to", output_filename)


def main():
    parser = argparse.ArgumentParser(description="Config-driven spend pipeline")
    parser.add_argument("--config", help="YAML config file")
    parser.add_argument("--output", help="Output CSV path")
    parser.add_argument(
        "--classifier",
        choices=["local", "openai"],
        help="Override the config classifier (OpenAI sends item descriptions to the API).",
    )
    args = parser.parse_args()

    if args.config:
        run_pipeline(args.config, args.output or "output.csv", args.classifier)
        return

    config_files = sorted(glob.glob(os.path.join("configs", "*.yaml")) +
                          glob.glob(os.path.join("configs", "*.yml")))
    for config_path in config_files:
        name = os.path.basename(config_path).replace("config_", "").replace(".yaml", "").replace(".yml", "").upper()
        run_pipeline(config_path, "output_" + name + ".csv", args.classifier)


if __name__ == "__main__":
    main()
