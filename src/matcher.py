import re
import pandas as pd


def normalize_number(value):
    if pd.isna(value):
        return ""
    text = str(value)
    groups = re.findall(r"\d+", text)
    if not groups:
        return ""
    group = max(groups, key=len).lstrip("0")
    return group


def normalize_invoice(value):
    if pd.isna(value):
        return ""
    text = str(value).upper()
    text = re.sub(r"\b(TAX\s*INV|INVOICE|INV|PO)\b", "", text)
    return re.sub(r"[^A-Z0-9]", "", text)


def match_company_a_systems(df_erp, df_tool):
    erp = df_erp.copy()
    tool = df_tool.copy()
    erp["record_id"] = "ERP:" + erp["line_id"].astype(str)
    tool["record_id"] = "TOOL:" + tool["tool_record_id"].astype(str)

    erp_po = {normalize_number(v) for v in erp["po_number"] if normalize_number(v)}
    erp_invoices = set()
    if "receipt_document" in erp:
        erp_invoices.update(normalize_invoice(v) for v in erp["receipt_document"] if normalize_invoice(v))

    seen_invoices = set()
    for idx, row in tool.iterrows():
        if row["include"] == 0:
            continue
        po = normalize_number(row.get("po_reference"))
        inv = normalize_invoice(row.get("invoice_number"))
        if po and po in erp_po:
            tool.loc[idx, "include"] = 0
            tool.loc[idx, "spend"] = 0.0
            tool.loc[idx, "exclusion_reason"] = "DUPLICATE"
        elif inv and inv in erp_invoices:
            tool.loc[idx, "include"] = 0
            tool.loc[idx, "spend"] = 0.0
            tool.loc[idx, "exclusion_reason"] = "DUPLICATE"
        elif inv:
            if inv in seen_invoices:
                tool.loc[idx, "include"] = 0
                tool.loc[idx, "spend"] = 0.0
                tool.loc[idx, "exclusion_reason"] = "DUPLICATE"
            else:
                seen_invoices.add(inv)
    return erp, tool
