import pandas as pd
import re


def parse_number(value):
    if pd.isna(value):
        return None
    text = str(value).strip().replace("\u00a0", " ")
    text = re.sub(r"[^\d,\.\-\(\)]", "", text)
    if not text:
        return None
    negative = text.startswith("(") and text.endswith(")")
    text = text.replace("(", "").replace(")", "")
    if "," in text and "." in text:
        text = text.replace(",", "") if text.rfind(".") > text.rfind(",") else text.replace(".", "").replace(",", ".")
    elif "," in text:
        parts = text.split(",")
        text = ".".join(parts) if len(parts[-1]) <= 2 else "".join(parts)
    try:
        result = float(text)
        return -abs(result) if negative else result
    except (TypeError, ValueError):
        return None


def parse_dates(series):
    text = series.astype("string")
    parsed = pd.to_datetime(text, errors="coerce", dayfirst=True)
    mask = parsed.isna() & text.notna()
    if mask.any():
        parsed.loc[mask] = pd.to_datetime(text.loc[mask], errors="coerce",
                                          format="%d-%b-%y")
    return parsed


def clean_company_a_amounts(df_erp, df_tool):
    df_erp = df_erp.copy()
    df_tool = df_tool.copy()
    erp_amount = df_erp["value_zar"].map(parse_number).fillna(0.0)
    df_erp["spend"] = erp_amount
    df_erp["include"] = (erp_amount != 0).astype(int)
    df_erp["exclusion_reason"] = ""
    erp_text = pd.concat([
        df_erp.get("material_short_text", pd.Series("", index=df_erp.index)).fillna("").astype(str),
        df_erp.get("service_short_text", pd.Series("", index=df_erp.index)).fillna("").astype(str),
        df_erp.get("supplier_name", pd.Series("", index=df_erp.index)).fillna("").astype(str),
    ], axis=1).agg(" ".join, axis=1).str.lower()
    erp_nonproc = erp_text.str.contains(
        r"\b(?:tax|vat|gst|intercompany|intercompany|royalt|partner bill|revenue|financ|interest)\b",
        regex=True,
    )
    df_erp.loc[erp_nonproc, ["include", "spend"]] = [0, 0.0]
    df_erp.loc[erp_nonproc, "exclusion_reason"] = "NON_PROCUREMENT"

    gross = df_tool["gross_amount"].map(parse_number).fillna(0.0)
    vat = df_tool["vat_rate"].map(parse_number)
    vat = vat.apply(lambda x: x / 100 if x is not None and x > 1 else x)
    vat = vat.fillna(0.15)
    df_tool["spend"] = gross / (1.0 + vat)

    status = df_tool["status"].astype("string").str.strip().str.lower()
    df_tool["include"] = status.isin(["approved", "paid"]).astype(int)
    df_tool["exclusion_reason"] = ""
    df_tool.loc[status.eq("cancelled"), "exclusion_reason"] = "CANCELLED"
    df_tool.loc[status.eq("cancelled"), "include"] = 0
    df_tool.loc[status.isin(["rejected", "on hold", "submitted", "disputed"]), "exclusion_reason"] = "NOT_APPROVED"
    df_tool.loc[df_tool["include"].eq(0) & df_tool["exclusion_reason"].eq(""), "exclusion_reason"] = "NOT_APPROVED"
    df_tool.loc[df_tool["include"].eq(0), "spend"] = 0.0
    return df_erp, df_tool


def clean_company_b(df_inv, df_fx=None, seed_labels=None):
    df = df_inv.copy()
    df["line_amount"] = df["line_amount"].map(parse_number)
    df["include"] = 1
    df["exclusion_reason"] = ""

    date_col = "invoice_date" if "invoice_date" in df else "date"
    df["invoice_date_dt"] = parse_dates(df[date_col]) if date_col in df else pd.NaT
    df["month"] = df["invoice_date_dt"].dt.to_period("M").astype("string")

    if df_fx is not None and "currency" in df:
        fx_map = df_fx.set_index(["currency", "month"])["units_per_usd"].to_dict()
    else:
        fx_map = {}

    def convert(row):
        amount = row["line_amount"]
        if pd.isna(amount):
            return 0.0
        curr = str(row.get("currency", "USD")).strip().upper()
        if curr in ("", "USD", "US DOLLAR"):
            return float(amount)
        rate = parse_number(row.get("currency_rate"))
        if rate is None or rate == 0:
            rate = fx_map.get((curr, str(row["month"]))) if pd.notna(row["month"]) else None
        if rate is None or rate == 0:
            return float(amount)
        return float(amount) / float(rate)

    df["spend"] = df.apply(convert, axis=1)

    excluded_gl = set()
    excluded_supplier_types = set()
    if seed_labels is not None:
        labels = seed_labels
        excluded = labels[labels["exclusion_reason"].fillna("").str.len() > 0]
        if "gl_account" in df and "gl_account" in excluded:
            excluded_gl.update(excluded["gl_account"].dropna().astype(str))
        if "supplier_type" in df and "supplier_type" in excluded:
            excluded_supplier_types.update(excluded["supplier_type"].dropna().astype(str).str.strip())

    text = pd.concat([
        df.get("line_description", pd.Series("", index=df.index)).fillna("").astype(str),
        df.get("gl_account_description", pd.Series("", index=df.index)).fillna("").astype(str),
    ], axis=1).agg(" ".join, axis=1).str.lower()
    nonproc = text.str.contains(
        r"\b(?:tax|vat|gst|intercompany|intercompany|royalt|partner bill|revenue|financing|interest|fx [wl]|exchange [wl])\b",
        regex=True,
    )
    if "gl_account_description" in df:
        nonproc |= df["gl_account_description"].fillna("").astype(str).str.lower().str.startswith("ic ")
        nonproc |= df["gl_account_description"].fillna("").astype(str).str.lower().str.contains("inter.?company")
    if "gl_account" in df:
        nonproc |= df["gl_account"].astype(str).isin(excluded_gl)
    if "supplier_type" in df:
        supplier_type = df["supplier_type"].fillna("").astype(str).str.strip()
        nonproc |= supplier_type.isin(excluded_supplier_types)

    df.loc[nonproc, ["include", "spend"]] = [0, 0.0]
    df.loc[nonproc, "exclusion_reason"] = "NON_PROCUREMENT"
    return df
