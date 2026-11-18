# Decisions & Assumptions

1. **A VAT:** ERP values are net. Tool gross amounts are divided by `1 + VAT`;
   blank or invalid VAT defaults to 15%.
2. **A status:** Paid and approved count. Cancelled becomes `CANCELLED`; rejected,
   on hold, submitted and disputed become `NOT_APPROVED`.
3. **A matching:** ERP PO numbers and normalized tool PO references are matched,
   with leading-zero-insensitive digit normalization. Repeated tool invoice exports
   are marked `DUPLICATE`; the ERP record wins when both systems describe the purchase.
4. **B currency:** USD is unchanged. A supplied currency rate is used when present;
   otherwise the matching currency/month FX table row is used.
5. **B exclusions:** Observed seed labels learn excluded GL accounts and supplier
   types. Intercompany, tax, VAT, royalty, partner, revenue and financing text is
   additionally classified as `NON_PROCUREMENT`.
6. **Impossible values:** Unparseable amounts become zero. A zero-spend row is
   excluded rather than counted; impossible dates remain missing.
7. **Supplier and item keys:** Keys are stable within a run and based on canonical
   supplier fields or description similarity, not display names.
