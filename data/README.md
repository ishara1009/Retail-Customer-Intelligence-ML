
# Dataset

These small CSVs contain fictional, synthetic retail activity; they do not
describe real people.

- `raw/customers.csv`: 1,000 customers (`customer_id`, `signup_date`).
- `raw/retail_transactions.csv`: 22,135 orders with purchase dates and LKR amounts.
- `processed/customer_monthly_ml.csv`: 12,000 customer-month rows used by the models.

One processed row describes what was known immediately before `snapshot_date`
and what happened during the target month beginning on that date. For example,
at `2025-04-01`, the 90-day history is `[2025-01-01, 2025-04-01)` and the target
window is `[2025-04-01, 2025-05-01)`. Thus “next month” means the target month
immediately after the historical observation window.

Historical inputs are tenure, 90-day order/spend summaries, recency, and a
previous-purchase indicator. Targets are `next_month_spend_lkr` and
`will_purchase_next_month`. A blank recency means no purchase existed before the
snapshot; it is not corrupt data.

The supplied CSV files are preserved. Because the original random-data generator
and its seed were not included, `scripts/generate_dataset.py` safely verifies and
preserves the supplied processed table by default. `--force-rebuild` reconstructs
only that table from the rounded raw amounts; a few derived values can differ by
LKR 0.01 from calculations made before the raw values were rounded.
