-- Berka / PKDD'99 source profiling
-- Run against data/processed/berka.sqlite
--
-- Purpose:
--   1) Verify row coverage across the relational source tables.
--   2) Check transaction date coverage.
--   3) Inspect customer-initiated transaction behaviour used by the churn model.

-- 1. Source table row counts
SELECT 'account' AS table_name, COUNT(*) AS row_count FROM account
UNION ALL
SELECT 'card', COUNT(*) FROM card
UNION ALL
SELECT 'client', COUNT(*) FROM client
UNION ALL
SELECT 'disp', COUNT(*) FROM disp
UNION ALL
SELECT 'district', COUNT(*) FROM district
UNION ALL
SELECT 'loan', COUNT(*) FROM loan
UNION ALL
SELECT 'order_table', COUNT(*) FROM order_table
UNION ALL
SELECT 'trans', COUNT(*) FROM trans
ORDER BY table_name;


-- 2. Transaction coverage
SELECT
    COUNT(*) AS transaction_rows,
    COUNT(DISTINCT account_id) AS accounts_with_transactions,
    MIN(date) AS first_transaction_date_raw,
    MAX(date) AS last_transaction_date_raw
FROM trans;


-- 3. Transaction operation distribution
SELECT
    COALESCE(operation, 'NULL') AS operation,
    COUNT(*) AS transaction_count,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS pct_of_transactions
FROM trans
GROUP BY COALESCE(operation, 'NULL')
ORDER BY transaction_count DESC;


-- 4. Customer-initiated activity by account over the full source period
-- Customer-initiated operations mirror the behavioural target definition:
-- VYBER, PREVOD NA UCET, VYBER KARTOU, VKLAD.
WITH customer_activity AS (
    SELECT
        account_id,
        COUNT(*) AS customer_tx_count,
        SUM(amount) AS customer_tx_amount,
        COUNT(DISTINCT date) AS active_days
    FROM trans
    WHERE operation IN (
        'VYBER',
        'PREVOD NA UCET',
        'VYBER KARTOU',
        'VKLAD'
    )
    GROUP BY account_id
)
SELECT
    COUNT(*) AS active_accounts,
    ROUND(AVG(customer_tx_count), 2) AS avg_customer_tx_count,
    ROUND(AVG(customer_tx_amount), 2) AS avg_customer_tx_amount,
    ROUND(AVG(active_days), 2) AS avg_active_days
FROM customer_activity;
