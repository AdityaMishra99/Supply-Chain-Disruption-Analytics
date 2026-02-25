## Most Common Disruption Types
SELECT disruption_type, COUNT(*) 
FROM supply_chain
GROUP BY disruption_type
ORDER BY COUNT(*) DESC;

## Average Recovery Time by Severity
SELECT disruption_severity,
       AVG(full_recovery_days) AS avg_recovery_days
FROM supply_chain
GROUP BY disruption_severity
ORDER BY disruption_severity;

## Revenue Loss by Industry
SELECT industry,
       SUM(revenue_loss_usd) AS total_loss
FROM supply_chain
GROUP BY industry
ORDER BY total_loss DESC;

## Impact of Backup Suppliers
SELECT has_backup_supplier,
       AVG(full_recovery_days) AS avg_recovery
FROM supply_chain
GROUP BY has_backup_supplier;

## Tier Cascade Effect
SELECT supplier_tier,
       AVG(revenue_loss_usd) AS avg_loss
FROM supply_chain
GROUP BY supplier_tier
ORDER BY supplier_tier;

## WINDOW FUNCTIONS (ADVANCED SQL ANALYTICS)

## Which disruptions caused the highest losses per industry?
SELECT
    industry,
    disruption_type,
    revenue_loss_usd,
    RANK() OVER (
        PARTITION BY industry
        ORDER BY revenue_loss_usd DESC
    ) AS loss_rank
FROM supply_chain;

## Cumulative Revenue Loss Over Time (Simulation)
SELECT
    disruption_id,
    industry,
    revenue_loss_usd,
    SUM(revenue_loss_usd) OVER (
        PARTITION BY industry
        ORDER BY disruption_id
    ) AS cumulative_loss
FROM supply_chain;

## Tier-wise Recovery Comparison (Average vs Individual)
## Is this supplier recovering slower or faster than its tier average?
SELECT
    supplier_tier,
    full_recovery_days,
    AVG(full_recovery_days) OVER (
        PARTITION BY supplier_tier
    ) AS tier_avg_recovery,
    full_recovery_days -
    AVG(full_recovery_days) OVER (
        PARTITION BY supplier_tier
    ) AS deviation_from_avg
FROM supply_chain;

## Backup Supplier Impact (Before vs After)
SELECT
    has_backup_supplier,
    full_recovery_days,
    AVG(full_recovery_days) OVER (
        PARTITION BY has_backup_supplier
    ) AS avg_recovery
FROM supply_chain;

## Identify High-Risk Disruptions (Percentile)
SELECT
    disruption_id,
    revenue_loss_usd,
    NTILE(10) OVER (
        ORDER BY revenue_loss_usd DESC
    ) AS loss_decile
FROM supply_chain;

## STORED PROCEDURES (ENTERPRISE LEVEL)
CREATE OR REPLACE PROCEDURE industry_risk_summary()
LANGUAGE plpgsql
AS $$
BEGIN
    SELECT
        industry,
        COUNT(*) AS total_disruptions,
        ROUND(AVG(full_recovery_days),2) AS avg_recovery_days,
        SUM(revenue_loss_usd) AS total_revenue_loss
    FROM supply_chain
    GROUP BY industry
    ORDER BY total_revenue_loss DESC;
END;
$$;

## Stored Procedure: Supplier Tier Risk Analyzer
CREATE OR REPLACE PROCEDURE tier_risk_analysis()
LANGUAGE plpgsql
AS $$
BEGIN
    SELECT
        supplier_tier,
        COUNT(*) AS disruptions,
        AVG(disruption_severity) AS avg_severity,
        AVG(full_recovery_days) AS avg_recovery,
        SUM(revenue_loss_usd) AS total_loss
    FROM supply_chain
    GROUP BY supplier_tier
    ORDER BY supplier_tier;
END;
$$;

## Stored Procedure: High Severity Alert System
CREATE OR REPLACE PROCEDURE critical_disruptions(
    severity_threshold INT
)
LANGUAGE plpgsql
AS $$
BEGIN
    SELECT
        disruption_id,
        industry,
        disruption_type,
        disruption_severity,
        revenue_loss_usd,
        full_recovery_days
    FROM supply_chain
    WHERE disruption_severity >= severity_threshold
    ORDER BY revenue_loss_usd DESC;
END;
$$;

## Stored Procedure: Backup Supplier ROI Analysis
CREATE OR REPLACE PROCEDURE backup_supplier_roi()
LANGUAGE plpgsql
AS $$
BEGIN
    SELECT
        has_backup_supplier,
        COUNT(*) AS disruptions,
        AVG(full_recovery_days) AS avg_recovery,
        AVG(revenue_loss_usd) AS avg_loss
    FROM supply_chain
    GROUP BY has_backup_supplier;
END;
$$;


## SQL-BASED SCENARIO SIMULATION (Baseline vs Scenario)
## Create Baseline Metrics View
CREATE OR REPLACE VIEW baseline_metrics AS
SELECT
    industry,
    supplier_tier,
    AVG(full_recovery_days) AS avg_recovery_days,
    AVG(revenue_loss_usd) AS avg_revenue_loss
FROM supply_chain
GROUP BY industry, supplier_tier;

## What if disruptions become more severe?
CREATE OR REPLACE VIEW severity_increase_scenario AS
SELECT
    industry,
    supplier_tier,
    AVG(full_recovery_days * 1.2) AS simulated_recovery_days,
    AVG(revenue_loss_usd * 1.25) AS simulated_revenue_loss
FROM supply_chain
GROUP BY industry, supplier_tier;

## Scenario: Backup Supplier Added
CREATE OR REPLACE VIEW backup_supplier_scenario AS
SELECT
    industry,
    supplier_tier,
    AVG(
        CASE 
            WHEN has_backup_supplier = FALSE
            THEN full_recovery_days * 0.7
            ELSE full_recovery_days
        END
    ) AS simulated_recovery_days,
    AVG(
        CASE 
            WHEN has_backup_supplier = FALSE
            THEN revenue_loss_usd * 0.8
            ELSE revenue_loss_usd
        END
    ) AS simulated_revenue_loss
FROM supply_chain
GROUP BY industry, supplier_tier;

## Scenario: Faster Response Time
CREATE OR REPLACE VIEW faster_response_scenario AS
SELECT
    industry,
    supplier_tier,
    AVG(full_recovery_days * 0.85) AS simulated_recovery_days,
    AVG(revenue_loss_usd * 0.9) AS simulated_revenue_loss
FROM supply_chain
GROUP BY industry, supplier_tier;

## Build Supply Chain Resilience Index
## STEP 1: NORMALIZATION (CRITICAL)
CREATE OR REPLACE VIEW normalized_metrics AS
SELECT
    disruption_id,
    industry,
    supplier_tier,
    supplier_region,

    1 - (
        disruption_severity - MIN(disruption_severity) OVER ()
    ) / (
        MAX(disruption_severity) OVER () - MIN(disruption_severity) OVER ()
    ) AS severity_score,

    1 - (
        production_impact_pct - MIN(production_impact_pct) OVER ()
    ) / (
        MAX(production_impact_pct) OVER () - MIN(production_impact_pct) OVER ()
    ) AS impact_score,

    1 - (
        full_recovery_days - MIN(full_recovery_days) OVER ()
    ) / (
        MAX(full_recovery_days) OVER () - MIN(full_recovery_days) OVER ()
    ) AS recovery_score,

    CASE
        WHEN has_backup_supplier THEN 1
        ELSE 0
    END AS backup_score
FROM supply_chain;

## Create Final Resilience Index View
CREATE OR REPLACE VIEW resilience_index AS
SELECT
    industry,
    supplier_tier,
    supplier_region,

    ROUND(
        (
            severity_score * 0.30 +
            impact_score * 0.25 +
            recovery_score * 0.30 +
            backup_score * 0.15
        )::NUMERIC * 100,
        2
    ) AS resilience_index
FROM normalized_metrics;

## AGGREGATIONS FOR INSIGHTS
## Resilience by Industry
SELECT
    industry,
    ROUND(AVG(resilience_index),2) AS avg_resilience
FROM resilience_index
GROUP BY industry
ORDER BY avg_resilience DESC;

## Resilience by Supplier Tier
SELECT
    supplier_tier,
    ROUND(AVG(resilience_index),2) AS avg_resilience
FROM resilience_index
GROUP BY supplier_tier
ORDER BY supplier_tier;

## Identify Fragile Suppliers
SELECT *
FROM resilience_index
WHERE resilience_index < 40
ORDER BY resilience_index;

## CREATE ALERTS TABLE
CREATE TABLE disruption_alerts (
    alert_id SERIAL PRIMARY KEY,
    disruption_id BIGINT,
    alert_type VARCHAR(50),
    alert_message TEXT,
    alert_severity VARCHAR(20),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

## TRIGGER FUNCTION (CORE LOGIC)
CREATE OR REPLACE FUNCTION disruption_alert_trigger()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
DECLARE
    resilience_score NUMERIC;
BEGIN
    -- Calculate resilience score (same logic as view)
    resilience_score :=
        (
            NEW.disruption_severity * -0.30 +
            NEW.production_impact_pct * -0.25 +
            NEW.full_recovery_days * -0.30 +
            (CASE WHEN NEW.has_backup_supplier THEN 1 ELSE 0 END) * 0.15
        );

    -- High severity alert
    IF NEW.disruption_severity >= 8 THEN
        INSERT INTO disruption_alerts (
            disruption_id, alert_type, alert_message, alert_severity
        )
        VALUES (
            NEW.disruption_id,
            'High Severity',
            'Critical disruption detected with high severity',
            'CRITICAL'
        );
    END IF;

    -- Slow recovery alert
    IF NEW.full_recovery_days > 60 THEN
        INSERT INTO disruption_alerts (
            disruption_id, alert_type, alert_message, alert_severity
        )
        VALUES (
            NEW.disruption_id,
            'Slow Recovery',
            'Recovery time exceeds acceptable threshold',
            'HIGH'
        );
    END IF;

    -- No backup supplier alert
    IF NEW.has_backup_supplier = FALSE THEN
        INSERT INTO disruption_alerts (
            disruption_id, alert_type, alert_message, alert_severity
        )
        VALUES (
            NEW.disruption_id,
            'No Backup Supplier',
            'Supplier lacks backup source',
            'MEDIUM'
        );
    END IF;

    RETURN NEW;
END;
$$;

## CREATE THE TRIGGER
CREATE TRIGGER disruption_alerts_trigger
AFTER INSERT OR UPDATE
ON supply_chain
FOR EACH ROW
EXECUTE FUNCTION disruption_alert_trigger();

## Create Predictive Risk View
CREATE OR REPLACE VIEW predictive_risk_model AS
SELECT
    disruption_id,
    industry,
    supplier_region,
    supplier_tier,

    -- Normalized risk inputs
    disruption_severity,
    production_impact_pct,
    full_recovery_days,
    revenue_loss_usd,
    has_backup_supplier,

    -- Risk Score (0–1 scale)
    ROUND(
        (
            disruption_severity * 0.25 +
            production_impact_pct * 0.20 +
            (full_recovery_days / 100.0) * 0.20 +
            (revenue_loss_usd / 10000000.0) * 0.15 +
            (CASE WHEN has_backup_supplier = FALSE THEN 0.20 ELSE 0 END)
        )::NUMERIC,
        4
    ) AS predicted_risk_score

FROM supply_chain;

## Classify Risk Levels
CREATE OR REPLACE VIEW early_warning_flags AS
SELECT
    *,
    CASE
        WHEN predicted_risk_score >= 0.75 THEN 'CRITICAL'
        WHEN predicted_risk_score >= 0.50 THEN 'HIGH'
        WHEN predicted_risk_score >= 0.30 THEN 'MEDIUM'
        ELSE 'LOW'
    END AS risk_category
FROM predictive_risk_model;

## Add Rolling Risk Trend (Window Function)
CREATE OR REPLACE VIEW rolling_risk_trend AS
SELECT
    disruption_id,
    industry,
    supplier_region,
    predicted_risk_score,

    AVG(predicted_risk_score)
        OVER (
            PARTITION BY industry
            ORDER BY disruption_id
            ROWS BETWEEN 3 PRECEDING AND CURRENT ROW
        ) AS rolling_4_event_avg

FROM predictive_risk_model;

## Auto-Insert Early Warning Alerts
CREATE TABLE IF NOT EXISTS early_warning_alerts (
    alert_id SERIAL PRIMARY KEY,
    disruption_id BIGINT,
    predicted_risk_score NUMERIC,
    risk_category VARCHAR(20),
    alert_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

## Predictive Trigger Function
CREATE OR REPLACE FUNCTION predictive_warning_trigger()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
DECLARE
    risk_score NUMERIC;
BEGIN

    -- Compute risk score inline
    risk_score :=
        NEW.disruption_severity * 0.25 +
        NEW.production_impact_pct * 0.20 +
        (NEW.full_recovery_days / 100.0) * 0.20 +
        (NEW.revenue_loss_usd / 10000000.0) * 0.15 +
        (CASE WHEN NEW.has_backup_supplier = FALSE THEN 0.20 ELSE 0 END);

    -- Insert alert if high probability
    IF risk_score >= 0.50 THEN
        INSERT INTO early_warning_alerts (
            disruption_id,
            predicted_risk_score,
            risk_category,
            alert_message
        )
        VALUES (
            NEW.disruption_id,
            ROUND(risk_score, 4),
            CASE
                WHEN risk_score >= 0.75 THEN 'CRITICAL'
                ELSE 'HIGH'
            END,
            'Predictive model indicates elevated disruption risk'
        );
    END IF;

    RETURN NEW;
END;
$$;

##
DROP TRIGGER IF EXISTS predictive_warning_trigger
ON supply_chain;

CREATE TRIGGER predictive_warning_trigger
AFTER INSERT OR UPDATE
ON supply_chain
FOR EACH ROW
EXECUTE FUNCTION predictive_warning_trigger();
