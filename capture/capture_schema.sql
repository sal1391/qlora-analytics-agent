-- Warehouse-side logging table for anonymized agent interactions.
-- Deploy this in a NON-production analytics/observability schema. It stores
-- only anonymized templates and behavioral metadata -- never raw PII or rows.
--
-- Dialect: Snowflake. Adjust types for other warehouses as needed.

CREATE TABLE IF NOT EXISTS AGENT_CAPTURE_LOG (
    capture_id                  STRING       DEFAULT UUID_STRING(),
    captured_at                 TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),

    -- Anonymized request (PII/literals stripped BEFORE insert)
    anonymized_question_template STRING      NOT NULL,
    question_hash               STRING,           -- SHA-256 (first 16 chars)

    -- Routing
    skill_selected              STRING,           -- e.g. SQL_ANALYST
    skill_correct               STRING,           -- human label, nullable

    -- SQL (literals replaced with ? placeholders)
    sql_generated               STRING,
    sql_final                   STRING,
    execution_success           BOOLEAN,

    -- Safety / guardrails
    safety_flags                ARRAY,            -- e.g. ['write_blocked']
    clarification_needed        BOOLEAN,

    -- Context + performance
    schema_version              STRING,           -- version tag only
    latency_ms                  NUMBER(10,2),
    user_feedback               STRING,           -- 'up' | 'down' | rating | NULL

    -- Provenance
    source                      STRING DEFAULT 'production_capture'
);

-- Helper view: labeled, review-ready rows for training-data curation.
CREATE OR REPLACE VIEW AGENT_CAPTURE_LABELED AS
SELECT
    capture_id,
    anonymized_question_template,
    COALESCE(skill_correct, skill_selected) AS skill,
    sql_final,
    execution_success,
    schema_version
FROM AGENT_CAPTURE_LOG
WHERE skill_correct IS NOT NULL;

-- Example READ-ONLY export of anonymized templates for offline curation.
-- (No row-level business data is ever selected here.)
SELECT
    anonymized_question_template,
    skill,
    sql_final,
    schema_version
FROM AGENT_CAPTURE_LABELED
LIMIT 1000;
