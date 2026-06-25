SELECT
  id_user
  , DATE(session_start) AS session_date
  , SUM(TIMESTAMP_DIFF(session_end, session_start, SECOND) / 3600.0) AS total_hours
FROM (
  SELECT
    id_user
    , action
    , timestamp_action AS session_start
    , LEAD(timestamp_action) OVER (PARTITION BY id_user ORDER BY timestamp_action) AS session_end
  FROM `data.1`
)
WHERE
  action = 'open'
  AND session_end IS NOT NULL
  AND session_start >= TIMESTAMP_SUB((SELECT MAX(timestamp_action) FROM `data.1`), INTERVAL 10 DAY)
GROUP BY id_user, session_date
ORDER BY total_hours DESC, id_user, session_date;