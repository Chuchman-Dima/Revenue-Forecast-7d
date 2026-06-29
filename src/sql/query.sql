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
    , LEAD(action) OVER(PARTITION BY id_user ORDER BY timestamp_action) AS next_action
  FROM `data.task_1`
  WHERE timestamp_action >= TIMESTAMP_SUB((SELECT MAX(timestamp_action) FROM `data.task_1`), INTERVAL 11 DAY)
)
WHERE
  action = 'open'
  AND next_action = 'close'
  AND session_end IS NOT NULL
  -- Проблема з 10 днем пофікшена
  AND DATE(session_start) >= DATE_SUB(DATE((SELECT MAX(timestamp_action) FROM `data.task_1`)), INTERVAL 9 DAY)
GROUP BY id_user, session_date
ORDER BY id_user, session_date;