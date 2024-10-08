# Grafana Queries for Dashboard Creation

This document outlines the SQL queries used to create the Grafana dashboards for analyzing data from the application.

## 1. Distribution of Topic in the Questions asked
```sql
WITH unnested_topics AS (
    SELECT 
        TRIM(unnest(string_to_array(trim(both '[]' from topic), ','))) AS topic,
        COUNT(*) AS occurrence
    FROM conversations
    GROUP BY topic
)
SELECT 
    topic,
    SUM(occurrence) AS total_occurrences
FROM unnested_topics 
GROUP BY topic
ORDER BY total_occurrences DESC;

```

## 2. Top 5 highest response time (in ms)

```sql
select selected_talk,time_taken 
from conversations
order by time_taken desc
limit 5;
```

## 3. Average Time Taken (in ms) for Retrieval Across Topics
```sql
WITH unnested_topics AS (
    SELECT 
        TRIM(unnest(string_to_array(trim(both '[]' from topic), ','))) AS topic,
        LENGTH(answer) AS answer_length
    FROM conversations
)
SELECT 
    topic,
    AVG(answer_length) as avg_time_taken
FROM unnested_topics 
GROUP BY topic
ORDER BY avg_time_taken DESC;
```

## 4. Average length of answers across topics
```sql
WITH unnested_topics AS (
    SELECT 
        TRIM(unnest(string_to_array(trim(both '[]' from topic), ','))) AS topic,
        LENGTH(answer) AS answer_length
    FROM conversations
)

SELECT 
    topic,
    AVG(answer_length) AS avg_answer_length
FROM unnested_topics
GROUP BY topic
ORDER BY avg_answer_length desc
-- limit 5
;

```

## 5. Feedback ratio across topics
```sql
SELECT 
    unnest(string_to_array(trim(both '[]' from t.topic), ',')) as topic,
    AVG(CASE WHEN f.feedback = 1 THEN 1 ELSE 0 END) as positive_feedback_ratio,
    COUNT(*) as total_feedback
FROM conversations t
JOIN feedback f ON t.id = f.conversation_id
GROUP BY topic
ORDER BY positive_feedback_ratio DESC
LIMIT 10;
```