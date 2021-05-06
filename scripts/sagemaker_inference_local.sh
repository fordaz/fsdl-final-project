curl http://127.0.0.1:5000/invocations -H 'Content-Type: application/json' -d '{
    "columns": ["num_samples", "sample_size", "temperature"],
    "data": [[3, 50, 1.0]]
}'