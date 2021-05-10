curl http://127.0.0.1:5000/invocations -H 'Content-Type: application/json' -d '{
    "columns": ["num_pages", "min_num_annot", "max_num_annot", "max_annot_length", "temperature"],
    "data": [[1, 20, 30, 100, 1.0]]
}'