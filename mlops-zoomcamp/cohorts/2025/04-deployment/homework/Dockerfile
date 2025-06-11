FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

WORKDIR /app

COPY starter.py yellow_tripdata_2023-04.parquet yellow_tripdata_2023-05.parquet ./

RUN pip install pandas pyarrow scikit-learn==1.5.0

ENTRYPOINT ["python", "starter.py"]
