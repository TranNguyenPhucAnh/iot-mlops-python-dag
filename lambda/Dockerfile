FROM public.ecr.aws/lambda/python:3.12

# Cài đặt thư viện inference nhẹ
RUN pip install "scikit-learn==1.6.1" joblib dill boto3 pandas

# Copy mã nguồn xử lý và model file
COPY app.py ${LAMBDA_TASK_ROOT}
COPY model.pkl ${LAMBDA_TASK_ROOT}
COPY scaler.pkl ${LAMBDA_TASK_ROOT}

CMD [ "app.handler" ]
