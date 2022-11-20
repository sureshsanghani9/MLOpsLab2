FROM python:3.9
COPY ./requirements.txt /exp1/requirements.txt
RUN pip3 install --no-cache-dir -r /exp1/requirements.txt
COPY ./*.py /exp1/
WORKDIR /exp1
#RUN export FLASK_APP=app.py
#RUN flask run
#CMD ["python3", "app.py"]