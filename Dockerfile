FROM python:3.12

WORKDIR /usr/src/app

COPY requirements.txt requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .


CMD [ "python3", "experiments.py" ]