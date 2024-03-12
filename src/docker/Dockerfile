FROM python:3.10.4-slim 

#Set environment vars 
#ENV PYTHONDONTWRITEBYTECODE 1
#ENV PYTHONBUFFERED 1 

#Set working directory
WORKDIR /code


#Get requirements
COPY requirements.txt /code/requirements.txt

#Install dependencies 
#RUN pip install --upgrade -r /code/requirements.txt

#FOR DEPENDENCIES WITH HEAVY WHEELS
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

#Copy Code 
COPY ./src /code 