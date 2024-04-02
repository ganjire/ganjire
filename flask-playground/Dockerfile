# Usage
# docker build -t ganjireb/flask-playground /Users/rebecca/Desktop/ZHAW/Frühlingssemester2024/azure2/flask-playground
# docker run -p 9001:5000 -d ganjireb/flask-playground:latest

FROM python:3.10.11

# Copy Files
WORKDIR /usr/src/app
COPY . .

# Install
RUN pip install -r requirements.txt

# Docker Run Command
EXPOSE 5000
CMD [ "python", "-m" , "flask", "run", "--host=0.0.0.0"]