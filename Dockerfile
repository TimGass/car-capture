FROM python:3.12.3-slim as base

WORKDIR /home/app

ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONFAULTHANDLER 1
RUN apt-get update && apt-get install -y build-essential ffmpeg libsm6 libxext6

COPY Pipfile .
COPY Pipfile.lock .
RUN python -m pip install --upgrade pip
RUN python -m pip install cmake
RUN python -m pip install pipenv
RUN pipenv install

# Install application into container
COPY . .

# Run the application
EXPOSE 5000
CMD ["pipenv", "run", "start"]