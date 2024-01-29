# TennisBot
Custom bot specialized in tennis knowledge

## Overview
This tennis bot uses OpenAI chatGPT API to access historical knowledge about tennis. Since ChatGPT model was trained last in 2021, we also capture recent tennis information by using the wikipedia API.

## Setup
The project needs to access the OpenAI ChatGPT API key. This API key is stored in the .env file. **Please enter your Open AI key in the .env file before running the project**.

## Project information
The project is composed of a Jupyter Notebook which describes step-by-step how we built the custom chatbot.

We then also created a stand alone tennis chatbot which can be activated by using the following command line:
> python tennisbot.py