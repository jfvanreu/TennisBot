import os
import tiktoken
from openai import OpenAI
import pandas as pd
import numpy as np
import ast

# define models and constants
EMBEDDING_MODEL_NAME = "text-embedding-ada-002"
COMPLETION_MODEL_NAME="gpt-3.5-turbo-instruct"
MAX_PROMPT_TOKENS=1800
MAX_RESPONSE_TOKENS=150
TIKTOKEN_ENCODING="cl100k_base"

# load dataframe with latest tennis news/data
df = pd.read_csv('embeddings_tennis_2022_2024.csv')
# embeddings are read as a string, so we need to convert them to a list of floats. read_csv issue.
df["embeddings"] = df["embeddings"].apply(lambda x : ast.literal_eval(x))

# load OPENAI API key from .env file
openai_api_key = os.getenv("OPENAI_API_KEY")

# create session with openai
client = OpenAI(api_key=openai_api_key)

def num_tokens_from_string(string):
    """Helper function which returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(TIKTOKEN_ENCODING)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def get_embedding(text, model=EMBEDDING_MODEL_NAME): 
    return client.embeddings.create(input = [text], model=model).data[0].embedding

def get_rows_sorted_by_relevance(question, df):
    """
    Function which takes in a question string and a dataframe containing
    rows of text and associated embeddings, and returns that dataframe
    sorted from least to most relevant for that question
    """

    # Get embeddings for the question text
    question_embeddings = get_embedding(question, model=EMBEDDING_MODEL_NAME)

    # Make a copy of the dataframe and add a "distances" column containing
    # the cosine distances between each row's embeddings and the
    # embeddings of the question

    df_copy = df.copy()
    
    # compute cosine similarities between each dataframe row embedding and the question embedding to identify most relevants rows.
    df_copy["distances"] = df.embeddings.apply(lambda x: 1-cosine_similarity(x, question_embeddings))

    # Sort the copied dataframe by the distances and return it
    # (shorter distance = more relevant so we sort in ascending order)
    df_copy.sort_values("distances", ascending=True, inplace=True)
    return df_copy

def create_prompt(question, df, max_token_count):
    """
    Given a question and a dataframe containing rows of text and their
    embeddings, return a text prompt to send to a Completion model
    """
    # Count the number of tokens in the prompt template and question
    prompt_template = """
        You are a tennis expert. Answer the question and if the question can't be answered based on the context, provide a response which starts with "You can't be serious!".
        When the question is "quit", answer with a tennis joke.

        Context:

        {} 

        ---
        Question: {}
        Answer:
    """
    current_token_count = num_tokens_from_string(prompt_template) + num_tokens_from_string(question)

    # add context to the question
    context = []
    
    for text in get_rows_sorted_by_relevance(question, df)["text"].values:

        # Increase the counter based on the number of tokens in this row
        text_token_count = num_tokens_from_string(text)
        current_token_count += text_token_count

        # Add the row of text to the list if we haven't exceeded the max
        if current_token_count <= max_token_count:
            context.append(text)
        else:
            break
    
    prompt=prompt_template.format("\n\n###\n\n".join(context), question)
    return prompt

def answer_question(question, df, max_prompt_tokens=MAX_PROMPT_TOKENS, max_answer_tokens=MAX_RESPONSE_TOKENS):
    """
    Given a question, a dataframe containing rows of text, and a maximum
    number of desired tokens in the prompt and response, return the
    answer to the question according to an OpenAI Completion model

    If the model produces an error, print the error and return an error message
    """
    prompt = create_prompt(question, df, max_prompt_tokens)

    try:
        response = client.completions.create(
            model=COMPLETION_MODEL_NAME,
            prompt=prompt,
            max_tokens=max_answer_tokens
        )
        return response.choices[0].text.strip()
    except Exception as e:
        print(e)
        return "Oops! Something went wrong..."

def main():
    question = ""
    while question != "quit":
        # ask question to user
        question = input("Please ask your question or type in 'quit': ")
        tennis_answer = answer_question(question, df, MAX_PROMPT_TOKENS, MAX_RESPONSE_TOKENS)
        # display openai answer to the question
        print(tennis_answer)

if __name__ == "__main__":
    main()
