#!/usr/bin/env python3
import os
import argparse
import google.generativeai as genai
import config


def configure_gemma():
    genai.configure(api_key=config.GEMINI_API_KEY)

    # Define generation settings for the model
    generation_config = {
        "temperature": 0.4,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 1024,
        "response_mime_type": "text/plain",
    }

    model = genai.GenerativeModel(
        model_name="gemini-2.0-flash",
        generation_config=generation_config,
    )
    chat_session = model.start_chat(history=[])
    return chat_session


def build_prompt(a, b, operation):
    operator_map = {"add": "+", "sub": "-", "mul": "*", "div": "/"}
    operator = operator_map[operation]
    expression = f"{a} {operator} {b}"

    # Construct the prompt instructing the LLM to act as a calculator
    prompt = (
        f"You are a calculator. Evaluate the expression: {expression}. "
        "Provide only the final numerical result."
    )
    return prompt


def main():
    # Set up CLI argument parsing
    parser = argparse.ArgumentParser(description="CLI Calculator using Gemma API")
    parser.add_argument("a", type=float, help="First number")
    parser.add_argument("b", type=float, help="Second number")
    parser.add_argument(
        "operation",
        choices=["add", "sub", "mul", "div"],
        help="Operation to perform: add, sub, mul, or div",
    )
    args = parser.parse_args()

    # Initialize Gemma API chat session
    chat_session = configure_gemma()

    # Build the prompt based on input numbers and the selected operation
    prompt = build_prompt(args.a, args.b, args.operation)

    # Send the prompt to Gemma API and print the result
    response = chat_session.send_message(prompt)
    print(response.text)


if __name__ == "__main__":
    main()
