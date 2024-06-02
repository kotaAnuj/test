# agent.py

import requests
import logging

class InstructionAgent:
    def __init__(self, api_url, api_key):
        self.api_url = api_url
        self.headers = {"Authorization": f"Bearer {api_key}"}
        logging.basicConfig(level=logging.INFO)
    
    def query(self, payload):
        try:
            response = requests.post(self.api_url, headers=self.headers, json=payload)
            response.raise_for_status()
            result = response.json()
            if result:
                return self.format_response(result[0]['generated_text'])
            else:
                return "Sorry, I couldn't find any information."
        except requests.exceptions.HTTPError as http_err:
            logging.error(f"HTTP error occurred: {http_err}")
        except Exception as err:
            logging.error(f"An error occurred: {err}")
        return "An error occurred while processing your request."

    def format_response(self, response_text):
        response_lines = response_text.split('\n')
        bullet_points = "\n".join([f"- {line.strip()}" for line in response_lines if line.strip()])
        friendly_response = f"Hi there! Here is the information you requested:\n\n{bullet_points}\n\nHave a great day!"
        return friendly_response
