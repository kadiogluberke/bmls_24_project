import requests
import random

# Define the API URL
api_url = "http://127.0.0.1:8000/api/v1/predict"

# Generate random input data
random_input_data = [[random.uniform(0, 10) for _ in range(10)] for _ in range(5)]

# Define the request payload
payload = {"data": random_input_data}

# Send the POST request to the API
response = requests.post(api_url, json=payload)

# Print the response from the API
if response.status_code == 200:
    print("Predictions:", response.json())
else:
    print("Failed to get predictions. Status code:", response.status_code)
    print("Response:", response.text)
