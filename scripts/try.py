import requests

csv_data = """
7162,1.46,21.0,50.0,1.0,30.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0
61578,4.4,6.0,39.0,1.0,30.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0
60446,2.24,7.0,19.0,1.0,30.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0
3377,0.72,22.0,52.0,1.0,30.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0
5205,1.01,22.0,24.0,1.0,30.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0
"""

# Process CSV data into a list of lists of floats
data_list = []
for line in csv_data.strip().split("\n"):
    # Convert each line into a list of floats
    float_values = list(map(float, line.split(",")))
    data_list.append(float_values)

# Define the API URL
api_url = "http://127.0.0.1:8000/api/v1/predict"

# Define the request payload
payload = {"data": data_list}

# Send the POST request to the API
response = requests.post(api_url, json=payload)

# Print the response from the API
if response.status_code == 200:
    print("Predictions:", response.json())
else:
    print("Failed to get predictions. Status code:", response.status_code)
    print("Response:", response.text)
