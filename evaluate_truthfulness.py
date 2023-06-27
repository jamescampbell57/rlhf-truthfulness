# %%
import requests


json_data = {
    'inputs': 'My name is James',
}

response = requests.post('https://l3r668fyrhayzneq.us-east-1.aws.endpoints.huggingface.cloud', headers=headers, json=json_data)

print(response.json())

# Note: json_data will not be serialized by requests
# exactly as it was in the original request.
#data = '{"inputs":"My name is Teven and I am"}'
#response = requests.post('https://l3r668fyrhayzneq.us-east-1.aws.endpoints.huggingface.cloud', headers=headers, data=data)