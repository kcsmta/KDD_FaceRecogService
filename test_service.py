import requests

url = "http://localhost:8000/predict"

files = [('file', open('0.jpg','rb'))]


response = requests.request("POST", url, files = files)

print(response.text.encode('utf8'))

