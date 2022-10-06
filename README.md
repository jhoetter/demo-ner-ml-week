## About this project
This project was created for a demonstration at the machine learning week 2022 in Berlin. In this demonstration, we show how to create an a NER model using the Kern AI refinery as well as our SDK and sequence-learn library. We then pack up our custom model with Truss, a library to make deployment to the cloud much easier. Truss is created by Baseten

## Setup
Clone this repository, then install all the requirements with:
```
pip install -r requirements.txt
```
In this repository, you'll find a Python Notebook called truss-creation.ipynb. There you'll find code to log into the Kern AI refinery and access the data or your project. You can also upload the already finished ner-project.zip that accompanies this repository. 

## Creating a Truss
Truss supports all major machine learning frameworks. Our sequence-learn is not (yet) supported. Luckily, Truss allows us to pack up custom models, too. That's why we included a prepared Turss with the name ner-truss. If you follow along with the truss-creation.ipynb and create a CRFtagger, it will automatically save the model in the right location inside the Truss. The Truss is structured like this:
```
config.yaml
examples.yaml
data/
    <data> 
    <serialized model> # <-- model is saved here
model/
    __init__.py
    model.py
```

After the model is saved to the Truss, you can try out if the Truss is working by either using:
```python
tr = truss.from_directory(truss_path)
tr.server_predict({"inputs": embedder.transform(sentence)})
```
or
```python
tr = truss.from_directory(truss_path)
tr.docker_predict({"inputs": embedder.transform(sentence)})
```

Use the latter if you want to test of the model is working correctly from within a Docker container. This method is only used for testing purposes.
To build a Docker image from the Truss, run: 
```bash
truss build-image ner-truss
```

## Deploying the model to the cloud
The container image can then be deployed to any cloud platform that can run Docker containers (like AWS, GCP or Azure). In our demo, we are going to deploy to Microsoft Azure cloud using Azure Container Instances (ACI).

If you want to find out how to deploy a Truss/ Docker container to Azure, check out [this video here](https://www.youtube.com/watch?v=HM8roUY1oaE)!

Once a container instance is deployed to the cloud, we can embed it in already existing services or get predictions from the Container Instance itself. The following code snipped sends out a texts input in the JSON format using the Python requests library. The response are the predictions that the model gave:
```python
# Call the running NER model from Azure container instance
import json 
import requests

sentence = ["This is a sample sentence"]
input_dict = {"inputs": embedder.transform(sentence)}
with open("data.json", "w") as fp:
    json.dump(input_dict, fp)

headers = {
    "Content-Type": "application/json"
}

with open("data.json") as f:
    data = f.read().replace("\n", "")

aci_domain = "10.200.300.40" # Replace with the domain of your container instance
response  = requests.post(f"http://{aci_domain}:8080/v1/models/model:predict", headers=headers, data=data)
print(response.text)
```

