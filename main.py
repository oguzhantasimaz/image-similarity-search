import pinecone
import numpy as np
from PIL import Image, ExifTags
import os
from torchvision import transforms
import torch
from transformers import CLIPModel

api_key = os.environ.get('PINECONE_API_KEY')
environment = os.environ.get('PINECONE_ENVIRONMENT')

# Initialize Pinecone client
pinecone.init(api_key=api_key, environment=environment)

# Connect to your Pinecone index
index_name = "<your_index_name>"
index = pinecone.Index(index_name)

# Define your image directory
image_dir = "./images/views"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

# Define image preprocessing (ensure consistency with CLIP's training setup)
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def transform_image(image):
    image = preprocess(image)
    image = image.unsqueeze(0)
    with torch.no_grad():
        features = model.get_image_features(pixel_values=image)
    embedding = features.squeeze().cpu().numpy()
    return embedding.astype(np.float32)

# Extract features for each image and insert into Pinecone index
for filename in os.listdir(image_dir):
    if filename.endswith(".jpg"):
        image_path = os.path.join(image_dir, filename)
        image = Image.open(image_path)
        embedding = transform_image(image)

        id = filename

        vector = [{"values": embedding, "metadata": {}, "id": id }]
        index.upsert(vector, namespace='ot')

        print(f"Upserted image {filename} with ID {filename}")


# Query Pinecone index

# Define your query image
query_image_path = "./query_image.jpg"

# Preprocess query image
query_image = Image.open(query_image_path)
query_embedding = transform_image(query_image)  # Squeeze the tensor to remove batch dimension

# Query Pinecone index

# The top_k parameter controls the number of results to retrieve
top_k = 5
query_vector = query_embedding.tolist()
result = index.query(vector=query_vector, namespace='ot', top_k=top_k, include_metadata=True, include_values=True)

# Print results
print(f"Query image: {query_image_path}")
print(f"Top {top_k} results:")

for i, res in enumerate(result.matches):
    print(f"Rank {i + 1}: ID={res.id}, score={res.score}")
