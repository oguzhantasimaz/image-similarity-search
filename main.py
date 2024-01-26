import numpy as np
from PIL import Image, ExifTags
import os
from torchvision import transforms
import torch
from transformers import CLIPModel
import upstash_vector as uv

upstash_url = os.environ.get('UPSTASH_URL')
token = os.environ.get('UPSTASH_TOKEN')
index = uv.Index(url=upstash_url, token=token)

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
        embedding = transform_image(image).tolist()

        id = filename

        index.upsert(vectors = [(id, embedding, {"metadata_field": "metadata_value"})])

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
result = index.query(vector=query_vector,  top_k=top_k, include_metadata=True)

# Print results
print(f"Query image: {query_image_path}")
print(f"Top {top_k} results:")

for i, res in enumerate(result):
    print(f"Rank {i + 1}: ID={res.id}, score={res.score}")

