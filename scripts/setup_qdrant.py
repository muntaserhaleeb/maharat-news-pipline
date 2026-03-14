import yaml
import os
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PayloadSchemaType

def setup_qdrant():
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'qdrant.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Initialize client
    qdrant_config = config['qdrant']
    client = QdrantClient(host=qdrant_config['host'], port=qdrant_config['port'])
    collection_name = qdrant_config['collection_name']

    # Vector params
    vector_config = config['vector']
    distance_map = {
        'Cosine': Distance.COSINE,
        'Euclid': Distance.EUCLID,
        'Dot': Distance.DOT
    }
    
    # Recreate collection to ensure it exists with the right configuration
    # Note: This will delete existing data in the collection if it already exists.
    # We do a check to avoid deletion if it exists, or just recreate for setup.
    if client.collection_exists(collection_name):
        print(f"Collection '{collection_name}' already exists.")
        # Optional: You can choose to skip or delete and recreate
        # client.delete_collection(collection_name)
    else:
        print(f"Creating collection '{collection_name}' with vector size {vector_config['size']}...")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_config['size'],
                distance=distance_map.get(vector_config['distance'], Distance.COSINE)
            )
        )
        print("Collection created successfully.")

    # Apply payload schema
    payload_schema = config.get('payload_schema', {})
    
    type_map = {
        'keyword': PayloadSchemaType.KEYWORD,
        'text': PayloadSchemaType.TEXT,
        'datetime': PayloadSchemaType.DATETIME,
        'integer': PayloadSchemaType.INTEGER,
        'float': PayloadSchemaType.FLOAT,
        'bool': PayloadSchemaType.BOOL,
        'geo': PayloadSchemaType.GEO,
        'keyword_array': PayloadSchemaType.KEYWORD, # Arrays use same base type
    }

    print("Configuring payload schema indexes...")
    for field_name, field_config in payload_schema.items():
        field_type_str = field_config.get('type')
        if field_type_str in type_map:
            payload_type = type_map[field_type_str]
            # Qdrant schema indexing
            try:
                client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field_name,
                    field_schema=payload_type
                )
                print(f" - Created index for payload field: {field_name} ({field_type_str})")
            except Exception as e:
                # Might fail if index already exists
                print(f" - Note: Index for '{field_name}' could not be created or already exists: {e}")
        else:
            print(f" - Warning: Unsupported schema type '{field_type_str}' for field '{field_name}'")

    print(f"\nQdrant setup complete! Database is ready to receive vectors for '{collection_name}'.")

if __name__ == "__main__":
    setup_qdrant()
