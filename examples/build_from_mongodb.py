"""Example: Build SLM from MongoDB database."""

from slm_builder import SLMBuilder


def main():
    """Build instruction-following model from MongoDB."""
    print("🗄️ Building SLM from MongoDB")
    print("=" * 50)

    # MongoDB connection parameters
    connection_params = {
        "host": "localhost",
        "port": 27017,
        "database": "training_db",
        "username": "your_username",  # Optional
        "password": "your_password",  # Optional
    }

    # Collection name (instead of SQL query)
    collection_name = "instruction_data"

    # MongoDB query filter
    query_filter = {"status": "verified", "language": "en", "quality_score": {"$gte": 4.0}}

    # Field projection (which fields to include)
    projection = {"text": 1, "instruction": 1, "response": 1, "metadata": 1, "_id": 0}

    # Initialize builder
    builder = SLMBuilder(
        project_name="mongodb-instruction-model",
        base_model="distilgpt2",
        device="auto",
        work_dir="./mongo_slm",
    )

    try:
        # Build from MongoDB
        print("\n📥 Loading data from MongoDB...")
        result = builder.build_from_database(
            query=collection_name,  # Collection name for MongoDB
            connection_params=connection_params,
            db_type="mongodb",
            task="instruction",
            recipe="lora",
            query_filter=query_filter,
            projection=projection,
            limit=1000,  # Limit documents
        )

        print("\n✅ Training complete!")
        print(f"Model saved to: {result['model_dir']}")
        print(f"Metrics: {result['metrics']}")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure:")
        print("1. MongoDB is running")
        print("2. Database credentials are correct")
        print("3. Collection 'instruction_data' exists")
        print("4. pymongo is installed: pip install pymongo")


if __name__ == "__main__":
    main()
