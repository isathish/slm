"""Example: Build SLM from PostgreSQL database."""

from slm_builder import SLMBuilder


def main():
    """Build QA model from PostgreSQL database."""
    print("🗄️ Building SLM from PostgreSQL Database")
    print("=" * 50)

    # Database connection parameters
    connection_params = {
        "dialect": "postgresql",
        "host": "localhost",
        "port": 5432,
        "database": "training_db",
        "user": "your_username",
        "password": "your_password",
    }

    # SQL query to fetch training data
    query = """
    SELECT
        id,
        question_text as text,
        answer_text,
        category,
        created_at as metadata
    FROM qa_training_data
    WHERE status = 'verified'
    AND created_at >= '2024-01-01'
    LIMIT 1000
    """

    # Column mapping (maps database columns to canonical format)
    column_mapping = {
        "text": "question_text",
        "label.question": "question_text",
        "label.answer": "answer_text",
    }

    # Initialize builder
    builder = SLMBuilder(
        project_name="database-qa-model", base_model="gpt2", device="auto", work_dir="./db_slm"
    )

    try:
        # Build from database
        print("\n📥 Loading data from PostgreSQL...")
        result = builder.build_from_database(
            query=query,
            connection_params=connection_params,
            db_type="sql",
            task="qa",
            recipe="lora",
            column_mapping=column_mapping,
        )

        print("\n✅ Training complete!")
        print(f"Model saved to: {result['model_dir']}")
        print(f"Metrics: {result['metrics']}")

        # Export model
        print("\n📦 Exporting model...")
        exported_path = builder.export(
            model_dir=result["model_dir"], format="onnx", optimize_for="cpu"
        )
        print(f"Exported to: {exported_path}")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure:")
        print("1. PostgreSQL is running")
        print("2. Database credentials are correct")
        print("3. Table 'qa_training_data' exists")
        print("4. psycopg2-binary is installed: pip install psycopg2-binary sqlalchemy")


if __name__ == "__main__":
    main()
