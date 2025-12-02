"""Example: Build SLM from REST API."""

from slm_builder import SLMBuilder


def main():
    """Build classification model from REST API."""
    print("🌐 Building SLM from REST API")
    print("=" * 50)

    # API authentication
    auth = {
        "type": "bearer",  # or 'basic', 'api_key'
        "token": "your-api-token-here",
    }

    # Pagination configuration
    pagination = {
        "type": "offset",  # or 'page', 'cursor'
        "param": "offset",
        "size_param": "limit",
        "page_size": 100,
    }

    # Initialize builder
    builder = SLMBuilder(
        project_name="api-classification-model",
        base_model="gpt2",
        device="auto",
        work_dir="./api_slm",
    )

    try:
        # Build from API
        print("\n📥 Loading data from API...")
        result = builder.build_from_api(
            base_url="https://api.example.com",
            endpoint="/v1/training-data",
            task="classification",
            recipe="lora",
            auth=auth,
            pagination=pagination,
            max_pages=10,  # Limit pages to fetch
            rate_limit=5.0,  # 5 requests per second
        )

        print("\n✅ Training complete!")
        print(f"Model saved to: {result['model_dir']}")
        print(f"Metrics: {result['metrics']}")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure:")
        print("1. API endpoint is accessible")
        print("2. Authentication token is valid")
        print("3. API returns data in expected format")
        print("4. requests is installed: pip install requests")


# Example with API key authentication
def api_key_example():
    """Example using API key authentication."""
    auth = {
        "type": "api_key",
        "key_name": "X-API-Key",  # Header name
        "key_value": "your-api-key",
    }

    builder = SLMBuilder(project_name="api-model", base_model="gpt2")

    result = builder.build_from_api(
        base_url="https://api.example.com", endpoint="/data", task="qa", auth=auth, max_pages=5
    )

    return result


# Example with custom response parser
def custom_parser_example():
    """Example with custom API response parser."""

    def custom_parser(response):
        """Parse custom API response format."""
        # Adjust based on your API response structure
        if "payload" in response and "items" in response["payload"]:
            return response["payload"]["items"]
        return []

    from slm_builder.data import APILoader

    loader = APILoader(task="qa")
    records = loader.load(
        base_url="https://api.example.com",
        endpoint="/custom",
        response_parser=custom_parser,
        max_pages=10,
    )

    builder = SLMBuilder(project_name="custom-api-model", base_model="gpt2")
    result = builder.build_from_dataset(records=records, task="qa", recipe="lora")

    return result


if __name__ == "__main__":
    main()
