"""API data loaders for REST APIs."""

import time
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import urljoin

import requests
from tqdm import tqdm

from slm_builder.data.schemas import (
    create_classification_record,
    create_instruction_record,
    create_qa_record,
    record_to_dict,
)
from slm_builder.utils import get_logger

logger = get_logger(__name__)


class APILoader:
    """Load data from REST APIs with pagination and authentication."""

    def __init__(self, task: str = "qa"):
        """Initialize API loader.

        Args:
            task: Task type
        """
        self.task = task

    def load(
        self,
        base_url: str,
        endpoint: str = "",
        auth: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, Any]] = None,
        pagination: Optional[Dict[str, Any]] = None,
        response_parser: Optional[Callable] = None,
        max_pages: int = 10,
        rate_limit: Optional[float] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """Load data from REST API.

        Args:
            base_url: Base API URL
            endpoint: API endpoint path
            auth: Authentication config
            headers: HTTP headers
            params: Query parameters
            pagination: Pagination configuration
            response_parser: Custom parser function
            max_pages: Maximum pages to fetch
            rate_limit: Requests per second limit
            **kwargs: Additional arguments

        Returns:
            List of records

        Example:
            auth = {
                'type': 'bearer',  # or 'basic', 'api_key'
                'token': 'your-token'
            }
            pagination = {
                'type': 'offset',  # or 'cursor', 'page'
                'param': 'offset',
                'size_param': 'limit',
                'page_size': 100
            }
        """
        url = urljoin(base_url, endpoint)
        logger.info("Loading from API", url=url, max_pages=max_pages)

        # Setup authentication
        session = requests.Session()
        if auth:
            session = self._setup_auth(session, auth)

        if headers:
            session.headers.update(headers)

        # Setup pagination
        pagination_config = pagination or {"type": "none"}
        params = params or {}

        all_data = []
        page = 0
        has_more = True

        with tqdm(total=max_pages, desc="Fetching API pages") as pbar:
            while has_more and page < max_pages:
                # Add pagination params
                page_params = self._get_pagination_params(pagination_config, page, len(all_data))
                current_params = {**params, **page_params}

                try:
                    # Make request
                    response = session.get(url, params=current_params, timeout=30)
                    response.raise_for_status()

                    data = response.json()

                    # Parse response
                    if response_parser:
                        parsed_data = response_parser(data)
                    else:
                        parsed_data = self._default_parser(data)

                    if not parsed_data:
                        has_more = False
                        break

                    all_data.extend(parsed_data)

                    # Check if there's more data
                    has_more = self._has_more_pages(data, pagination_config, len(parsed_data))

                    page += 1
                    pbar.update(1)

                    # Rate limiting
                    if rate_limit and has_more:
                        time.sleep(1.0 / rate_limit)

                except requests.RequestException as e:
                    logger.error("API request failed", error=str(e), page=page)
                    break

        session.close()

        logger.info("API load complete", total_items=len(all_data), pages=page)

        # Convert to canonical records
        records = []
        for item in all_data:
            record = self._item_to_record(item)
            if record:
                records.append(record_to_dict(record))

        logger.info("Records created", count=len(records))
        return records

    def _setup_auth(self, session: requests.Session, auth: Dict[str, Any]) -> requests.Session:
        """Setup authentication for session."""
        auth_type = auth.get("type", "").lower()

        if auth_type == "bearer":
            token = auth.get("token")
            session.headers["Authorization"] = f"Bearer {token}"
        elif auth_type == "basic":
            from requests.auth import HTTPBasicAuth

            username = auth.get("username")
            password = auth.get("password")
            session.auth = HTTPBasicAuth(username, password)
        elif auth_type == "api_key":
            key_name = auth.get("key_name", "X-API-Key")
            key_value = auth.get("key_value")
            session.headers[key_name] = key_value
        elif auth_type == "oauth2":
            # OAuth2 would require more complex flow
            logger.warning("OAuth2 authentication requires manual token management")
            token = auth.get("token")
            if token:
                session.headers["Authorization"] = f"Bearer {token}"

        return session

    def _get_pagination_params(
        self, pagination: Dict[str, Any], page: int, current_count: int
    ) -> Dict[str, Any]:
        """Get pagination parameters for current page."""
        pagination_type = pagination.get("type", "none")

        if pagination_type == "offset":
            page_size = pagination.get("page_size", 100)
            return {
                pagination.get("param", "offset"): current_count,
                pagination.get("size_param", "limit"): page_size,
            }
        elif pagination_type == "page":
            page_size = pagination.get("page_size", 100)
            return {
                pagination.get("param", "page"): page + 1,
                pagination.get("size_param", "per_page"): page_size,
            }
        elif pagination_type == "cursor":
            # Cursor-based pagination requires storing the cursor
            cursor = pagination.get("next_cursor")
            if cursor:
                return {pagination.get("param", "cursor"): cursor}
            return {}
        else:
            return {}

    def _has_more_pages(
        self, response: Dict[str, Any], pagination: Dict[str, Any], items_count: int
    ) -> bool:
        """Check if there are more pages to fetch."""
        pagination_type = pagination.get("type", "none")

        if pagination_type == "none":
            return False

        # Check if response indicates more data
        if "has_more" in response:
            return response["has_more"]

        if "next" in response and response["next"]:
            return True

        if "next_cursor" in response and response["next_cursor"]:
            pagination["next_cursor"] = response["next_cursor"]
            return True

        # If we got fewer items than page size, assume no more pages
        page_size = pagination.get("page_size", 100)
        if items_count < page_size:
            return False

        return True

    def _default_parser(self, response: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Default response parser."""
        # Try common response structures
        if "data" in response:
            data = response["data"]
            if isinstance(data, list):
                return data
            return [data]
        elif "results" in response:
            return response["results"]
        elif "items" in response:
            return response["items"]
        elif isinstance(response, list):
            return response
        else:
            return [response]

    def _item_to_record(self, item: Dict[str, Any]) -> Optional[Any]:
        """Convert API item to canonical record."""
        # Try to extract text from common fields
        text = (
            item.get("text")
            or item.get("content")
            or item.get("body")
            or item.get("description")
            or ""
        )

        if not text:
            return None

        record_id = str(item.get("id", ""))
        metadata = item.get("metadata", {})

        if self.task == "qa":
            question = item.get("question", "")
            answer = item.get("answer", "")
            return create_qa_record(text, question, answer, record_id, metadata)
        elif self.task == "classification":
            label = item.get("label") or item.get("category", "")
            return create_classification_record(text, label, record_id, metadata)
        elif self.task == "instruction":
            instruction = item.get("instruction") or item.get("prompt", "")
            response = item.get("response") or item.get("completion", "")
            return create_instruction_record(text, instruction, response, record_id, metadata)
        else:
            return create_qa_record(text, "", "", record_id, metadata)


def load_from_api(
    base_url: str,
    task: str = "qa",
    endpoint: str = "",
    auth: Optional[Dict[str, Any]] = None,
    pagination: Optional[Dict[str, Any]] = None,
    max_pages: int = 10,
    **kwargs,
) -> List[Dict[str, Any]]:
    """Convenience function to load from REST API.

    Args:
        base_url: Base API URL
        task: Task type
        endpoint: API endpoint
        auth: Authentication configuration
        pagination: Pagination configuration
        max_pages: Maximum pages to fetch
        **kwargs: Additional arguments

    Returns:
        List of records
    """
    loader = APILoader(task=task)
    return loader.load(
        base_url=base_url,
        endpoint=endpoint,
        auth=auth,
        pagination=pagination,
        max_pages=max_pages,
        **kwargs,
    )
