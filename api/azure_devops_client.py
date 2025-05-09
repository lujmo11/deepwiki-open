import os
import requests
from typing import Optional, Dict, Any, List

class AzureDevOpsAPIError(Exception):
    """Custom exception for Azure DevOps API errors."""
    pass

class AzureDevOpsClient:
    """
    Client for interacting with the Azure DevOps Repos REST API.

    Features:
    - Authenticates using a Personal Access Token (PAT) from environment variables.
    - Fetches repository metadata.
    - Retrieves file tree structure.
    - Fetches file content.
    - Handles API errors robustly.
    """

    def __init__(self, base_url: Optional[str] = None, pat_env_var: str = "AZURE_DEVOPS_PAT"):
        """
        Initialize the AzureDevOpsClient.

        Args:
            base_url (str, optional): Base URL for Azure DevOps API. If None, uses config or default.
            pat_env_var (str): Environment variable name for the PAT.
        """
        self.base_url = base_url or os.getenv("AZURE_DEVOPS_BASE_URL", "https://dev.azure.com")
        self.pat = os.getenv(pat_env_var)
        if not self.pat:
            raise AzureDevOpsAPIError(f"Personal Access Token not found in environment variable '{pat_env_var}'.")

        self.session = requests.Session()
        self.session.auth = ("", self.pat)
        self.session.headers.update({"Content-Type": "application/json"})

    def _handle_response(self, response: requests.Response) -> Any:
        """
        Handle API responses, raising for errors.

        Args:
            response (requests.Response): The response object.

        Returns:
            Parsed JSON response.

        Raises:
            AzureDevOpsAPIError: For HTTP errors or API issues.
        """
        if response.status_code == 401:
            raise AzureDevOpsAPIError("Authentication failed: Invalid Personal Access Token (PAT).")
        if response.status_code == 404:
            raise AzureDevOpsAPIError("Resource not found (404). Check organization, project, or repository name.")
        if response.status_code == 429:
            raise AzureDevOpsAPIError("Rate limit exceeded (429). Please try again later.")
        if not response.ok:
            try:
                detail = response.json()
            except Exception:
                detail = response.text
            raise AzureDevOpsAPIError(f"API Error {response.status_code}: {detail}")
        try:
            return response.json()
        except Exception:
            return response.text

    def get_repository_metadata(self, organization: str, project: str, repository_name: str) -> Dict[str, Any]:
        """
        Fetch metadata for a repository.

        Args:
            organization (str): Azure DevOps organization name.
            project (str): Project name.
            repository_name (str): Repository name.

        Returns:
            dict: Repository metadata.
        """
        url = f"{self.base_url}/{organization}/{project}/_apis/git/repositories/{repository_name}?api-version=7.1-preview.1"
        resp = self.session.get(url)
        return self._handle_response(resp)

    def get_repository_items(self, organization: str, project: str, repository_name: str, scope_path: str = "/", recursion_level: str = "Full") -> List[Dict[str, Any]]:
        """
        Retrieve the file tree structure for a repository.

        Args:
            organization (str): Azure DevOps organization name.
            project (str): Project name.
            repository_name (str): Repository name.
            scope_path (str): Path to start from (default: root '/').
            recursion_level (str): 'Full' for all files, 'OneLevel' for top-level only.

        Returns:
            list: List of items (files/folders) in the repository.
        """
        url = (
            f"{self.base_url}/{organization}/{project}/_apis/git/repositories/"
            f"{repository_name}/items?scopePath={scope_path}&recursionLevel={recursion_level}&api-version=7.1-preview.1"
        )
        resp = self.session.get(url)
        data = self._handle_response(resp)
        return data.get("value", [])

    def get_file_content(self, organization: str, project: str, repository_name: str, path: str, version: Optional[str] = None) -> str:
        """
        Fetch the content of a specific file.

        Args:
            organization (str): Azure DevOps organization name.
            project (str): Project name.
            repository_name (str): Repository name.
            path (str): File path in the repository.
            version (str, optional): Branch, tag, or commit (default: default branch).

        Returns:
            str: File content as text.
        """
        url = (
            f"{self.base_url}/{organization}/{project}/_apis/git/repositories/"
            f"{repository_name}/items?path={path}&api-version=7.1-preview.1&includeContent=true"
        )
        if version:
            url += f"&version={version}"
        resp = self.session.get(url)
        data = self._handle_response(resp)
        # The API returns content in 'content' field for text files
        if isinstance(data, dict) and "content" in data:
            return data["content"]
        raise AzureDevOpsAPIError("File content not found or not a text file.")