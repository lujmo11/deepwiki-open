"""Azure DevOps repository integration stub.

This module defines the AzureDevOpsRepository class, which provides an interface
for interacting with Azure DevOps repositories. All methods are currently stubs.
"""

from .base import RepositoryBase
from .base import RepositoryAuthError, RepositoryNotFoundError
import requests
import re
from enum import Enum

class RepoObjectType(Enum):
    FILE = "file"
    FOLDER = "folder"

class AzureDevOpsRepository(RepositoryBase):
    """
    Azure DevOps repository interface.

    Args:
        url (str): The repository URL.
        token (str): The access token for authentication.
        branch (str, optional): The branch to operate on. Defaults to None.
    """

    def __init__(self, url: str, token: str = None, branch: str = None):
        import os
        self.url = url
        if token is None:
            token = os.environ.get("AZURE_DEVOPS_TOKEN")
        self.token = token
        self.branch = branch
    def get_file_sha(self, path: str) -> str:
        """
        Return the objectId (SHA) for a file at the given path.
        """
        # Parse org, project, repo from self.url
        import re
        m = re.match(r"https://dev\.azure\.com/([^/]+)/([^/]+)/_git/([^/]+)", self.url)
        if not m:
            raise ValueError("Invalid Azure DevOps repo URL")
        org, project, repo = m.groups()
        url = (
            f"https://dev.azure.com/{org}/{project}/_apis/git/repositories/"
            f"{repo}/items?path={path}&includeContent=false&api-version=7.1-preview.1"
        )
        resp = requests.get(
            url,
            auth=("", self.token),
            headers={"Accept": "application/json"},
        )
        if resp.status_code in (401, 403):
            raise RepositoryAuthError(f"Authentication failed: {resp.status_code}")
        if resp.status_code == 404:
            raise RepositoryNotFoundError(f"File not found: {path}")
        resp.raise_for_status()
        data = resp.json()
        object_id = data.get("objectId")
        if not object_id or not re.fullmatch(r"[a-fA-F0-9]{40}", object_id):
            raise ValueError("objectId not found or not a 40-char hex")
        return object_id

    def get_tree(self):
        """
        Retrieve the root tree of the Azure DevOps repository.

        Returns:
            list of dict: [{"path": "/README.md", "type": "blob"}, ...]
        """
        # Example Azure DevOps URL:
        # https://dev.azure.com/{organization}/{project}/_git/{repo}
        m = re.match(
            r"https://dev\.azure\.com/(?P<org>[^/]+)/(?P<project>[^/]+)/_git/(?P<repo>[^/?#]+)",
            self.url,
        )
        if not m:
            raise ValueError("Repository URL format is invalid for Azure DevOps.")

        org = m.group("org")
        project = m.group("project")
        repo = m.group("repo")

        api_url = (
            f"https://dev.azure.com/{org}/{project}/_apis/git/repositories/"
            f"{repo}/items?recursionLevel=OneLevel&api-version=4.1"
        )

        response = requests.get(
            api_url,
            auth=("", self.token),
            headers={"Accept": "application/json"},
        )
        response.raise_for_status()
        data = response.json()
        items = []
        for item in data.get("value", []):
            path = item.get("path")
            git_type = item.get("gitObjectType", "").lower()
            # Map ADO type to RepoObjectType
            if git_type == "blob":
                obj_type = "blob"
            elif git_type == "tree":
                obj_type = "tree"
            else:
                continue
            # Filter out unwanted paths
            if path.startswith("/.git"):
                continue
            items.append({
                "path": path,
                "type": obj_type
            })
        return items

    def get_blob(self, path):
        """
        Retrieve the contents of a file (blob) at the given path.

        Args:
            path (str): The file path.

        Raises:
            NotImplementedError: This method is not implemented.
        """
        raise NotImplementedError("get_blob() is not implemented.")

    def get_default_branch(self):
        """
        Retrieve the default branch of the repository.

        Returns:
            str: The default branch name (e.g., "main" or "master").
        """
        # Parse org, project, repo from URL
        m = re.match(
            r"https://dev\.azure\.com/(?P<org>[^/]+)/(?P<project>[^/]+)/_git/(?P<repo>[^/?#]+)",
            self.url,
        )
        if not m:
            raise ValueError("Repository URL format is invalid for Azure DevOps.")

        org = m.group("org")
        project = m.group("project")
        repo = m.group("repo")

        # 1. Try refs API for heads/main or heads/master
        refs_url = (
            f"https://dev.azure.com/{org}/{project}/_apis/git/repositories/"
            f"{repo}/refs"
        )
        params = {
            "filter": "heads/",
            "filterContains": "master,main",
            "api-version": "7.1-preview.1",
        }
        response = requests.get(
            refs_url,
            params=params,
            auth=("", self.token),
            headers={"Accept": "application/json"},
        )
        if response.status_code in (401, 403):
            raise RepositoryAuthError(f"Authentication failed: {response.status_code}")
        if response.status_code == 404:
            raise RepositoryNotFoundError("Repository or refs not found")
        response.raise_for_status()
        data = response.json()
        refs = data.get("value", [])
        for ref in refs:
            name = ref.get("name", "")
            if name == "refs/heads/main":
                return "main"
            if name == "refs/heads/master":
                return "master"

        # 2. Fallback: repository metadata API
        repo_url = (
            f"https://dev.azure.com/{org}/{project}/_apis/git/repositories/"
            f"{repo}?api-version=7.1-preview.1"
        )
        response = requests.get(
            repo_url,
            auth=("", self.token),
            headers={"Accept": "application/json"},
        )
        response.raise_for_status()
        repo_data = response.json()
        default_branch = repo_data.get("defaultBranch")
        if default_branch and default_branch.startswith("refs/heads/"):
            return default_branch[len("refs/heads/") :]
        raise RuntimeError("Could not determine default branch from Azure DevOps.")

    def get_file_content(self, path: str) -> bytes:
        """
        Retrieve the content of a file at the given path using Azure DevOps REST API.

        Args:
            path (str): The file path.

        Returns:
            bytes: The file content as bytes.
        """
        import base64

        m = re.match(
            r"https://dev\.azure\.com/(?P<org>[^/]+)/(?P<project>[^/]+)/_git/(?P<repo>[^/?#]+)",
            self.url,
        )
        if not m:
            raise ValueError("Repository URL format is invalid for Azure DevOps.")

        org = m.group("org")
        project = m.group("project")
        repo = m.group("repo")

        api_url = (
            f"https://dev.azure.com/{org}/{project}/_apis/git/repositories/"
            f"{repo}/items"
        )
        params = {
            "path": path,
            "includeContent": "true",
            "api-version": "7.1-preview.1",
        }
        if self.branch:
            params["versionDescriptor.version"] = self.branch

        response = requests.get(
            api_url,
            params=params,
            auth=("", self.token),
            headers={"Accept": "application/json"},
        )
        response.raise_for_status()
        data = response.json()
        if "contentBytes" in data:
            return base64.b64decode(data["contentBytes"])
        elif "content" in data:
            return data["content"].encode("utf-8")
        else:
            raise RuntimeError("No file content found in Azure DevOps API response.")