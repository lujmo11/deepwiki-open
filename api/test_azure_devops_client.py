from deepwiki_open.repositories import factory_from_url, AzureDevOpsRepository
import os
import pytest
import requests
from unittest.mock import patch, MagicMock
from api.azure_devops_client import AzureDevOpsClient, AzureDevOpsAPIError

@pytest.fixture
def valid_env(monkeypatch):
    monkeypatch.setenv("AZURE_DEVOPS_PAT", "dummy_pat")
    monkeypatch.setenv("AZURE_DEVOPS_BASE_URL", "https://dev.azure.com")

def test_init_missing_pat(monkeypatch):
    monkeypatch.delenv("AZURE_DEVOPS_PAT", raising=False)
    with pytest.raises(AzureDevOpsAPIError):
        AzureDevOpsClient()

def test_init_with_pat(valid_env):
    client = AzureDevOpsClient()
    assert client.pat == "dummy_pat"
def test_factory_from_url_returns_azure_devops_repo():
    url = "https://dev.azure.com/myorg/myproj/_git/myrepo"
    repo = factory_from_url(url, "dummy_token")
    assert isinstance(repo, AzureDevOpsRepository)
from deepwiki_open.repositories.azure_devops import AzureDevOpsRepository

@patch("deepwiki_open.repositories.azure_devops.requests.get")
def test_get_default_branch_main(mock_get):
    # Arrange
    repo_url = "https://dev.azure.com/myorg/myproj/_git/myrepo"
    token = "dummy_token"
    repo = AzureDevOpsRepository(repo_url, token)
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "value": [
            {"name": "refs/heads/main"}
        ]
    }
    mock_get.return_value = mock_response

    # Act
    branch = repo.get_default_branch()

    # Assert
    assert branch == "main"
@patch("deepwiki_open.repositories.azure_devops.requests.get")
def test_get_file_sha_raises_auth_error(mock_get):
    from deepwiki_open.repositories.azure_devops import AzureDevOpsRepository, RepositoryAuthError
    repo_url = "https://dev.azure.com/myorg/myproj/_git/myrepo"
    token = "dummy_token"
    repo = AzureDevOpsRepository(repo_url, token)
    mock_response = MagicMock()
    mock_response.status_code = 401
    mock_response.json.return_value = {}
    mock_get.return_value = mock_response

    import pytest
    with pytest.raises(RepositoryAuthError):
        repo.get_file_sha("/README.md")
@patch("deepwiki_open.repositories.azure_devops.requests.get")
def test_get_file_sha_returns_40char_hex(mock_get):
    from deepwiki_open.repositories.azure_devops import AzureDevOpsRepository
    repo_url = "https://dev.azure.com/myorg/myproj/_git/myrepo"
    token = "dummy_token"
    repo = AzureDevOpsRepository(repo_url, token)
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "objectId": "a" * 40
    }
    mock_get.return_value = mock_response

    sha = repo.get_file_sha("/README.md")
    assert isinstance(sha, str)
    assert len(sha) == 40
    assert all(c in "0123456789abcdef" for c in sha.lower())

@patch("requests.Session.get")
def test_get_repository_metadata_success(mock_get, valid_env):
    client = AzureDevOpsClient()
    mock_resp = MagicMock()
    mock_resp.ok = True
    mock_resp.json.return_value = {"id": "repo1"}
    mock_resp.status_code = 200
    mock_get.return_value = mock_resp
    result = client.get_repository_metadata("org", "proj", "repo")
    assert result["id"] == "repo1"

@patch("requests.Session.get")
def test_get_repository_metadata_404(mock_get, valid_env):
    client = AzureDevOpsClient()
    mock_resp = MagicMock()
    mock_resp.ok = False
    mock_resp.status_code = 404
    mock_resp.text = "Not found"
    mock_get.return_value = mock_resp
    with pytest.raises(AzureDevOpsAPIError):
        client.get_repository_metadata("org", "proj", "repo")

@patch("requests.Session.get")
def test_get_repository_items_success(mock_get, valid_env):
    client = AzureDevOpsClient()
    mock_resp = MagicMock()
    mock_resp.ok = True
    mock_resp.json.return_value = {"value": [{"path": "/file1.txt"}]}
    mock_resp.status_code = 200
    mock_get.return_value = mock_resp
    items = client.get_repository_items("org", "proj", "repo")
    assert items == [{"path": "/file1.txt"}]

@patch("requests.Session.get")
def test_get_file_content_success(mock_get, valid_env):
    client = AzureDevOpsClient()
    mock_resp = MagicMock()
    mock_resp.ok = True
    mock_resp.json.return_value = {"content": "file content"}
    mock_resp.status_code = 200
    mock_get.return_value = mock_resp
    content = client.get_file_content("org", "proj", "repo", "/file.txt")
    assert content == "file content"

@patch("requests.Session.get")
def test_get_file_content_not_found(mock_get, valid_env):
    client = AzureDevOpsClient()
    mock_resp = MagicMock()
    mock_resp.ok = True
    mock_resp.json.return_value = {}
    mock_resp.status_code = 200
    mock_get.return_value = mock_resp
    with pytest.raises(AzureDevOpsAPIError):
        client.get_file_content("org", "proj", "repo", "/file.txt")

@patch("requests.Session.get")
def test_handle_response_auth_error(mock_get, valid_env):
    client = AzureDevOpsClient()
    mock_resp = MagicMock()
    mock_resp.ok = False
    mock_resp.status_code = 401
    mock_resp.text = "Unauthorized"
    mock_get.return_value = mock_resp
    with pytest.raises(AzureDevOpsAPIError):
        client.get_repository_metadata("org", "proj", "repo")
from deepwiki_open.repositories.azure_devops import AzureDevOpsRepository

@patch("requests.get")
def test_azure_devops_get_tree_root_items(mock_get):
    # Arrange
    repo_url = "https://dev.azure.com/myorg/myproj/_git/myrepo"
    token = "dummy_token"
    repo = AzureDevOpsRepository(repo_url, token)
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "value": [
            {"path": "/README.md", "gitObjectType": "blob"},
            {"path": "/src", "gitObjectType": "tree"},
        ]
    }
    mock_get.return_value = mock_response

    # Act
    result = repo.get_tree()

    # Assert
    mock_get.assert_called_once()
    called_url = mock_get.call_args[0][0]
    assert called_url == (
        "https://dev.azure.com/myorg/myproj/_apis/git/repositories/"
        "myrepo/items?recursionLevel=OneLevel&api-version=4.1"
    )
    assert result == [
        {"path": "/README.md", "type": "blob"},
        {"path": "/src", "type": "tree"},
    ]