from .azure_devops import AzureDevOpsRepository
import re

REPO_TYPE_BY_URL = {
    r"(dev\.azure\.com|visualstudio\.com).*_git/": "azure_devops",
    # Add other repo types here as needed
}

REPO_CLASS_BY_TYPE = {
    "azure_devops": AzureDevOpsRepository,
    # Add other repo types here as needed
}

def get_repo_type_from_url(url):
    for pattern, repo_type in REPO_TYPE_BY_URL.items():
        if re.search(pattern, url):
            return repo_type
    return None

def factory_from_url(url, *args, **kwargs):
    repo_type = get_repo_type_from_url(url)
    if repo_type and repo_type in REPO_CLASS_BY_TYPE:
        return REPO_CLASS_BY_TYPE[repo_type](url, *args, **kwargs)
    raise ValueError(f"No repository adapter found for URL: {url}")