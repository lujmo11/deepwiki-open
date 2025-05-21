"""This script will cut a release from the main branch and publish it to a release/ branch."""

import subprocess
from datetime import date


def main() -> None:
    # Check if the current branch is main
    current_branch = (
        subprocess.check_output(["git", "branch", "--show-current"]).decode("utf-8").strip()
    )
    if current_branch != "main":
        print("You must be on the main branch to create a release.")
        return

    # Check that main is up to date with the remote
    subprocess.check_output(["git", "pull"])

    # define branch name prefix
    branch_prefix = f"release/{date.today().strftime('%Y-%m-%d')}"
    # get all git branches with that prefix
    branches = (
        subprocess.check_output(["git", "branch", "--all", "--list", f"*{branch_prefix}*"])
        .decode("utf-8")
        .split("\n")
    )
    branches = [branch.strip() for branch in branches if branch.strip() != ""]

    n_releases = len(branches)
    branch_prefix = f"{branch_prefix}-{n_releases + 1}"

    # Specify the name of the release
    release_name = input("Enter the name of the release: ")
    # normalize the name
    release_name = release_name.lower().replace(" ", "-").replace("_", "-")
    branch_name = f"{branch_prefix}-{release_name}"
    # Check with user before proceeding
    print(f"Creating release: '{branch_name}' from main branch.")
    proceed = input("Proceed? (y/n): ")
    if proceed.lower() != "y":
        return
    # create a new branch
    subprocess.check_output(["git", "checkout", "-b", f"{branch_name}"])
    # push the branch to the remote
    subprocess.check_output(["git", "push", "--set-upstream", "origin", f"{branch_name}"])


if __name__ == "__main__":
    main()
