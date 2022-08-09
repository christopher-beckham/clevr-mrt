from subprocess import check_output
import json

def get_current_git_metadata():
    # Fetch branch.
    git_branch = check_output("git rev-parse --symbolic-full-name --abbrev-ref HEAD", shell=True)
    git_branch = git_branch.decode('utf-8').rstrip()
    # Fetch commit.
    git_hash = check_output("git rev-parse HEAD", shell=True)
    git_hash = git_hash.decode('utf-8').rstrip()
    # Get the patch, if it's a dirty commit
    git_patch = check_output("git stash show -p", shell=True)
    git_patch = git_patch.decode('utf-8').rstrip()
    return {
        'branch': git_branch,
        'hash': git_hash,
        'patch': git_patch
    }

if __name__ == '__main__':
    print("testing function...")
    dd = get_current_git_metadata()
    print(json.dumps(dd))
