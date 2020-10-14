import git
repo = git.Repo(".")
repo.git.add("git-log.py")
sha = repo.commit().hexsha
