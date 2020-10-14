import git
repo = git.Repo(".")
repo.git.add("git-log.py")
repo.git.commit("-m add git log")
repo.git.status()
sha = repo.commit().hexsha
sha
