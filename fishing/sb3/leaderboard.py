from datetime import datetime
from git import Repo
import csv
import os

def leaderboard(agent, env, mean, std, script, file = "leaderboard.csv"):
    ## Commit file and compute GitHub URL
    repo = Repo(".", search_parent_directories=True)
    path = os.path.relpath(script, repo.git.working_dir)
    repo.git.add(path)
    if len(repo.index.diff("HEAD")) > 0:
        repo.git.commit("-m 'robot commit before running script'")
    sha = repo.commit().hexsha
    url = repo.git.remote("get-url", "origin") + "/blob/" + sha + "/" + path
    row_contents = {"agent": agent,
                    "env": env,
                    "mean": mean, 
                    "std": std,
                    "url": url,
                    "date": datetime.now()}
    with open(file, 'a+') as stream:
        writer = csv.DictWriter(stream, 
                                fieldnames = ["agent", 
                                              "env", 
                                              "mean", 
                                              "std", 
                                              "url", 
                                              "date"])
        writer.writeheader()
        writer.writerow(row_contents)


