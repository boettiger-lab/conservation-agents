from datetime import datetime
from csv import writer
from git import Repo
import os
def leaderboard(agent, env, mean, std, file = "results/leaderboard.csv"):
    
    ## Commit file and compute GitHub URL
    repo = Repo(".", search_parent_directories=True)
    script = os.path.basename(__file__)
    path = os.path.relpath(script, repo.git.working_dir)
    repo.git.add(path)
    if len(repo.index.diff("HEAD")) > 0:
        repo.git.commit("-m 'robot commit before running script'")
    sha = repo.commit().hexsha
    url = repo.git.remote("get-url", "origin") + "/blob/" + sha + "/" + path

    
    stream = open(file, 'a+')
    now = datetime.now()
    row_contents = [agent,
                    env,
                    mean, 
                    std,
                    url,
                    now]
    csv_writer = writer(stream)
    csv_writer.writerow(row_contents)
