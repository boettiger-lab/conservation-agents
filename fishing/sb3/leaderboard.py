from datetime import datetime
from csv import writer
from git import Repo

def leaderboard(agent, mean, std, file = "results/leaderboard.csv"):
    
    repo = Repo(".")
    repo.git.add("git-log.py")
    repo.git.commit("-m add git log")
    repo.git.status()
    sha = repo.commit().hexsha
    sha

    
    stream = open(file, 'a+')
    now = datetime.now()
    row_contents = [agent,
                    mean, 
                    std,
                    now]
    csv_writer = writer(stream)
    csv_writer.writerow(row_contents)
