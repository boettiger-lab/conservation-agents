import sh
git = sh.git.bake(_cwd='.')
git.status()
# add a file
print git.add('git-log.py')
# commit
print git.commit(m='robot update')
# now we are one commit ahead
print git.status()
