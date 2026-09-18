## The working way
You should always work on your branch, and push to your branch.
Once you pushed to your branch, if you want to bring the changes FROM main INTO your branch:

    git checkout la-branche-de-tali
    git merge main

Once you pushed to your branch, if you want to bring the changes FROM your branch INTO main:

    git checkout main
    git merge la-branche-de-tali
    git push
    
# To push on own branch

    git checkout la-branche-de-tali    # make sure you're on your branch
    git add .
    git commit -m "describe what you changed"
    git push

