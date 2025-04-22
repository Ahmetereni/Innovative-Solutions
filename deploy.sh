# cd dist
git init
num=$(( RANDOM % 1000 + 1 )) # generate a random number between 1-1000
git add .
git commit -m "Random commit #$num"
git remote add origin https://github.com/Ahmetereni/Innovative-Solutions.git
git push origin main --force
