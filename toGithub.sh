# cd dist
git init
num=$(( RANDOM % 1000 + 1 )) # generate a random number between 1-1000
git add .
git commit -m "Random commit #$num"
echo "Do you want to continue (y,n) \c"
read answer
if [ "$answer" = "y" ] || [ "$answer" = "Y" ]; then
   git remote add origin https://github.com/Ahmetereni/Innovative-Solutions.git
   git push origin main --force
else
    echo "Exiting."
    exit 1
fi

