rm -rf tmp
mkdir tmp
cd tmp
rm -rf .git
git init
git pull https://github.com/Ahmetereni/ASTRO-AI.git
sudo rm -rf /var/www/html/
sudo mkdir /var/www/html
sudo mv * /var/www/html
sudo systemctl restart nginx.service
echo Nginx dosyalari guncellendi

cd ..
rm -rf tmp
echo Gecici Dosya silindi
echo
