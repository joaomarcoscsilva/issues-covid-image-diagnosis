wget https://dl.dafont.com/dl/?f=linux_libertine
mv index.html\?f\=linux_libertine font.zip
mkdir -p ~/.local/share/fonts
cp *.ttf ~/.local/share/fonts/
fc-cache -f -v
fc-list | grep Libertine
python3
import shutil
import matplotlib
shutil.rmtree(matplotlib.get_cachedir())
quit()