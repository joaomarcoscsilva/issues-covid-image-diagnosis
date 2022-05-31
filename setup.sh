# BEFORE RUNNING: Add kaggle.json to ~/.kaggle/
pip3 install -r requirements
./download_mendeley.sh
sudo apt-get update
sudo apt-get install ffmpeg libsm6 libxext6  -y
mv data/curated_xray_dataset/ mendeley/
pip3 install kaggle --upgrade
sudo ln -s ~/.local/bin/kaggle /usr/bin/kaggle
kaggle datasets download -d tawsifurrahman/covid19-radiography-database
unzip covid19-radiography-database.zip
mv COVID-19_Radiography_Dataset/ tawsifur/
rm covid19-radiography-database.zip
mv tawsifur/COVID tawsifur/COVID-19
mv "mendeley/Pneumonia-Bacterial" "mendeley/Bacterial pneumonia"
mv "mendeley/Pneumonia-Viral" "mendeley/Viral pneumonia"
mkdir covidx
mkdir dup_data
mkdir models
cd covidx
kaggle datasets download -d andyczhao/covidx-cxr2
unzip covidx-cxr2.zip
rm covidx-cxr2.zip
wget https://raw.githubusercontent.com/lindawangg/COVID-Net/master/labels/train_COVIDx9A.txt
wget https://raw.githubusercontent.com/lindawangg/COVID-Net/master/labels/test_COVIDx9A.txt
cd ..
#python3 process_data.py mendeley
#python3 process_data.py tawsifur
#python3 process_covidx.py