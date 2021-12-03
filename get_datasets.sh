# Add kaggle.json to ~/.kaggle/
./download_mendeley.sh
pip3 install -r requirements
mv data/curated_xray_dataset/ mendeley/
pip3 install kaggle --upgrade
kaggle datasets download -d tawsifurrahman/covid19-radiography-database
unzip covid19-radiography-database.zip
mv COVID-19_Radiography_Dataset/ tawsifur/
rm covid19-radiography-database.zip
mv tawsifur/COVID tawsifur/COVID-19
mv "mendeley/Pneumonia-Bacterial" "mendeley/Bacterial pneumonia"
mv "mendeley/Pneumonia-Viral" "mendeley/Viral pneumonia"
python3 process_data.py mendeley
python3 process_data.py tawsifur