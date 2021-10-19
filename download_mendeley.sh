wget https://md-datasets-cache-zipfiles-prod.s3.eu-west-1.amazonaws.com/9xkhgts2s6-3.zip
unzip 9xkhgts2s6-3.zip -d data
mv data/Curated\ X-Ray\ Dataset/Curated\ X-Ray\ Dataset/ data/curated_xray_dataset
rm -r data/Curated\ X-Ray\ Dataset
rm 9xkhgts2s6-3.zip