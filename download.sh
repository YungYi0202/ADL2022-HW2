mkdir qa_ckpt
cd qa_ckpt
mkdir 8
cd 8
wget https://www.dropbox.com/s/79o5c64jgnkxnlz/qa_ckpt.zip?dl=0 -O model.zip
unzip model.zip
rm model.zip

cd ../..

mkdir mc_ckpt
cd mc_ckpt
mkdir 11
cd 11
wget https://www.dropbox.com/s/ohueneqv5hkzs97/mc_ckpt.zip?dl=0 -O mc_ckpt.zip
unzip mc_ckpt.zip
rm mc_ckpt.zip
