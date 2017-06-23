sudo apt install unzip python3 python3-pip
pip3 install -r requirements.txt
mkdir ../datasets
cd ../datasets
wget http://openi.nlm.nih.gov/imgs/collections/ChinaSet_AllFiles.zip
wget http://openi.nlm.nih.gov/imgs/collections/NLM-MontgomeryCXRSet.zip
unzip ChinaSet_AllFiles.zip
unzip NLM-MontgomeryCXRSet.zip
rm *.zip
rm __MACOSX -rf
