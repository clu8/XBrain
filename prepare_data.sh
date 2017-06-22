sudo apt install unp python3 python3-pip
pip3 install -r requirements.txt
mkdir ~/datasets
python3 get_data.py
cd ~/datasets
unp pilot_images.tgz
rm pilot_images.tgz
cd pilot_images
unp *.gz
rm *.gz
