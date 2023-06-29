pip install gdown
gdown https://drive.google.com/uc?id=1wJDDRvqBlaF7R8n473XJzdinh2qGkOI7 -O ./dataset/data.tar

#unzip dataset
tar -xvf ./dataset/data.tar -C ./dataset/

#remove tar file
rm ./dataset/data.tar