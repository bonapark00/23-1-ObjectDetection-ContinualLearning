pip install gdown

# Create precomputed_proposals directory if it doesn't exist
mkdir -p ./precomputed_proposals

gdown https://drive.google.com/uc?id=136iRRmxCrV0naYK1wYPVYHaeOdHml9XD -O ./precomputed_proposals/ssl_shift.tar
tar -xvf ./precomputed_proposals/ssl_shift.tar -C ./precomputed_proposals/
