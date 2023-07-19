pip install gdown

# Create precomputed_proposals directory if it doesn't exist
mkdir -p ./precomputed_proposals

gdown https://drive.google.com/uc?id=1mrXcsK4TC63931UTrw3cDnOWjeGSM4F- -O ./precomputed_proposals/ssl_clad.tar
tar -xvf ./precomputed_proposals/ssl_clad.tar -C ./precomputed_proposals/
