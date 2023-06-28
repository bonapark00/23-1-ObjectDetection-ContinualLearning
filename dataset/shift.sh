#python3 -m pip install -r ./requirements.txt

# download all files in shift
python ./dataset/download_shift.py --view "front" --group "[img, det_2d]" --split "all" --framerate "[images]" --shift "discrete" dataset/SHIFT_dataset/
