
1. Pull the GPU image of tensorflow from docker in the folder with data.

Make sure you have nvidia docker in your machine for GPU support.

`docker run -it --rm --gpus all -e DISPLAY=$DISPLAY --device /dev/video0 --ipc=host  --net=host  -v $(pwd):/app -w /app tensorflow/tensorflow:latest-gpu /bin/bash`


2. Update and install python-3 tk
`apt-get update`
`apt-get install python3-tk`

3. Install dependencies

`pip install matplotlib`



4. Folder structure is \
`-Data` \
`----ApplePie` \
`----BagelSandwich` \
`----Bibimbop` \
`----Bread` \
`----FriedRice` \
`----Pork` \
`-train.py` \
`-infer.py` \
-`README.md `

6. Run `train.py` file for training model.

7. To infer the model run `infer.py` using `Data` as the folder.


