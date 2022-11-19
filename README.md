# Less is More: Facial Landmarks can Recognize a Spontaneous Smile (BMVC 2022)
## MeshSmileNet PyTorch Implementation

### Dependency 
* python 3.8
* numpy 1.21.5 
* Pillow 9.0.1
* dlib 19.24.0
* opencv-python 4.6.0.66
* torch 1.11.0 
* torchvision 0.12.0
* vidaug 1.5
* einops 0.6.0
* tqdm 4.64.1
* colorama 0.4.6 

## Dataset
* obtain the whole UVA-NEMO database from https://www.uva-nemo.org
* obtain the whole MMI database from https://mmifacedb.eu
* obtain the whole BBC database from https://www.bbc.co.uk/science/humanbody/mind/surveys/smiles/
* obtain the whole SPOS database from https://www.oulu.fi/cmvs/node/41317

## Train MeshSmileNet
* ```smile_point.py``` contains the train program for MeshSmileNet.

* run ```python smile_point.py --fold 0``` for the sample training of UVA-NEMO databases. Note only landmarks data from UVA-NEMO are provided. Other can be extracted by following the code from here https://google.github.io/mediapipe/solutions/face_mesh.html

* The model weight will be saved in ```uva/labels``` folder.

The Github Repo is under construction.

