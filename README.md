# Domian Fusion in Latent Space (DFLS) for Face Video Super-Resolution
This is the program page of our latest DFLS approach for Face Video Super-Resolution(FVSR).

## System requirments

### Dataset
you can download training data from https://drive.google.com/drive/folders/19DLr27P9xMOTn_W6hxpxxm8_5jJoX-nR

### Training
Train the model by following the command lines below
python train.py

Details:
--data_path: [should be filled in the directory to the training dataset]
--ckpt_path: [the location of your training model]

### Inference
After the training you can run the following command to FVSR for evaluation.
and we also offer a demo, you can vision the demo from https://github.com/jiashaohua1993/DFLS/blob/main/face.mp4

python eval.py

<img width="557" height="320" alt="image" src="https://github.com/user-attachments/assets/8137d6f7-11f7-44b2-9d12-0231c04b0db8" />


### Related Code
https://github.com/GreyCC/DTLS_1024
https://github.com/yangxy/GPEN
https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement
https://github.com/LabShuHangGU/MIA-VSR
