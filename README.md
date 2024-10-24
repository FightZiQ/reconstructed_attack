# Privacy Does Not Work: A Sample Reconstruction Attack against Federated Learning with Differential Privacy

**The implementation of paper:<br>
Privacy Does Not Work: A Sample Reconstruction Attack against Federated Learning with Differential Privacy**

We propose an attack against federated learning (FL) with differential privacy (DP) in which gradients uploaded by
users are clipped and perturbed.

Here are some examples:

* Original samples:
![original.png](fig%2Foriginal.png)

* Our reconstructed samples:
![res.png](fig%2Fres.png)

* Reconstructed samples generated by other attacks:
![when.png](fig%2Fwhen.png)

Other attacks are implemented by Jonas Geiping. If you are interested in more reconstruction samples in FL, please check
out his project: https://github.com/JonasGeiping/breaching

---
## Note
Our paper is under review now. We will update the latest and more complete code when the paper is accepted. (10.24.2024)

---

## What is this?

Generally, this project provides a sample reconstruction attack against users' gradients in federated learning with
differential privacy.

* **Federated learning (FL):** FL is a distributed learning for user privacy preservation in which users train a given model 
and upload the gradients or model updates instead of original training samples.

* **Differential privacy (DP):** DP enables a database owner to release data statistics features without revealing personal privacy.
In our paper, users' gradients are protected by clipping and perturbation, i.e., users clip and perturb their gradients before
uploading to the server.

* **Sample reconstruction attack:**  Sample reconstruction attacks aim to reconstruct users' training samples through their
uploaded gradients, as shown in the beginning example.

* **Our works and this project:** The existing reconstruction attacks are ineffective in FL with DP. In our paper, we
propose new reconstruction attack to reconstruct users' sample in FL with DP. Simulation results show that the proposed 
attack can still effectively reconstruct the sensitive information of users' training although gradients are protected by DP.
This project is a code implementation of the attack proposed in our paper.

---

## Requirement

The following are the requirements for the implementation of this attack.

### 1. Dataset
We use four datasets in the experiments. The default dataset is ImageNet, please download it before you run this attack:
https://www.image-net.org/.

### 2. SAM

We apply SAM to training samples for masks generation and extract images' subjects before training. Please visit the
project website to get the latest `segment_anything`: https://github.com/facebookresearch/segment-anything.
Besides, you also need to download the latest SAM model checkpoint to your local devices.

### 3. Dependency path

After you download the dataset and SAM, please provide their local path in `dataset_path` and `sam_path` of `file_path.py`.

---

## Try with a given example

After meeting the above requirement, try `python example.py` for a given example.

