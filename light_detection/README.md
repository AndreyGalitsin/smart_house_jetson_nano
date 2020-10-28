# ResNet18 on Jetson Nano
You need:
- Jetson Nano
- USB web_cam

Steps we need to do:
- Install PyTorch v1.2 and torchvision v0.4.0 for python3
  - [To install follow this link](https://forums.developer.nvidia.com/t/pytorch-for-jetson-nano-version-1-4-0-now-available/72048)
- For training your own classificator: 
  - Create your dataset with architecture like here:

    dataset -> train -> class_1 -> photos

    dataset -> train -> class_2 -> photos

    ...

    dataset -> val -> class_1 -> photos


    dataset -> val -> class_2 -> photos

    ...

  - Change in the following code '2' to the number of your classes (train.py)
  
    ```python
    model_conv.fc = nn.Linear(num_ftrs, 2)
    ```
  - Change in the following code '2' to the number of your classes (resnet.py)
  
    ```python
    self.model_ft.fc = nn.Linear(num_ftrs, 2)
    ```
    
  - Choose path and name for weights in the following code (train.py)
  ```python
    torch.save(model_conv, "./model/resnet18_feature_extractor.pt")
    ```
    
  - RUN script train.py using comand in terminal 'python3 train.py'
  
- To load and run your model:
  - Choose the path to your weights in the following code (resnet.py)
  
    ```python
    self.path_to_state_dict = './model/state_dict_v1.pt'
    ```
  - RUN script main.py using comand in terminal 'python3 main.py'

**For more information about resnet and Jetson Nano I recommend the following links:**
- [Official Pytorch documentation](https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html)
- [Example how to train pytorch models and create datasets](https://www.youtube.com/channel/UCi46pc2k7P6FeFJWhS7qFwQ/featured)
- [Useful advises about Jetson Nano](https://www.youtube.com/watch?v=5INy0FvaWLw&list=PLGs0VKk2DiYxP-ElZ7-QXIERFFPkOuP4_)
  
