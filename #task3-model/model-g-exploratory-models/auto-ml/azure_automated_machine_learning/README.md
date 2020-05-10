# Azure Automated ML

Before reproducing this notebook read following docs articles:
* https://docs.microsoft.com/en-us/azure/machine-learning/tutorial-1st-experiment-sdk-setup
* https://docs.microsoft.com/en-us/azure/machine-learning/how-to-set-up-training-targets#local
* https://docs.microsoft.com/en-us/azure/machine-learning/how-to-auto-train-remote


# Docker based jupyter notebook template

Everyone on linux with docker installed can open this code in jupyter notebook  
Just execute `run-jupyter.sh` and follow the link printed in console

![alt text](docker_jupyter.png?raw=true "Docker jupyter")

To export your new conda libraries execute following command in jupyter lab terminal
`conda list --explicit > requirements.txt`
