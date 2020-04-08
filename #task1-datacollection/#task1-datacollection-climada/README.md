# Population by area - subtask

## Proposed structure
* notebooks folder - contains subfolders for each jupyter 
  * jupyter subfolder
    * notebook.ipynb
    * requirements.txt - with all packages listed
      * you can achieve that by using 

        `conda list --export > requirements.txt` on linux
      * you can than install those dependencies using 
  
        `conda install --file requirements.txt -c conda-forge`
    * all other staff conected to this specific notebook ex. images or other things
* data folder - containing all shared thata for this task ( not sure if we should copy over data from other parts of repo or just use it but someone could possibly remove it )
