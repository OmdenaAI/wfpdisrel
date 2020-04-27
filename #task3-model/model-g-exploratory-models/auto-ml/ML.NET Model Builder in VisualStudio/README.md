# ML.Net Model Builder in Visual Studio

To replicate whole process of auto-training please follow this steps:
1. Install the dotnet sdk and mlnet tool following instructions from https://dotnet.microsoft.com/learn/ml-dotnet/get-started-tutorial/install
2. Locate your dataset 
3. Execute following command
   
    `mlnet auto-train --task regression --dataset ./data/OUTPUT_WBI_exposer_cyclones.csv --label-column-name TOTAL_AFFECTED --max-exploration-time 360 --name CyclonesDataAutoModel`

    * --task - is one of the following: `regression`, `binary-classification` or `multiclass-classification`
    * --dataset - path to the dataset
    * --label-column-name - the name of target column (if the dataset does not have header row please use --label-column-index parameter instead)
    * --max-exploration-time - time for which the search should be conducted
    * --name - name of result model folder