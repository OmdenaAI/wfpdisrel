Folder content:
Process: 
1. Apply 'WBI_Selection_of_Indicators.ipynb'/'Selected_Indicators.csv': this process transforms original WBI dataset of 1430 indicators and produces a smaller WBI dataset with indicators selected. Result: 'OUTPUT_WBI_Selection_Country_Year.csv'

2. Apply 'WBI_Final.ipynb': this process transforms 'OUTPUT_WBI_Selection_Country_Year.csv' , arranges matrix anf fills na for countries that appear, producing an output 'WBI_Final.csv' that can be merged with cyclones dataset. Pandas profiling report of this file in  'WBIFinal Profiling report final.html '


a. OUTPUT_WBI_Selection_Country_Year.csv                    : contains final dataset to fill cyclones dataset
b. Selected_Indicators.csv             	                   : Selected Indicators from a total of 1430 indicators
c. WBI_Selection_of_Indicators.ipynb                       : Jupyter Notebook that drops not selected indicators from original dataset
d. WBI_Final.ipynb                                          : Jupyter Notebook that aranges matrix anf fills na for countries that appear in cyclones dataset
e. WBI_Final.csv                                            : dataset to be appended to cyclones dataset

f. WBIFinal Profiling report final.html                     : Pandas profiling of the final dataset
