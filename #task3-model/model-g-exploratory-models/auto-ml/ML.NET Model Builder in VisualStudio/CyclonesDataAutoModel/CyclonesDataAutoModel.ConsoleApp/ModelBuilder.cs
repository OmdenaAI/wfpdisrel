//*****************************************************************************************
//*                                                                                       *
//* This is an auto-generated file by Microsoft ML.NET CLI (Command-Line Interface) tool. *
//*                                                                                       *
//*****************************************************************************************

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using CyclonesDataAutoModel.Model.DataModels;
using Microsoft.ML.Trainers.LightGbm;

namespace CyclonesDataAutoModel.ConsoleApp
{
    public static class ModelBuilder
    {
        private static string TRAIN_DATA_FILEPATH = @"/home/augustowski/repos/wfpdisrel/#task3-model/model-g-exploratory-models/auto-ml/ML.NET Model Builder in VisualStudio/data/OUTPUT_WBI_exposer_cyclones.csv";
        private static string MODEL_FILEPATH = @"../../../../CyclonesDataAutoModel.Model/MLModel.zip";

        // Create MLContext to be shared across the model creation workflow objects 
        // Set a random seed for repeatable/deterministic results across multiple trainings.
        private static MLContext mlContext = new MLContext(seed: 1);

        public static void CreateModel()
        {
            // Load Data
            IDataView trainingDataView = mlContext.Data.LoadFromTextFile<ModelInput>(
                                            path: TRAIN_DATA_FILEPATH,
                                            hasHeader: true,
                                            separatorChar: ',',
                                            allowQuoting: true,
                                            allowSparse: false);

            // Build training pipeline
            IEstimator<ITransformer> trainingPipeline = BuildTrainingPipeline(mlContext);

            // Evaluate quality of Model
            Evaluate(mlContext, trainingDataView, trainingPipeline);

            // Train Model
            ITransformer mlModel = TrainModel(mlContext, trainingDataView, trainingPipeline);

            // Save model
            SaveModel(mlContext, mlModel, MODEL_FILEPATH, trainingDataView.Schema);
        }

        public static IEstimator<ITransformer> BuildTrainingPipeline(MLContext mlContext)
        {
            // Data process configuration with pipeline data transformations 
            var dataProcessPipeline = mlContext.Transforms.Categorical.OneHotEncoding(new[] { new InputOutputColumnPair("ISO", "ISO"), new InputOutputColumnPair("BASIN", "BASIN"), new InputOutputColumnPair("SUB_BASIN", "SUB_BASIN"), new InputOutputColumnPair("NATURE", "NATURE"), new InputOutputColumnPair("Income_level_Final", "Income_level_Final"), new InputOutputColumnPair("country_susan", "country_susan"), new InputOutputColumnPair("Country_susan", "Country_susan"), new InputOutputColumnPair("Alpha_2_code_susan", "Alpha_2_code_susan"), new InputOutputColumnPair("Alpha_3_code_susan", "Alpha_3_code_susan") })
                                      .Append(mlContext.Transforms.Categorical.OneHotHashEncoding(new[] { new InputOutputColumnPair("NAME", "NAME"), new InputOutputColumnPair("name_susan", "name_susan") }))
                                      .Append(mlContext.Transforms.Text.FeaturizeText("ISO_TIME_tf", "ISO_TIME"))
                                      .Append(mlContext.Transforms.Text.FeaturizeText("COORDS_tf", "COORDS"))
                                      .Append(mlContext.Transforms.Concatenate("Features", new[] { "ISO", "BASIN", "SUB_BASIN", "NATURE", "Income_level_Final", "country_susan", "Country_susan", "Alpha_2_code_susan", "Alpha_3_code_susan", "NAME", "name_susan", "ISO_TIME_tf", "COORDS_tf", "col0", "YEAR", "TOTAL_HRS", "DAY_HRS", "NIGHT_HRS", "USA_SSHS", "WIND_CALC_MEAN", "PRES_CALC_MEAN", "STORM_SPD_MEAN", "STORM_DR_MEAN", "V_LAND_KN", "34KN_POP", "34KN_ASSETS", "64KN_POP", "64KN_ASSETS", "96KN_POP", "96KN_ASSETS", "CPI", "TOTAL_DAMAGE__000__", "TOTAL_DEATHS", "Air_transport__freight__million_ton_km_", "Arable_land__hectares_per_person_", "Cereal_yield__kg_per_hectare_", "Food_production_index__2004_2006___100_", "GDP_growth__annual___", "GDP_per_capita__constant_2010_US__", "Net_flows_from_UN_agencies_US_", "Life_expectancy_at_birth__total__years_", "Mobile_cellular_subscriptions__per_100_people_", "Population_density__people_per_sq__km_of_land_area_", "Adjusted_savings__education_expenditure____of_GNI_", "Rural_population____of_total_population_", "Population__total", "Population_2000", "Population_2005", "Population_2010", "Population_2015", "Population_2020", "pop_max_34", "pop_max_50", "pop_max_64", "Numeric", "Unnamed__0_susan", "year_susan", "pop_max_34_susan", "pop_max_50_susan", "pop_max_64_susan", "Numeric_susan" }));

            // Set the training algorithm 
            var trainer = mlContext.Regression.Trainers.LightGbm(new LightGbmRegressionTrainer.Options() { NumberOfIterations = 100, LearningRate = 0.0403179f, NumberOfLeaves = 75, MinimumExampleCountPerLeaf = 50, UseCategoricalSplit = false, HandleMissingValue = true, MinimumExampleCountPerGroup = 10, MaximumCategoricalSplitPointCount = 8, CategoricalSmoothing = 1, L2CategoricalRegularization = 0.5, Booster = new GradientBooster.Options() { L2Regularization = 0, L1Regularization = 0.5 }, LabelColumnName = "TOTAL_AFFECTED", FeatureColumnName = "Features" });
            var trainingPipeline = dataProcessPipeline.Append(trainer);

            return trainingPipeline;
        }

        public static ITransformer TrainModel(MLContext mlContext, IDataView trainingDataView, IEstimator<ITransformer> trainingPipeline)
        {
            Console.WriteLine("=============== Training  model ===============");

            ITransformer model = trainingPipeline.Fit(trainingDataView);

            Console.WriteLine("=============== End of training process ===============");
            return model;
        }

        private static void Evaluate(MLContext mlContext, IDataView trainingDataView, IEstimator<ITransformer> trainingPipeline)
        {
            // Cross-Validate with single dataset (since we don't have two datasets, one for training and for evaluate)
            // in order to evaluate and get the model's accuracy metrics
            Console.WriteLine("=============== Cross-validating to get model's accuracy metrics ===============");
            var crossValidationResults = mlContext.Regression.CrossValidate(trainingDataView, trainingPipeline, numberOfFolds: 5, labelColumnName: "TOTAL_AFFECTED");
            PrintRegressionFoldsAverageMetrics(crossValidationResults);
        }
        private static void SaveModel(MLContext mlContext, ITransformer mlModel, string modelRelativePath, DataViewSchema modelInputSchema)
        {
            // Save/persist the trained model to a .ZIP file
            Console.WriteLine($"=============== Saving the model  ===============");
            mlContext.Model.Save(mlModel, modelInputSchema, GetAbsolutePath(modelRelativePath));
            Console.WriteLine("The model is saved to {0}", GetAbsolutePath(modelRelativePath));
        }

        public static string GetAbsolutePath(string relativePath)
        {
            FileInfo _dataRoot = new FileInfo(typeof(Program).Assembly.Location);
            string assemblyFolderPath = _dataRoot.Directory.FullName;

            string fullPath = Path.Combine(assemblyFolderPath, relativePath);

            return fullPath;
        }

        public static void PrintRegressionMetrics(RegressionMetrics metrics)
        {
            Console.WriteLine($"*************************************************");
            Console.WriteLine($"*       Metrics for regression model      ");
            Console.WriteLine($"*------------------------------------------------");
            Console.WriteLine($"*       LossFn:        {metrics.LossFunction:0.##}");
            Console.WriteLine($"*       R2 Score:      {metrics.RSquared:0.##}");
            Console.WriteLine($"*       Absolute loss: {metrics.MeanAbsoluteError:#.##}");
            Console.WriteLine($"*       Squared loss:  {metrics.MeanSquaredError:#.##}");
            Console.WriteLine($"*       RMS loss:      {metrics.RootMeanSquaredError:#.##}");
            Console.WriteLine($"*************************************************");
        }

        public static void PrintRegressionFoldsAverageMetrics(IEnumerable<TrainCatalogBase.CrossValidationResult<RegressionMetrics>> crossValidationResults)
        {
            var L1 = crossValidationResults.Select(r => r.Metrics.MeanAbsoluteError);
            var L2 = crossValidationResults.Select(r => r.Metrics.MeanSquaredError);
            var RMS = crossValidationResults.Select(r => r.Metrics.RootMeanSquaredError);
            var lossFunction = crossValidationResults.Select(r => r.Metrics.LossFunction);
            var R2 = crossValidationResults.Select(r => r.Metrics.RSquared);

            Console.WriteLine($"*************************************************************************************************************");
            Console.WriteLine($"*       Metrics for Regression model      ");
            Console.WriteLine($"*------------------------------------------------------------------------------------------------------------");
            Console.WriteLine($"*       Average L1 Loss:       {L1.Average():0.###} ");
            Console.WriteLine($"*       Average L2 Loss:       {L2.Average():0.###}  ");
            Console.WriteLine($"*       Average RMS:           {RMS.Average():0.###}  ");
            Console.WriteLine($"*       Average Loss Function: {lossFunction.Average():0.###}  ");
            Console.WriteLine($"*       Average R-squared:     {R2.Average():0.###}  ");
            Console.WriteLine($"*************************************************************************************************************");
        }
    }
}
