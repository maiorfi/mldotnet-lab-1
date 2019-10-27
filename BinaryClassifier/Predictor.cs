using System;
using System.ComponentModel.Design;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace BinaryClassifier
{
    public class Predictor
    {
        private const string DATA_PATH = @"C:\Progetti\Lab\NetCore\ML\BinaryClassificationSample\data\yelp_labelled.txt";

        MLContext _mlContext = new MLContext();

        private IDataView _allData;
        
        private IDataView _trainData;
        private IDataView _testData;

        private ITransformer _model;
        

        public void LoadTestData()
        {
            _allData = _mlContext.Data.LoadFromTextFile<SentimentData>(DATA_PATH, hasHeader: false);

            var trainingSplitData = _mlContext.Data.TrainTestSplit(_allData, testFraction: 0.2);
            
            _trainData = trainingSplitData.TrainSet;
            _testData = trainingSplitData.TestSet;
        }

        public void BuildAndTrainModel()
        {
            var estimator = _mlContext.Transforms.Text.FeaturizeText(outputColumnName: "Features", inputColumnName: nameof(SentimentData.SentimentText))
                .Append(_mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: "Label", featureColumnName: "Features"));

            _model = estimator.Fit(_testData);
        }

        public CalibratedBinaryClassificationMetrics EvaluateModelMetrics()
        {
            var predictions = _model.Transform(_trainData);

            return _mlContext.BinaryClassification.Evaluate(predictions, "Label");
        }

        public (bool prediction, float score, float probability) Predict(string inputData)
        {
            var predictionFunction = _mlContext.Model.CreatePredictionEngine<SentimentData, SentimentPrediction>(_model);

            var inputSample = new SentimentData()
            {
                SentimentText = inputData
            };

            var resultPrediction = predictionFunction.Predict(inputSample);

            return (resultPrediction.Prediction, resultPrediction.Score, resultPrediction.Probability);
        }
    }
}