using System;
using BinaryClassifier;
using Xunit;
using Xunit.Abstractions;
using System.IO;
using System.Reflection;

namespace BinaryClassificationSample
{
    public class MainTester
    {
        private readonly ITestOutputHelper _output;
        private readonly string _dataFolderPath;

        public MainTester(ITestOutputHelper output)
        {
            this._output = output;
            this._dataFolderPath = Path.Combine(Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location), "..\\..\\..\\..\\data\\");
        }
        
        [Fact]
        public void Test_Check_Data_Folder()
        {
            _output.WriteLine($"Data Folder Path : {this._dataFolderPath}");
        }

        [Fact]
        public void Test_Evaluate_Prediction_Model_Score_Yelp()
        {
            _output.WriteLine("Test_Evaluate_Prediction_Model_Score_Yelp()...");

            var predictor = new Predictor();
            
            predictor.LoadTestData(Path.Combine(_dataFolderPath,"yelp_labelled.txt"));
            predictor.BuildAndTrainModel();
            var metrics = predictor.EvaluateModelMetrics();
            
            _output.WriteLine($"Model Metrics - Accuracy:{metrics.Accuracy:P2}, AreaUnderRocCurve:{metrics.AreaUnderRocCurve:P2}, F1Score:{metrics.F1Score:P2}");
            
            Assert.InRange(metrics.Accuracy,0.5, 0.95);
            
            _output.WriteLine("...Test_Evaluate_Prediction_Model_Score_Yelp() DONE.");
        }
        
        [Fact]
        public void Test_Predict_Output_Positive_Yelp()
        {
            _output.WriteLine("Test_Predict_Output_Positive_Yelp()...");

            const string INPUT_DATA = "Food was very good";
            
            var predictor = new Predictor();
            
            predictor.LoadTestData(Path.Combine(_dataFolderPath,"yelp_labelled.txt"));
            predictor.BuildAndTrainModel();

            var predictionTuple = predictor.Predict(INPUT_DATA);
            
            _output.WriteLine($"Prediction for \"{INPUT_DATA}\" : {predictionTuple.prediction} (Probability: {predictionTuple.probability}, Score: {predictionTuple.score})");
            
            Assert.True(predictionTuple.prediction);
            
            _output.WriteLine("...Test_Predict_Output_Positive_Yelp() DONE.");
        }
        
        [Fact]
        public void Test_Predict_Output_Negative_Yelp()
        {
            _output.WriteLine("Test_Predict_Output_Negative_Yelp()...");

            const string INPUT_DATA = "This was quite a horrible meal";
            
            var predictor = new Predictor();
            
            predictor.LoadTestData(Path.Combine(_dataFolderPath,"yelp_labelled.txt"));
            predictor.BuildAndTrainModel();

            var predictionTuple = predictor.Predict(INPUT_DATA);
            
            _output.WriteLine($"Prediction for \"{INPUT_DATA}\" : {predictionTuple.prediction} (Probability: {predictionTuple.probability}, Score: {predictionTuple.score})");
            
            Assert.False(predictionTuple.prediction);
            
            _output.WriteLine("...Test_Predict_Output_Negative_Yelp() DONE.");
        }
        
        [Fact]
        public void Test_Evaluate_Prediction_Model_Score_Amazon()
        {
            _output.WriteLine("Test_Evaluate_Prediction_Model_Score_Amazon()...");

            var predictor = new Predictor();
            
            predictor.LoadTestData(Path.Combine(_dataFolderPath,"amazon_cells_labelled.txt"));
            predictor.BuildAndTrainModel();
            var metrics = predictor.EvaluateModelMetrics();
            
            _output.WriteLine($"Model Metrics - Accuracy:{metrics.Accuracy:P2}, AreaUnderRocCurve:{metrics.AreaUnderRocCurve:P2}, F1Score:{metrics.F1Score:P2}");
            
            Assert.InRange(metrics.Accuracy,0.5, 0.95);
            
            _output.WriteLine("...Test_Evaluate_Prediction_Model_Score_Amazon() DONE.");
        }
        
        [Fact]
        public void Test_Predict_Output_Positive_Amazon()
        {
            _output.WriteLine("Test_Predict_Output_Positive_Amazon()...");

            const string INPUT_DATA = "this smartphone is very good and light";
            
            var predictor = new Predictor();
            
            predictor.LoadTestData(Path.Combine(_dataFolderPath,"amazon_cells_labelled.txt"));
            predictor.BuildAndTrainModel();

            var predictionTuple = predictor.Predict(INPUT_DATA);
            
            _output.WriteLine($"Prediction for \"{INPUT_DATA}\" : {predictionTuple.prediction} (Probability: {predictionTuple.probability}, Score: {predictionTuple.score})");
            
            Assert.True(predictionTuple.prediction);
            
            _output.WriteLine("...Test_Predict_Output_Positive_Amazon() DONE.");
        }
        
        [Fact]
        public void Test_Predict_Output_Negative_Amazon()
        {
            _output.WriteLine("Test_Predict_Output_Negative_Amazon()...");

            const string INPUT_DATA = "this smartphone does not work at all, and is heavy too";
            
            var predictor = new Predictor();
            
            predictor.LoadTestData(Path.Combine(_dataFolderPath,"amazon_cells_labelled.txt"));
            predictor.BuildAndTrainModel();

            var predictionTuple = predictor.Predict(INPUT_DATA);
            
            _output.WriteLine($"Prediction for \"{INPUT_DATA}\" : {predictionTuple.prediction} (Probability: {predictionTuple.probability}, Score: {predictionTuple.score})");
            
            Assert.False(predictionTuple.prediction);
            
            _output.WriteLine("...Test_Predict_Output_Negative_Amazon() DONE.");
        }
    }
}