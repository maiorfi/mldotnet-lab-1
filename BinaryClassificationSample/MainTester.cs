using System;
using BinaryClassifier;
using Xunit;
using Xunit.Abstractions;

namespace BinaryClassificationSample
{
    public class MainTester
    {
        private readonly ITestOutputHelper _output;

        public MainTester(ITestOutputHelper output)
        {
            this._output = output;
        }
        
        [Fact]
        public void Test_Tester_is_Working_As_Expected()
        {
            _output.WriteLine("Test_Tester_is_Working_As_Expected()...");
            Assert.True(true);
            _output.WriteLine("...Test_Tester_is_Working_As_Expected DONE.");
        }

        [Fact]
        public void Test_Evaluate_Prediction_Model_Score()
        {
            _output.WriteLine("Test_Evaluate_Prediction_Model_Score()...");

            var predictor = new Predictor();
            
            predictor.LoadTestData();
            predictor.BuildAndTrainModel();
            var metrics = predictor.EvaluateModelMetrics();
            
            _output.WriteLine($"Model Metrics - Accuracy:{metrics.Accuracy:P2}, AreaUnderRocCurve:{metrics.AreaUnderRocCurve:P2}, F1Score:{metrics.F1Score:P2}");
            
            Assert.InRange(metrics.Accuracy,0.5, 0.95);
            
            _output.WriteLine("...Test_Evaluate_Prediction_Model_Score() DONE.");
        }
        
        [Fact]
        public void Test_Predict_Output_Positive()
        {
            _output.WriteLine("Test_Predict_Output_Positive()...");

            const string INPUT_DATA = "Food was very good";
            
            var predictor = new Predictor();
            
            predictor.LoadTestData();
            predictor.BuildAndTrainModel();

            var predictionTuple = predictor.Predict(INPUT_DATA);
            
            _output.WriteLine($"Prediction for \"{INPUT_DATA}\" : {predictionTuple.prediction} (Probability: {predictionTuple.probability}, Score: {predictionTuple.score})");
            
            Assert.True(predictionTuple.prediction);
            
            _output.WriteLine("...Test_Predict_Output_Positive() DONE.");
        }
        
        [Fact]
        public void Test_Predict_Output_Negative()
        {
            _output.WriteLine("Test_Predict_Output_Negative()...");

            const string INPUT_DATA = "This was quite a horrible meal";
            
            var predictor = new Predictor();
            
            predictor.LoadTestData();
            predictor.BuildAndTrainModel();

            var predictionTuple = predictor.Predict(INPUT_DATA);
            
            _output.WriteLine($"Prediction for \"{INPUT_DATA}\" : {predictionTuple.prediction} (Probability: {predictionTuple.probability}, Score: {predictionTuple.score})");
            
            Assert.False(predictionTuple.prediction);
            
            _output.WriteLine("...Test_Predict_Output_Negative() DONE.");
        }
    }
}