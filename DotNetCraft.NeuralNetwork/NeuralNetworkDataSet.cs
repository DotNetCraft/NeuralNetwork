using System.Collections.Generic;

namespace DotNetCraft.NeuralNetwork
{
    public class NeuralNetworkDataSet
    {
        private readonly int _featureSize;
        private readonly int _labelSize;
        public List<double> Features { get; set; }
        public List<double> Labels { get; set; }
        public int SampleCount { get; private set; }

        public NeuralNetworkDataSet(int sampleCount, int featureSize, int labelSize)
        {
            _featureSize = featureSize;
            _labelSize = labelSize;
            Features = new List<double>(sampleCount * featureSize);
            Labels = new List<double>(sampleCount * labelSize);
        }


        public void AddSample(double[] sampleFeatures, double[] sampleLabels)
        {
            Features.AddRange(sampleFeatures);
            Labels.AddRange(sampleLabels);
            SampleCount++;
        }

        public double[] GetFeatures(int sampleIndex)
        {
            var position = sampleIndex * _featureSize;
            var result = Features.GetRange(position, _featureSize).ToArray();
            return result;
        }

        public double[] GetLabels(int sampleIndex)
        {
            var position = sampleIndex * _labelSize;
            var result = Labels.GetRange(position, _labelSize).ToArray();
            return result;
        }
    }
}
