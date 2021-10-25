namespace DotNetCraft.NeuralNetwork.Core
{
    public interface IWeightBuilder
    {
        double BuildWeight();
        double[] BuildWeights(int count);
    }
}
