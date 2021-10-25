using DotNetCraft.NeuralNetwork.Core;

namespace DotNetCraft.NeuralNetwork.Domain.ActivationFunctions
{
    public class NoneActivationFunction : IActivationFunction
    {
        #region Implementation of IActivationFunction

        public double Calculate(double x)
        {
            return x;
        }

        public double CalculatePrime(double x)
        {
            return 1;
        }

        #endregion
    }
}
