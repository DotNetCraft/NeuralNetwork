using System;
using DotNetCraft.NeuralNetwork.Core;

namespace DotNetCraft.NeuralNetwork.Domain.ActivationFunctions
{
    public class TanhActivationFunction : IActivationFunction
    {
        #region Implementation of IActivationFunction

        public double Calculate(double x)
        {
            var temp = Math.Exp(2 * x);
            var result = (temp - 1) / (temp + 1);
            return result;
        }

        public double CalculatePrime(double x)
        {
            var result = 1 - x * x;
            return result;
        }

        #endregion
    }
}
