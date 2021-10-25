namespace DotNetCraft.NeuralNetwork.Domain
{
    public class TrainDetails
    {
        public double MSE { get; set; } = 0.0;
        public double RootMSE { get; set; } = 0.0;

        public void Reset()
        {
            MSE = 0;
            RootMSE = 0;
        }

        #region Overrides of Object

        /// <summary>Returns a string that represents the current object.</summary>
        /// <returns>A string that represents the current object.</returns>
        public override string ToString()
        {
            return $"MSE: {MSE:F10}; RootMSE: {RootMSE:F10}";
        }

        #endregion
    }
}
