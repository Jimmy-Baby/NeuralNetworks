#include <random>

#pragma warning(push, 0)
#include <MiniDNN.h>
#pragma warning(pop)

using namespace MiniDNN;

using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;

namespace DigitIdentifier
{
	class PrintOutCallback final : public Callback
	{
	public:
		void post_training_batch(const Network* Net, const Matrix& X, const Matrix& Y) override
		{
			const Scalar Loss = Net->get_output()->loss();
			std::cout << "[Epoch " << m_epoch_id << ", batch " << m_batch_id << "] Loss = " << Loss << std::endl;
		}

		void post_training_batch(const Network* Net, const Matrix& X, const IntegerVector& Y) override
		{
			const Scalar Loss = Net->get_output()->loss();
			std::cout << "[Epoch " << m_epoch_id << ", batch " << m_batch_id << "] Loss = " << Loss << std::endl;
		}
	};

	int Main(const int Argc, char* Argv[])
	{
		// Random generator stuff
		std::random_device RandomDevice;
		std::mt19937 RandomGenerator(RandomDevice());
		std::uniform_int_distribution RandomDistribution(1, INT_MAX);

		// Create the truth table for XOR
		Matrix X(2, 4); // 2 input neurons, 4 observations
		Matrix Y(1, 4); // 1 output neuron, 4 observations
		 
		// Populate the input matrix (each column is an observation)
		X << 0, 0, 1, 1,
			0, 1, 0, 1;

		// Populate the output matrix (each column is an observation)
		Y << 0, 0, 0, 1;

		Network DigitNetwork;

		// Input layer (2 neurons) -> Hidden layer (2 neurons, ReLU activation)
		Layer* InputLayer = new FullyConnected<ReLU>(2, 4);

		// Hidden layer (2 neurons) -> Output layer (1 neuron, Sigmoid activation)
		Layer* HiddenLayer1 = new FullyConnected<Sigmoid>(4, 4);

		// Hidden layer (2 neurons) -> Output layer (1 neuron, Sigmoid activation)
		Layer* HiddenLayer2 = new FullyConnected<Sigmoid>(4, 1);

		DigitNetwork.add_layer(InputLayer);
		DigitNetwork.add_layer(HiddenLayer1);
		DigitNetwork.add_layer(HiddenLayer2);

		// Set output layer to use Mean Squared Error for regression
		DigitNetwork.set_output(new RegressionMSE());

		// Create optimizer object
		Adam Optimiser;
		Optimiser.m_lrate = 0.01; // Learning rate

		// Set print out callback
		//PrintOutCallback Callback;
		//DigitNetwork.set_callback(Callback);

		DigitNetwork.init(0, 0.75, RandomDistribution(RandomGenerator));

		const int nlayer = DigitNetwork.num_layers();
		for (int i = 0; i < nlayer; ++i)
		{
			std::cout << "Layer " << i << " parameters:" << std::endl;
			const Layer* CurrentLayer = DigitNetwork.get_layers()[i];
			
			for (const Scalar Parameter : CurrentLayer->get_parameters())
			{
				std::cout << Parameter << ", ";
			}

			std::cout << '\n';
		}

		std::cout << "---------------------------------" << std::endl;

		DigitNetwork.fit(Optimiser, X, Y, 4, 100000, RandomDistribution(RandomGenerator));

		// Obtain prediction -- each column is an observation
		const Matrix Prediction = DigitNetwork.predict(X);

		// Print predictions
		std::cout << "Predictions:" << std::endl << Prediction << std::endl;

		return 0;
	}
}
