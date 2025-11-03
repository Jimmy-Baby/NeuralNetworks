// Precompiled headers
#include "Pch.h"

#include "TinyDnnAdder.h"

namespace TinyDnnAdder
{
	using namespace tiny_cnn;

	namespace
	{
		// Constants for network architecture and training
		constexpr cnn_size_t InputSize = 9;  // 4 bits (a) + 4 bits (b) + 1 bit (carry_in)
		constexpr cnn_size_t OutputSize = 5; // 4 bits (sum) + 1 bit (carry_out)
		constexpr cnn_size_t HiddenLayer1 = 32;
		constexpr cnn_size_t HiddenLayer2 = 16;

		constexpr cnn_size_t BitsPerNumber = 4;
		constexpr cnn_size_t Max4BitValue = 16;
		constexpr cnn_size_t CarryBitIndex = 8;
		constexpr cnn_size_t CarryOutIndex = 4;

		constexpr int BatchSize = 128;
		constexpr int TrainingEpochs = 5 * BatchSize;
		constexpr tiny_cnn::float_t BinaryThreshold = 0.5;

		/// <summary>
		/// Builds a complete dataset for 4-bit addition with carry.
		/// Generates all possible combinations of two 4-bit numbers (0-15) with carry-in (0-1).
		/// Total: 16 * 16 * 2 = 512 samples
		/// </summary>
		/// <returns>Pair of input vectors and corresponding output vectors</returns>
		std::pair<std::vector<vec_t>, std::vector<vec_t>> BuildTrainingDataset()
		{
			std::vector<vec_t> inputs;
			std::vector<vec_t> outputs;

			// Reserve memory upfront for better performance
			constexpr size_t totalSamples = Max4BitValue * Max4BitValue * 2;
			inputs.reserve(totalSamples);
			outputs.reserve(totalSamples);

			// Generate all combinations of 4-bit addition: a + b + carry_in
			for (size_t firstNumber = 0; firstNumber < Max4BitValue; ++firstNumber)
			{
				for (size_t secondNumber = 0; secondNumber < Max4BitValue; ++secondNumber)
				{
					for (size_t carryIn = 0; carryIn <= 1; ++carryIn)
					{
						// Create input vector: [a0, a1, a2, a3, b0, b1, b2, b3, carry_in]
						vec_t inputVector(InputSize, 0.0);

						// Encode first 4-bit number (a)
						for (size_t bitPosition = 0; bitPosition < BitsPerNumber; ++bitPosition)
						{
							inputVector[bitPosition] = static_cast<tiny_cnn::float_t>((firstNumber >> bitPosition) & 1);
						}

						// Encode second 4-bit number (b)
						for (size_t bitPosition = 0; bitPosition < BitsPerNumber; ++bitPosition)
						{
							inputVector[BitsPerNumber + bitPosition] = static_cast<tiny_cnn::float_t>((secondNumber >> bitPosition) & 1);
						}

						// Encode carry-in bit
						inputVector[CarryBitIndex] = static_cast<tiny_cnn::float_t>(carryIn);

						// Calculate expected output: sum and carry-out
						const int totalSum = static_cast<int>(firstNumber + secondNumber + carryIn);
						const int sum4Bit = totalSum & 0xF;       // Lower 4 bits
						const int carryOut = (totalSum >> 4) & 1; // Carry-out bit

						// Create output vector: [sum0, sum1, sum2, sum3, carry_out]
						vec_t outputVector(OutputSize, 0.0);

						// Encode 4-bit sum result
						for (size_t bitPosition = 0; bitPosition < BitsPerNumber; ++bitPosition)
						{
							outputVector[bitPosition] = static_cast<tiny_cnn::float_t>((sum4Bit >> bitPosition) & 1);
						}

						// Encode carry-out bit
						outputVector[CarryOutIndex] = static_cast<tiny_cnn::float_t>(carryOut);

						inputs.emplace_back(std::move(inputVector));
						outputs.emplace_back(std::move(outputVector));
					}
				}
			}

			return { inputs, outputs };
		}

		/// <summary>
		/// Converts a continuous neural network output to binary by thresholding
		/// </summary>
		int ToBinaryBit(const tiny_cnn::float_t value)
		{
			return (value > BinaryThreshold) ? 1 : 0;
		}

		/// <summary>
		/// Converts a 4-bit binary vector to an integer value
		/// </summary>
		int DecodeBinaryVector(const vec_t& vector, const size_t startIndex)
		{
			int result = 0;
			for (size_t bitPosition = 0; bitPosition < BitsPerNumber; ++bitPosition)
			{
				result |= ToBinaryBit(vector[startIndex + bitPosition]) << bitPosition;
			}
			return result;
		}

		/// <summary>
		/// Evaluates the trained network's accuracy on the dataset
		/// </summary>
		void EvaluateAndPrintAccuracy(network<sequential>& neuralNetwork, const std::vector<vec_t>& testInputs, const std::vector<vec_t>& expectedOutputs)
		{
			size_t correctPredictions = 0;

			for (size_t sampleIndex = 0; sampleIndex < testInputs.size(); ++sampleIndex)
			{
				const vec_t prediction = neuralNetwork.predict(testInputs[sampleIndex]);

				// Decode predicted sum and carry-out
				const int predictedSum = DecodeBinaryVector(prediction, 0);
				const int predictedCarryOut = ToBinaryBit(prediction[CarryOutIndex]);

				// Decode expected sum and carry-out
				const int expectedSum = DecodeBinaryVector(expectedOutputs[sampleIndex], 0);
				const int expectedCarryOut = ToBinaryBit(expectedOutputs[sampleIndex][CarryOutIndex]);

				// Check if both sum and carry-out match
				if (predictedSum == expectedSum && predictedCarryOut == expectedCarryOut)
				{
					++correctPredictions;
				}
			}

			const tiny_cnn::float_t accuracy = 100.0 * static_cast<tiny_cnn::float_t>(correctPredictions) / static_cast<tiny_cnn::float_t>(testInputs.size());
			std::printnl();
			std::println("Results:");
			std::println("  Accuracy: {:.2f}% ({}/{} correct)", accuracy, correctPredictions, testInputs.size());
		}

		/// <summary>
		/// Allow the user to enter decimal inputs and view the network's predictions.
		/// </summary>
		void InteractiveTest(network<sequential>& neuralNetwork)
		{
			std::printnl();
			std::println("Interactive test - enter: a b carry_in (format: a/b is 0-15, carry_in is 0 or 1)");
			std::println("Type 'q' to quit.");

			std::string line;
			for (;;)
			{
				std::print("> ");

				if (!std::getline(std::cin, line))
				{
					break;
				}

				std::istringstream firstToken(line);
				std::string token;

				if (!(firstToken >> token))
				{
					continue;
				}

				if (token == "q" || token == "Q" || token == "quit" || token == "QUIT")
				{
					break;
				}

				int a, b, carryIn;
				std::istringstream iss(line);
				if (!(iss >> a >> b >> carryIn))
				{
					std::println("Please enter three integers: a  b  carry_in");
					continue;
				}

				if (a < 0 || a >= static_cast<int>(Max4BitValue) || b < 0 || b >= static_cast<int>(Max4BitValue) || (carryIn != 0 && carryIn != 1))
				{
					std::println("Out of range. a/b must be 0..15 and carry_in must be 0 or 1.");
					continue;
				}

				// Build input vector
				vec_t input(InputSize, 0.0);
				for (size_t index = 0; index < BitsPerNumber; ++index)
				{
					input[index] = static_cast<tiny_cnn::float_t>((a >> index) & 1);
					input[BitsPerNumber + index] = static_cast<tiny_cnn::float_t>((b >> index) & 1);
				}

				input[CarryBitIndex] = static_cast<tiny_cnn::float_t>(carryIn);

				// Predict
				const vec_t y = neuralNetwork.predict(input);
				const int predictedSum = DecodeBinaryVector(y, 0);
				const int predictedCarry = ToBinaryBit(y[CarryOutIndex]);
				const int predictedTotal = (predictedCarry << 4) | predictedSum; // 0..31

				// Expected
				const int expectedTotal = a + b + carryIn;
				const int expectedSum = expectedTotal & 0xF;
				const int expectedCarry = (expectedTotal >> 4) & 1;

				std::println("Input: a = '{}' b = '{}' carry_in = '{}'", a, b, carryIn);

				std::println("Predicted:");
				std::println("	sum: {}", predictedSum);
				std::println("	carry_out: {}", predictedCarry);
				std::println("	total: {}", predictedTotal);
				std::println("	bits: {} {} {} {}", ToBinaryBit(y[0]), ToBinaryBit(y[1]), ToBinaryBit(y[2]), ToBinaryBit(y[3]));

				std::println("Expected:");
				std::println("	sum: {}", expectedSum);
				std::println("	carry_out: {}", expectedCarry);
				std::println("	total: {}", expectedTotal);
				std::println("	bits: {} {} {} {}", (expectedSum >> 0) & 1, (expectedSum >> 1) & 1, (expectedSum >> 2) & 1, (expectedSum >> 3) & 1);
			}
		}
	}

	void Run()
	{
		std::println("Training 4 bit Adder Net");

		// Build complete training dataset (512 samples)
		auto [trainingInputs, trainingOutputs] = BuildTrainingDataset();

		// Input: 9 (two 4-bit numbers + carry-in)
		// Hidden layers: 32 -> 16
		// Output: 5 (4-bit sum + carry-out)
		network<sequential> neuralNetwork;
		neuralNetwork << fully_connected_layer<activation::relu>(InputSize, HiddenLayer1)
			<< fully_connected_layer<activation::relu>(HiddenLayer1, HiddenLayer2)
			<< fully_connected_layer<activation::sigmoid>(HiddenLayer2, OutputSize);

		adam optimiser;

		// Track epochs
		size_t epochCount = 0;

		// Progress display for monitoring training
		progress_display progressDisplay(BatchSize);

		auto onEpochComplete = [&]
		{
			++epochCount;
			++progressDisplay;

			// Print out current error/cost every BatchSize epochs
			if (epochCount % BatchSize == 0)
			{
				const tiny_cnn::float_t loss = neuralNetwork.get_loss<mse>(trainingInputs, trainingOutputs);
				std::printnl();
				std::println("Epoch {}/{}: Loss = {:.6f}", epochCount, TrainingEpochs, loss);
				progressDisplay.restart(BatchSize);
			}
		};

		auto onMinibatchComplete = [&]
		{
		};

		// Train the network using Mean Squared Error loss function
		// MSE is appropriate for regression-like tasks with continuous outputs (0-1 range)
		neuralNetwork.fit<mse>(optimiser, trainingInputs, trainingOutputs, BatchSize, TrainingEpochs, onMinibatchComplete, onEpochComplete);

		// Evaluate final accuracy on the training set
		EvaluateAndPrintAccuracy(neuralNetwork, trainingInputs, trainingOutputs);

		// Let the user test the network
		InteractiveTest(neuralNetwork);
	}
}
