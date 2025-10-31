// Precompiled headers
#include "Pch.h"

#include "FlexNetwork.h"
#include "Utility/Matrix.h"
#include "Utility/Layer.h"

namespace FlexNetwork
{
	struct FlexNetworkArchitecture
	{
		size_t InputLayerSize;
		std::vector<size_t> HiddenLayerSizes;
		size_t OutputLayerSize;
	};

	class FlexNetwork
	{
	public:
		explicit FlexNetwork(const FlexNetworkArchitecture& architecture, const bool randomise = true)
		{
			if (architecture.HiddenLayerSizes.empty())
			{
				OutputLayer = Layer(architecture.InputLayerSize, architecture.OutputLayerSize, randomise);
			}
			else
			{
				HiddenLayers.emplace_back(architecture.InputLayerSize, architecture.HiddenLayerSizes[0], randomise);

				for (size_t hiddenLayerIndex = 1; hiddenLayerIndex < architecture.HiddenLayerSizes.size(); ++hiddenLayerIndex)
				{
					HiddenLayers.emplace_back(HiddenLayers[hiddenLayerIndex - 1], architecture.HiddenLayerSizes[hiddenLayerIndex], randomise);
				}

				OutputLayer = Layer(HiddenLayers.back(), architecture.OutputLayerSize, randomise);
			}
		}

		bool operator==(const FlexNetwork& rhs) const
		{
			if (HiddenLayers.size() != rhs.HiddenLayers.size())
			{
				return false;
			}

			for (size_t hiddenLayerIndex = 0; hiddenLayerIndex < HiddenLayers.size(); ++hiddenLayerIndex)
			{
				if (HiddenLayers[hiddenLayerIndex].GetWeights().GetColumnCount() != rhs.HiddenLayers[hiddenLayerIndex].GetWeights().GetColumnCount())
				{
					return false;
				}

				if (HiddenLayers[hiddenLayerIndex].GetWeights().GetRowCount() != rhs.HiddenLayers[hiddenLayerIndex].GetWeights().GetRowCount())
				{
					return false;
				}
			}

			if (OutputLayer.GetWeights().GetColumnCount() != rhs.OutputLayer.GetWeights().GetColumnCount())
			{
				return false;
			}

			if (OutputLayer.GetWeights().GetRowCount() != rhs.OutputLayer.GetWeights().GetRowCount())
			{
				return false;
			}

			return true;
		}

		bool operator!=(const FlexNetwork& rhs) const
		{
			return !operator==(rhs);
		}

		void ForwardToModel(const Matrix& inputs)
		{
			if (HiddenLayers.empty())
			{
				OutputLayer.Activate(inputs);
			}
			else
			{
				HiddenLayers[0].Activate(inputs);

				for (size_t hiddenLayerIndex = 1; hiddenLayerIndex < HiddenLayers.size(); ++hiddenLayerIndex)
				{
					HiddenLayers[hiddenLayerIndex].Activate(HiddenLayers[hiddenLayerIndex - 1]);
				}

				OutputLayer.Activate(HiddenLayers.back());
			}
		}

		[[nodiscard]] const Matrix& GetOutputs() const
		{
			return OutputLayer.GetActivations();
		}

		void Print(const bool printHiddenLayers = true, const bool printOutputLayer = true) const
		{
			size_t layerIndex = 0;

			if (printHiddenLayers)
			{
				for (; layerIndex < HiddenLayers.size(); ++layerIndex)
				{
					HiddenLayers[layerIndex].PrintValues(static_cast<int32_t>(layerIndex));
					printf("\n");
				}
			}

			if (printOutputLayer)
			{
				OutputLayer.PrintValues(static_cast<int32_t>(layerIndex));
			}
		}

		float CalculateCost(const Matrix& inputs, const Matrix& expectedOutputs)
		{
			float totalCost = 0.0f;

			for (size_t sampleIndex = 0; sampleIndex < inputs.GetRowCount(); ++sampleIndex)
			{
				// Get the current training sample
				Matrix trainingSample = inputs.SubMatrix(sampleIndex, 0, 1, inputs.GetColumnCount());

				// Use the model to evaluate an output
				ForwardToModel(trainingSample);
				const Matrix modelPredictions = OutputLayer.GetActivations();

				for (size_t outputIndex = 0; outputIndex < modelPredictions.GetColumnCount(); ++outputIndex)
				{
					// Subtract the correct output from the model's predicted output to get the amount it was 'off' by
					const float modelPrediction = modelPredictions.At(0, outputIndex);
					const float expectedOutput = expectedOutputs.At(sampleIndex, outputIndex);
					const float predictionError = modelPrediction - expectedOutput;

					// Square the error to get a positive number, and add it to the result
					totalCost += predictionError * predictionError;
				}
			}

			// Average out all the accumulated error over samples and outputs
			const float denominator = static_cast<float>(inputs.GetRowCount()) * static_cast<float>(expectedOutputs.GetColumnCount());
			if (denominator > 0.0f)
			{
				totalCost /= denominator;
			}

			return totalCost;
		}

		// Backpropagation-based training
		void TrainBackprop(const Matrix& trainingInputs, const Matrix& trainingOutputs, size_t epochs = 5000, float learningRate = 0.1f, size_t logInterval = 100)
		{
			const size_t sampleCount = trainingInputs.GetRowCount();
			const size_t inputSize = trainingInputs.GetColumnCount();
			const size_t outputSize = trainingOutputs.GetColumnCount();

			// Pre-allocate gradient accumulators matching network topology
			std::vector<Matrix> weightGradientsForHiddenLayers;
			std::vector<Matrix> biasGradientsForHiddenLayers;
			weightGradientsForHiddenLayers.reserve(HiddenLayers.size());
			biasGradientsForHiddenLayers.reserve(HiddenLayers.size());

			for (auto& layer : HiddenLayers)
			{
				weightGradientsForHiddenLayers.emplace_back(layer.GetWeights().GetRowCount(), layer.GetWeights().GetColumnCount(), false);
				biasGradientsForHiddenLayers.emplace_back(1, layer.GetBiases().GetColumnCount(), false);
			}

			Matrix weightGradientsForOutputLayer(OutputLayer.GetWeights().GetRowCount(), OutputLayer.GetWeights().GetColumnCount(), false);
			Matrix biasGradientsForOutputLayer(1, OutputLayer.GetBiases().GetColumnCount(), false);

			// Deltas per layer (row vector)
			std::vector<Matrix> layerDeltas;
			layerDeltas.reserve(HiddenLayers.size());
			for (auto& layer : HiddenLayers)
			{
				layerDeltas.emplace_back(1, layer.GetBiases().GetColumnCount(), false);
			}

			Matrix outputLayerDelta(1, outputSize, false);
			const float outputScalingFactor = 2.0f / static_cast<float>(outputSize);

			for (size_t currentEpoch = 1; currentEpoch <= epochs; ++currentEpoch)
			{
				// Zero gradients
				for (auto& gradient : weightGradientsForHiddenLayers)
				{
					const size_t totalElements = gradient.GetRowCount() * gradient.GetColumnCount();
					std::memset(gradient.GetData(), 0, totalElements * sizeof(float));
				}

				for (auto& gradient : biasGradientsForHiddenLayers)
				{
					const size_t totalElements = gradient.GetRowCount() * gradient.GetColumnCount();
					std::memset(gradient.GetData(), 0, totalElements * sizeof(float));
				}
				std::memset(weightGradientsForOutputLayer.GetData(), 0, weightGradientsForOutputLayer.GetRowCount() * weightGradientsForOutputLayer.GetColumnCount() * sizeof(float));
				std::memset(biasGradientsForOutputLayer.GetData(), 0, biasGradientsForOutputLayer.GetRowCount() * biasGradientsForOutputLayer.GetColumnCount() * sizeof(float));

				// Accumulate gradients across all samples
				for (size_t sampleIndex = 0; sampleIndex < sampleCount; ++sampleIndex)
				{
					Matrix currentInput = trainingInputs.SubMatrix(sampleIndex, 0, 1, inputSize);
					Matrix expectedOutput = trainingOutputs.SubMatrix(sampleIndex, 0, 1, outputSize);

					// Forward pass stores activations in layers
					ForwardToModel(currentInput);

					// Output delta: (a - y) * sigmoid'(z) ; sigmoid'(z) = a*(1-a)
					for (size_t outputIndex = 0; outputIndex < outputSize; ++outputIndex)
					{
						const float activation = OutputLayer.GetActivations().At(0, outputIndex);
						const float difference = activation - expectedOutput.At(0, outputIndex);
						outputLayerDelta.At(0, outputIndex) = outputScalingFactor * difference * activation * (1.0f - activation);
					}

					// Gradients for output layer weights and biases
					if (HiddenLayers.empty())
					{
						// Use input x as previous activation
						for (size_t rowIndex = 0; rowIndex < weightGradientsForOutputLayer.GetRowCount(); ++rowIndex)
						{
							for (size_t columnIndex = 0; columnIndex < weightGradientsForOutputLayer.GetColumnCount(); ++columnIndex)
							{
								weightGradientsForOutputLayer.At(rowIndex, columnIndex) += currentInput.At(0, rowIndex) * outputLayerDelta.At(0, columnIndex);
							}
						}
					}
					else
					{
						const Matrix& previousLayerActivations = HiddenLayers.back().GetActivations();
						for (size_t rowIndex = 0; rowIndex < weightGradientsForOutputLayer.GetRowCount(); ++rowIndex)
						{
							const float previousActivation = previousLayerActivations.At(0, rowIndex);
							for (size_t columnIndex = 0; columnIndex < weightGradientsForOutputLayer.GetColumnCount(); ++columnIndex)
							{
								weightGradientsForOutputLayer.At(rowIndex, columnIndex) += previousActivation * outputLayerDelta.At(0, columnIndex);
							}
						}
					}
					// Bias gradients for output layer
					for (size_t columnIndex = 0; columnIndex < biasGradientsForOutputLayer.GetColumnCount(); ++columnIndex)
					{
						biasGradientsForOutputLayer.At(0, columnIndex) += outputLayerDelta.At(0, columnIndex);
					}

					// Backpropagate to hidden layers
					Matrix* nextLayerDelta = &outputLayerDelta;
					for (size_t layerIndex = HiddenLayers.size(); layerIndex-- > 0;)
					{
						Layer& currentLayer = HiddenLayers[layerIndex];

						// Compute delta for this hidden layer: nextDelta (1 x n_next) * W_next^T (n_next x n_l) =1 x n_l
						Matrix& currentLayerDelta = layerDeltas[layerIndex];
						const Matrix& currentLayerActivations = currentLayer.GetActivations();

						// Determine W_next
						const Matrix& nextLayerWeights = (layerIndex == HiddenLayers.size() - 1) ? OutputLayer.GetWeights() : HiddenLayers[layerIndex + 1].GetWeights();

						for (size_t hiddenNeuronIndex = 0; hiddenNeuronIndex < currentLayerActivations.GetColumnCount(); ++hiddenNeuronIndex)
						{
							float weightedSum = 0.0f;
							for (size_t nextNeuronIndex = 0; nextNeuronIndex < nextLayerDelta->GetColumnCount(); ++nextNeuronIndex)
							{
								// Wnext has shape (n_l x n_next)
								weightedSum += nextLayerDelta->At(0, nextNeuronIndex) * nextLayerWeights.At(hiddenNeuronIndex, nextNeuronIndex);
							}
							const float activationValue = currentLayerActivations.At(0, hiddenNeuronIndex);
							currentLayerDelta.At(0, hiddenNeuronIndex) = weightedSum * activationValue * (1.0f - activationValue);
						}

						// Gradients for this hidden layer weights and biases
						if (layerIndex == 0)
						{
							// Previous activation is input x
							for (size_t rowIndex = 0; rowIndex < currentLayer.GetWeights().GetRowCount(); ++rowIndex)
							{
								for (size_t columnIndex = 0; columnIndex < currentLayer.GetWeights().GetColumnCount(); ++columnIndex)
								{
									weightGradientsForHiddenLayers[layerIndex].At(rowIndex, columnIndex) += currentInput.At(0, rowIndex) * currentLayerDelta.At(0, columnIndex);
								}
							}
						}
						else
						{
							const Matrix& previousHiddenLayerActivations = HiddenLayers[layerIndex - 1].GetActivations();
							for (size_t rowIndex = 0; rowIndex < currentLayer.GetWeights().GetRowCount(); ++rowIndex)
							{
								const float previousActivation = previousHiddenLayerActivations.At(0, rowIndex);
								for (size_t columnIndex = 0; columnIndex < currentLayer.GetWeights().GetColumnCount(); ++columnIndex)
								{
									weightGradientsForHiddenLayers[layerIndex].At(rowIndex, columnIndex) += previousActivation * currentLayerDelta.At(0, columnIndex);
								}
							}
						}
						for (size_t columnIndex = 0; columnIndex < currentLayer.GetBiases().GetColumnCount(); ++columnIndex)
						{
							biasGradientsForHiddenLayers[layerIndex].At(0, columnIndex) += currentLayerDelta.At(0, columnIndex);
						}

						// Set nextDelta for the next iteration backward
						nextLayerDelta = &currentLayerDelta;
					}
				}

				// Average gradients over samples
				const float inverseSampleCount = 1.0f / static_cast<float>(sampleCount);
				for (auto& gradient : weightGradientsForHiddenLayers)
				{
					for (size_t rowIndex = 0; rowIndex < gradient.GetRowCount(); ++rowIndex)
					{
						for (size_t columnIndex = 0; columnIndex < gradient.GetColumnCount(); ++columnIndex)
						{
							gradient.At(rowIndex, columnIndex) *= inverseSampleCount;
						}
					}
				}
				for (auto& gradient : biasGradientsForHiddenLayers)
				{
					for (size_t columnIndex = 0; columnIndex < gradient.GetColumnCount(); ++columnIndex)
					{
						gradient.At(0, columnIndex) *= inverseSampleCount;
					}
				}
				for (size_t rowIndex = 0; rowIndex < weightGradientsForOutputLayer.GetRowCount(); ++rowIndex)
				{
					for (size_t columnIndex = 0; columnIndex < weightGradientsForOutputLayer.GetColumnCount(); ++columnIndex)
					{
						weightGradientsForOutputLayer.At(rowIndex, columnIndex) *= inverseSampleCount;
					}
				}
				for (size_t columnIndex = 0; columnIndex < biasGradientsForOutputLayer.GetColumnCount(); ++columnIndex)
				{
					biasGradientsForOutputLayer.At(0, columnIndex) *= inverseSampleCount;
				}

				// Gradient descent update: W -= lr * grad, b -= lr * grad
				for (size_t layerIndex = 0; layerIndex < HiddenLayers.size(); ++layerIndex)
				{
					Layer& layer = HiddenLayers[layerIndex];
					for (size_t rowIndex = 0; rowIndex < layer.GetWeights().GetRowCount(); ++rowIndex)
					{
						for (size_t columnIndex = 0; columnIndex < layer.GetWeights().GetColumnCount(); ++columnIndex)
						{
							layer.GetWeights().At(rowIndex, columnIndex) -= learningRate * weightGradientsForHiddenLayers[layerIndex].At(rowIndex, columnIndex);
						}
					}
					for (size_t columnIndex = 0; columnIndex < layer.GetBiases().GetColumnCount(); ++columnIndex)
					{
						layer.GetBiases().At(0, columnIndex) -= learningRate * biasGradientsForHiddenLayers[layerIndex].At(0, columnIndex);
					}
				}

				for (size_t rowIndex = 0; rowIndex < OutputLayer.GetWeights().GetRowCount(); ++rowIndex)
				{
					for (size_t columnIndex = 0; columnIndex < OutputLayer.GetWeights().GetColumnCount(); ++columnIndex)
					{
						OutputLayer.GetWeights().At(rowIndex, columnIndex) -= learningRate * weightGradientsForOutputLayer.At(rowIndex, columnIndex);
					}
				}
				for (size_t columnIndex = 0; columnIndex < OutputLayer.GetBiases().GetColumnCount(); ++columnIndex)
				{
					OutputLayer.GetBiases().At(0, columnIndex) -= learningRate * biasGradientsForOutputLayer.At(0, columnIndex);
				}

				if (logInterval != 0 && (currentEpoch % logInterval == 0 || currentEpoch == 1 || currentEpoch == epochs))
				{
					const float cost = CalculateCost(trainingInputs, trainingOutputs);
					printf("Epoch %zu/%zu - Cost: %.6f\n", currentEpoch, epochs, cost);
					if (cost < 1e-6f)
					{
						printf("Converged.\n");
						break;
					}
				}
			}
		}

	private:
		std::vector<Layer> HiddenLayers;
		Layer OutputLayer;
	};

	void Run()
	{
		// Architecture: 9 inputs (a[4], b[4], carry_in) -> hidden -> 5 outputs (sum[4], carry_out)
		const FlexNetworkArchitecture architecture
		{
			.InputLayerSize = 9,
			.HiddenLayerSizes = { 32, 16 },
			.OutputLayerSize = 5
		};

		// Build full 4-bit adder dataset: 16*16*2 = 512 rows
		Matrix trainingInputs(512, 9);
		Matrix trainingOutputs(512, 5);
		size_t datasetRowIndex = 0;

		for (size_t firstNumber = 0; firstNumber < 16; ++firstNumber)
		{
			for (size_t secondNumber = 0; secondNumber < 16; ++secondNumber)
			{
				for (size_t carryInBit = 0; carryInBit <= 1; ++carryInBit)
				{
					for (size_t bitIndex = 0; bitIndex < 4; ++bitIndex)
					{
						trainingInputs.At(datasetRowIndex, bitIndex) = static_cast<float>((firstNumber >> bitIndex) & 1);
					}

					for (size_t bitIndex = 0; bitIndex < 4; ++bitIndex)
					{
						trainingInputs.At(datasetRowIndex, 4 + bitIndex) = static_cast<float>((secondNumber >> bitIndex) & 1);
					}

					trainingInputs.At(datasetRowIndex, 8) = static_cast<float>(carryInBit);

					const int totalSum = static_cast<int>(firstNumber + secondNumber + carryInBit);
					const int fourBitSum = totalSum & 0xF;
					const int carryOutBit = (totalSum >> 4) & 1;

					for (size_t bitIndex = 0; bitIndex < 4; ++bitIndex)
					{
						trainingOutputs.At(datasetRowIndex, bitIndex) = static_cast<float>((fourBitSum >> bitIndex) & 1);
					}

					trainingOutputs.At(datasetRowIndex, 4) = static_cast<float>(carryOutBit);
					++datasetRowIndex;
				}
			}
		}

		printf("\nTraining 4-Bit Full Adder\n");
		printf("Dataset size: %zu samples\n\n", trainingInputs.GetRowCount());

		// Train using backpropagation
		FlexNetwork model(architecture, true);
		constexpr size_t epochs = 1000000;
		constexpr float learningRate = 0.1f;
		model.TrainBackprop(trainingInputs, trainingOutputs, epochs, learningRate, 200);

		printf("\nPost-trained Model (weights/biases):\n\n");
		model.Print(true, true);

		printf("------------------------------\n");

		// Evaluate accuracy on the full dataset
		{
			printf("\nTesting 4-Bit Full Adder\n");
			printf("Format: A + B + Cin = Sum (Cout)\n\n");
			size_t correctPredictions = 0;
			datasetRowIndex = 0;
			
			for (size_t firstNumber = 0; firstNumber < 16; ++firstNumber)
			{
				for (size_t secondNumber = 0; secondNumber < 16; ++secondNumber)
				{
					for (size_t carryInBit = 0; carryInBit <= 1; ++carryInBit)
					{
						Matrix inputVector(1, 9);

						for (size_t bitIndex = 0; bitIndex < 4; ++bitIndex)
						{
							inputVector.At(0, bitIndex) = static_cast<float>((firstNumber >> bitIndex) & 1);
						}

						for (size_t bitIndex = 0; bitIndex < 4; ++bitIndex)
						{
							inputVector.At(0, 4 + bitIndex) = static_cast<float>((secondNumber >> bitIndex) & 1);
						}

						inputVector.At(0, 8) = static_cast<float>(carryInBit);

						model.ForwardToModel(inputVector);
						const Matrix& networkOutputs = model.GetOutputs();

						int predictedSum = 0;

						for (size_t bitIndex = 0; bitIndex < 4; ++bitIndex)
						{
							const int bitValue = networkOutputs.At(0, bitIndex) > 0.5f ? 1 : 0;
							predictedSum |= (bitValue << bitIndex);
						}

						const int predictedCarryOut = networkOutputs.At(0, 4) > 0.5f ? 1 : 0;
						const int expectedTotal = static_cast<int>(firstNumber + secondNumber + carryInBit);
						const int expectedSum = expectedTotal & 0xF;
						const int expectedCarryOut = (expectedTotal >> 4) & 1;

						const bool isPredictionCorrect = (predictedSum == expectedSum) && (predictedCarryOut == expectedCarryOut);

						if (isPredictionCorrect)
						{
							++correctPredictions;
						}

						const char* statusLabel = isPredictionCorrect ? "(Good)" : "(Bad )";
						
						printf("%s A=%2zu B=%2zu Cin=%zu => Sum=%2d (Cout=%d) ", 
							statusLabel, firstNumber, secondNumber, carryInBit, predictedSum, predictedCarryOut);
						
						printf("[raw: %.3f %.3f %.3f %.3f | %.3f] ", 
							networkOutputs.At(0, 0), networkOutputs.At(0, 1), 
							networkOutputs.At(0, 2), networkOutputs.At(0, 3), 
							networkOutputs.At(0, 4));
						
						printf("Expected: %2d (%d)\n", expectedSum, expectedCarryOut);
						
						++datasetRowIndex;
					}
				}
			}

			printf("\nAccuracy: %.2f%% (%zu/%zu)\n", 
				100.0f * static_cast<float>(correctPredictions) / static_cast<float>(trainingInputs.GetRowCount()), 
				correctPredictions, 
				trainingInputs.GetRowCount());
		}
	}
}
